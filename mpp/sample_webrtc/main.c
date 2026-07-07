/**
 * @file main.c
 * @brief WebRTC LAN camera demo for K230 RT-Smart
 *
 * Architecture overview:
 *
 *   ┌──────────┐    bind     ┌──────────┐    bind     ┌──────────┐
 *   │  VICAP   │───────────> │    VO    │            │  VENC   │
 *   │ (camera) │ CHN0        │ (LCD/    │            │ (H.264/ │
 *   │          │───────────> │  HDMI)   │            │ H.265   │
 *   │          │ CHN1        └──────────┘            │ encode) │
 *   └──────────┘                                        │
 *                                                       │ H.264/H.265 frames
 *                                                       ▼
 *   ┌────────────────────────────────────────────────────────────┐
 *   │                    WebRTC (libpeer)                         │
 *   │  peer_connection_task ── ICE/DTLS ──> browser              │
 *   └────────────────────────────────────────────────────────────┘
 *                                                       ▲
 *   ┌────────────────────────────────────────────────────────────┐
 *   │                    HTTP signaling                          │
 *   │  GET /offer  ──> create SDP offer ──> browser              │
 *   │  POST /answer <── set remote SDP  <── browser              │
 *   └────────────────────────────────────────────────────────────┘
 *
 * Thread model:
 *   - Main thread:      signal wait loop, then orchestrates shutdown
 *   - http_server:      accepts connections, parses HTTP, calls on_http_request
 *   - peer_connection:  runs peer_connection_loop() at 1ms interval (ICE/DTLS)
 *   - venc_stream:      polls VENC for H.264 frames, sends via WebRTC
 *
 * Shutdown sequence (triggered by SIGINT):
 *   1. g_exit_requested = 1  (signal handler, safe for async-signal-safe)
 *   2. Main loop exits, sets g_interrupted = 1  (tells threads to stop)
 *   3. http_server_stop()    (closes server fd, joins http thread)
 *   4. pthread_join(venc)    (waits for encode thread to finish)
 *   5. pthread_join(peer)    (waits for peer connection thread to finish)
 *   6. Free resources, destroy peer connection, deinit MPP pipeline
 */

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <unistd.h>

#include "peer.h"
#include "http_server.h"
#include "web_page.h"
#include "mpp_pipeline.h"

#include "mpi_sys_api.h"
#include "mpi_venc_api.h"
#include "k_venc_comm.h"

/* ── Global state ────────────────────────────────────────────────── */

/**
 * g_interrupted: Set by main() after the signal wait loop exits.
 * Read by worker threads (peer_connection_task, venc_stream_task)
 * to know when to stop. Must be set AFTER the main loop exits
 * to avoid racing with the signal handler.
 */
int g_interrupted = 0;

/**
 * g_exit_requested: Set by the SIGINT signal handler.
 * Only read by the main thread's sleep loop.
 * Using volatile because it's written from a signal handler
 * and read from a different context.
 */
static volatile int g_exit_requested = 0;

/**
 * g_pc: The single PeerConnection instance.
 *
 * Drain mechanism: Before close+create_offer, the HTTP thread waits
 * until both worker threads have exited g_pc internals (in-use flags
 * are 0). close() sets pc->state=CLOSED so the next loop/send_video
 * call skips, but a call already past the state check may still be
 * using SRTP/DTLS internals. The drain guarantees no one is inside
 * g_pc before we destroy and rebuild it.
 *
 * Flags are volatile because they are written by worker threads and
 * read by the HTTP thread. This is not a mutex — there remains a
 * tiny window between setting the flag and entering pc internals —
 * but the drain eliminates the long window (close → create_offer
 * happens while another thread is mid-call), which is the one that
 * causes use-after-free crashes.
 */
PeerConnection* g_pc = NULL;

/** In-use flags: set to 1 while a worker thread is inside g_pc internals,
 *  cleared when it exits. The HTTP thread drains (busy-waits) on these
 *  after peer_connection_close() before calling create_offer(). */
static volatile int g_peer_in_pc = 0;
static volatile int g_venc_in_pc = 0;

/**
 * g_state: Current ICE connection state.
 *
 * Written from peer_connection_task (via onconnectionstatechange callback),
 * read from http_server thread and venc_stream_task.
 * Marked volatile as a minimal safety measure.
 */
static volatile PeerConnectionState g_state = PEER_CONNECTION_CLOSED;

/* ── SDP offer / ICE candidate synchronization ──────────────────── */

/** Mutex + condvar for synchronizing offer creation with ICE gathering.
 *  Flow: peer_connection_create_offer() → onicecandidate() → signal condvar
 *        ← on_http_request() waits on condvar until offer is ready
 */
static pthread_mutex_t g_offer_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_offer_cond = PTHREAD_COND_INITIALIZER;
static char* g_offer_sdp = NULL;  /**< Owned by offer flow, freed on next offer or shutdown */
static int g_offer_ready = 0;     /**< Flag: 1 when ICE gathering is complete */

/* ── H.264 SPS/PPS cache ─────────────────────────────────────────── */

/**
 * WebRTC requires SPS/PPS NAL units to precede every I-frame.
 * The K230 VENC emits SPS/PPS as a separate K_VENC_HEADER pack,
 * typically once at stream start. We cache it and prepend it
 * to every I-frame we send.
 */
static uint8_t* g_sps_pps_buf = NULL;  /**< Cached SPS+PPS NAL units */
static size_t g_sps_pps_size = 0;       /**< Size of cached SPS+PPS */

/* ── Callbacks ────────────────────────────────────────────────────── */

/** Called by libpeer when ICE connection state changes.
 *  Updates g_state for other threads to check. */
static void onconnectionstatechange(PeerConnectionState state, void* data) {
  printf("State: %s\n", peer_connection_state_to_string(state));
  g_state = state;
}

/** Called by libpeer when ICE gathering produces a candidate.
 *  Stores the complete SDP (offer + candidates) and signals the
 *  HTTP handler thread that the offer is ready. */
static void onicecandidate(char* sdp, void* userdata) {
  pthread_mutex_lock(&g_offer_mutex);
  /* Free any previous offer SDP */
  if (g_offer_sdp) {
    free(g_offer_sdp);
    g_offer_sdp = NULL;
  }
  g_offer_sdp = strdup(sdp);
  g_offer_ready = 1;
  pthread_cond_signal(&g_offer_cond);
  pthread_mutex_unlock(&g_offer_mutex);
}

/** SIGINT handler: sets flag for main loop to exit.
 *  Only sets g_exit_requested; does NOT set g_interrupted here
 *  because worker threads should continue running until main()
 *  orchestrates an orderly shutdown. */
static void signal_handler(int sig) {
  g_exit_requested = 1;
}

/* ── Frame sending logic ─────────────────────────────────────────── */

/**
 * Send an encoded VENC frame to the WebRTC peer.
 *
 * H.264/H.265 over RTP requires parameter sets (SPS/PPS or VPS/SPS/PPS)
 * before each I-frame for the decoder to initialize. The K230 VENC emits
 * them as a separate K_VENC_HEADER pack (usually once at stream start),
 * so we:
 *   1. Cache any K_VENC_HEADER packs we see
 *   2. Prepend cached parameter sets to every I-frame
 *   3. Send P-frames as-is
 */
static void send_venc_frame_to_webrtc(const uint8_t* data, size_t size,
                                        k_venc_pack_type type, uint64_t pts) {
  /* Cache parameter sets regardless of connection state,
     so they are available when the first I-frame is sent */
  if (type == K_VENC_HEADER) {
    if (g_sps_pps_buf) free(g_sps_pps_buf);
    g_sps_pps_buf = (uint8_t*)malloc(size);
    if (g_sps_pps_buf) {
      memcpy(g_sps_pps_buf, data, size);
      g_sps_pps_size = size;
    }
    return;
  }

  /* Only send video data if we have an active WebRTC connection.
   * Mark in-use BEFORE the state check so the reconnect path (GET /offer)
   * sees us and waits. If we checked state first, close() could happen
   * between the check and setting the flag — the drain would miss us. */
  g_venc_in_pc = 1;
  if (g_state != PEER_CONNECTION_COMPLETED && g_state != PEER_CONNECTION_CONNECTED) {
    g_venc_in_pc = 0;
    return;
  }

  /* I-frame: send parameter sets then I-frame as two separate video sends.
   * WebRTC/RTP handles NAL units independently — no need
   * to concatenate parameter sets + I-frame into a single buffer.
   * This avoids a per-I-frame malloc/free. */
  if (type == K_VENC_I_FRAME && g_sps_pps_buf && g_sps_pps_size > 0) {
    peer_connection_send_video(g_pc, g_sps_pps_buf, g_sps_pps_size, pts);
  }

  /* P-frame (or I-frame with failed SPS/PPS concat): send as-is */
  peer_connection_send_video(g_pc, data, size, pts);
  g_venc_in_pc = 0;
}

/* ── Network utilities ───────────────────────────────────────────── */

/**
 * Get the first non-loopback IPv4 address from common network interfaces.
 * Tries common RT-Smart interface names: u0, e0, eth0, en0, w0, wlan0.
 * Falls back to "0.0.0.0" if no IP is found.
 */
static void get_local_ip(char* buf, int buf_len) {
  static const char* ifnames[] = {"u0", "e0", "eth0", "en0", "w0", "wlan0", NULL};

  snprintf(buf, buf_len, "0.0.0.0");

  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    return;
  }

  struct ifreq ifr;
  for (int i = 0; ifnames[i] != NULL; i++) {
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ifnames[i], sizeof(ifr.ifr_name) - 1);

    if (ioctl(sock, SIOCGIFADDR, &ifr) == 0) {
      struct sockaddr_in* sin = (struct sockaddr_in*)&ifr.ifr_addr;
      if (sin->sin_addr.s_addr != htonl(INADDR_ANY)) {
        inet_ntop(AF_INET, &sin->sin_addr, buf, buf_len);
        break;
      }
    }
  }

  close(sock);
}

/* ── Worker threads ──────────────────────────────────────────────── */

/** Runs peer_connection_loop() at ~1kHz.
 *  This drives ICE candidate gathering, DTLS handshake,
 *  and SRTP packet processing. Must be called frequently. */
static void* peer_connection_task(void* data) {
  while (!g_interrupted) {
    g_peer_in_pc = 1;
    peer_connection_loop(g_pc);
    g_peer_in_pc = 0;
    usleep(1000);  /* ~1ms tick — balances responsiveness vs CPU usage */
  }
  pthread_exit(NULL);
  return NULL;
}

/** Polls VENC channel for encoded H.264 frames and sends them
 *  to the browser via WebRTC.
 *
 *  For each frame pack:
 *  1. mmap the physical address to get a CPU-accessible pointer
 *  2. If it's SPS/PPS (K_VENC_HEADER), cache it for later
 *  3. If it's an I-frame, prepend cached SPS/PPS
 *  4. If it's a P-frame, send as-is
 *  5. munmap and release the VENC stream buffer */
static void* venc_stream_task(void* data) {
  k_u32 venc_chn = mpp_pipeline_get_venc_chn();
  k_venc_stream output;
  k_venc_chn_status status;
  k_venc_pack static_packs[VENC_MAX_PACK_CNT];
  k_s32 ret;

  while (!g_interrupted) {
    memset(&output, 0, sizeof(output));

    /* Query how many packs are available in the current frame */
    kd_mpi_venc_query_status(venc_chn, &status);
    output.pack_cnt = status.cur_packs > 0 ? status.cur_packs : 1;
    if (output.pack_cnt > VENC_MAX_PACK_CNT) {
      output.pack_cnt = VENC_MAX_PACK_CNT;
    }
    output.pack = static_packs;

    /* Blocking get with 1s timeout — returns when a full frame is ready */
    ret = kd_mpi_venc_get_stream(venc_chn, &output, 1000);
    if (ret != K_SUCCESS) {
      continue;
    }

    /* Process each NAL unit pack in the frame */
    for (k_u32 i = 0; i < output.pack_cnt; i++) {
      /* mmap physical address to CPU-accessible virtual address */
      k_u8* pData = (k_u8*)kd_mpi_sys_mmap(output.pack[i].phys_addr, output.pack[i].len);
      if (pData) {
        send_venc_frame_to_webrtc(pData, output.pack[i].len,
                                  output.pack[i].type, output.pack[i].pts);
        kd_mpi_sys_munmap(pData, output.pack[i].len);
      }
    }

    /* Return the VB buffer to the pool so VENC can reuse it */
    kd_mpi_venc_release_stream(venc_chn, &output);
  }

  return NULL;
}

/* ── HTTP request handler (WebRTC signaling) ─────────────────────── */

/**
 * Handle HTTP requests for WebRTC signaling.
 *
 * Routes:
 *   GET  /           → Serve the embedded web page (web_page.h)
 *   GET  /index.html → Same as /
 *   GET  /offer      → Create a new SDP offer (closes any existing connection)
 *   POST /answer     → Set the remote SDP answer from the browser
 *   OPTIONS *        → CORS preflight
 *
 * Signaling flow:
 *   1. Browser calls GET /offer
 *   2. Server creates PeerConnection offer + gathers ICE candidates
 *   3. Server returns the full SDP offer (with candidates)
 *   4. Browser calls pc.setRemoteDescription(offer)
 *   5. Browser creates answer and POSTs it to /answer
 *   6. Server calls peer_connection_set_remote_description(answer)
 *   7. DTLS/SRTP handshake completes → video frames start flowing
 */
static void on_http_request(const char* method, const char* path,
                            const char* body, int body_len,
                            http_response_t* response) {

  /* ── Serve web page ── */
  if (strcmp(method, "GET") == 0 && (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0)) {
    response->status = 200;
    response->content_type = "text/html; charset=utf-8";
    response->body = WEB_PAGE_HTML;
    response->body_len = strlen(WEB_PAGE_HTML);

  /* ── Create SDP offer ── */
  } else if (strcmp(method, "GET") == 0 && strcmp(path, "/offer") == 0) {
    /* Close any existing peer connection before creating a new offer.
     * This allows the browser to reconnect without refreshing the page.
     *
     * After close(), drain: wait for both worker threads to exit g_pc
     * internals. close() sets pc->state=CLOSED so no new loop/send_video
     * calls will enter the dtls_srtp code path, but a call already past
     * the state check may still be using srtp_out. Wait until both
     * in-use flags are 0 — guaranteed within ~1ms. */
    if (g_state == PEER_CONNECTION_COMPLETED || g_state == PEER_CONNECTION_CONNECTED) {
      peer_connection_close(g_pc);
      g_state = PEER_CONNECTION_CLOSED;
      while (g_peer_in_pc || g_venc_in_pc) {
        usleep(1000);
      }
    }

    /* Reset offer state for new ICE gathering cycle */
    g_offer_ready = 0;
    if (g_offer_sdp) {
      free(g_offer_sdp);
      g_offer_sdp = NULL;
    }

    /* Start ICE gathering — onicecandidate() will be called when done */
    peer_connection_create_offer(g_pc);

    /* Wait for ICE gathering to complete (signaled by onicecandidate).
     * Also wakes on g_interrupted so we don't block forever during shutdown. */
    pthread_mutex_lock(&g_offer_mutex);
    while (!g_offer_ready && !g_interrupted) {
      pthread_cond_wait(&g_offer_cond, &g_offer_mutex);
    }

    if (g_offer_sdp) {
      response->status = 200;
      response->content_type = "application/sdp";
      response->body = g_offer_sdp;
      response->body_len = strlen(g_offer_sdp);
    } else {
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Failed to create offer";
      response->body_len = strlen(response->body);
    }
    pthread_mutex_unlock(&g_offer_mutex);

  /* ── Set remote SDP answer ── */
  } else if (strcmp(method, "POST") == 0 && strcmp(path, "/answer") == 0) {
    if (!body || body_len <= 0) {
      response->status = 400;
      response->content_type = "text/plain";
      response->body = "Missing SDP body";
      response->body_len = strlen(response->body);
      return;
    }

    /* Null-terminate the answer SDP for libpeer.
     * body is not guaranteed to be null-terminated (it points into
     * the receive buffer at the start of the body section). */
    char* answer_copy = (char*)calloc(1, body_len + 1);
    memcpy(answer_copy, body, body_len);
    answer_copy[body_len] = '\0';

    peer_connection_set_remote_description(g_pc, answer_copy, SDP_TYPE_ANSWER);

    response->status = 200;
    response->content_type = "text/plain";
    response->body = "OK";
    response->body_len = strlen(response->body);

    free(answer_copy);

  /* ── 404 for everything else ── */
  } else {
    response->status = 404;
    response->content_type = "text/plain";
    response->body = "Not Found";
    response->body_len = strlen(response->body);
  }
}

/* ── CLI usage ───────────────────────────────────────────────────── */

static void print_usage(const char* prog) {
  printf("Usage: %s [OPTIONS]\n", prog);
  printf("  -p port       HTTP server port (default: 8080)\n");
  printf("  -s csi_num    CSI device number 0-2 (default: 2)\n");
  printf("  -c connector  Connector type: 605274512=LCD, 757006876=HDMI (default: LCD)\n");
  printf("  -t type       Video encoder type: h264/h265 (default: h264)\n");
  printf("  -W width      VENC encode width (default: 1280)\n");
  printf("  -H height     VENC encode height (default: 720)\n");
  printf("  -b bitrate    VENC bitrate kbps (default: 2000)\n");
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
  /* Default configuration */
  int port = 8080;
  k_u32 csi_num = 2;
  k_connector_type connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
  k_u32 venc_width = 1280;
  k_u32 venc_height = 720;
  k_u32 venc_bitrate = 2000;
  VencType venc_type = VENC_TYPE_H264;

  /* Parse command-line arguments */
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-p") == 0 && (i + 1) < argc) {
      port = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-s") == 0 && (i + 1) < argc) {
      csi_num = (k_u32)atoi(argv[++i]);
    } else if (strcmp(argv[i], "-c") == 0 && (i + 1) < argc) {
      connector_type = (k_connector_type)atoi(argv[++i]);
    } else if (strcmp(argv[i], "-t") == 0 && (i + 1) < argc) {
      i++;
      if (strcmp(argv[i], "h265") == 0) {
        venc_type = VENC_TYPE_H265;
      } else if (strcmp(argv[i], "h264") == 0) {
        venc_type = VENC_TYPE_H264;
      } else {
        print_usage(argv[0]);
        return 1;
      }
    } else if (strcmp(argv[i], "-W") == 0 && (i + 1) < argc) {
      venc_width = (k_u32)atoi(argv[++i]);
    } else if (strcmp(argv[i], "-H") == 0 && (i + 1) < argc) {
      venc_height = (k_u32)atoi(argv[++i]);
    } else if (strcmp(argv[i], "-b") == 0 && (i + 1) < argc) {
      venc_bitrate = (k_u32)atoi(argv[++i]);
    } else {
      print_usage(argv[0]);
      return 1;
    }
  }

  /* Validate parameter ranges */
  if (venc_width < 64 || venc_width > 3840 || venc_height < 64 || venc_height > 2160) {
    printf("Error: resolution out of range (64-3840 x 64-2160), got %ux%u\n", venc_width, venc_height);
    return 1;
  }
  if (venc_bitrate < 100 || venc_bitrate > 20000) {
    printf("Error: bitrate out of range (100-20000 kbps), got %u\n", venc_bitrate);
    return 1;
  }

  /* Install SIGINT handler for clean shutdown (Ctrl+C) */
  signal(SIGINT, signal_handler);

  /* ── Initialize MPP pipeline ──
   * Sets up: VB pools → Display connector → VO layer →
 *          VICAP (CHN0→VO display, CHN1→VENC encode) → VENC H.264
 * Binds: VICAP-CHN0 → VO, VICAP-CHN1 → VENC */
  MppPipelineConfig pipeline_config = {
    .csi_num = csi_num,
    .connector_type = connector_type,
    .venc_width = venc_width,
    .venc_height = venc_height,
    .venc_bitrate_kbps = venc_bitrate,
    .venc_type = venc_type,
  };

  if (mpp_pipeline_init(&pipeline_config) != 0) {
    printf("Failed to initialize MPP pipeline\n");
    return 1;
  }

  if (mpp_pipeline_start() != 0) {
    printf("Failed to start MPP pipeline\n");
    mpp_pipeline_deinit();
    return 1;
  }

  /* ── Initialize WebRTC (libpeer) ──
   * Video-only configuration: H.264/H.265 codec, no audio, no DataChannel.
   * DataChannel was intentionally removed — this demo only streams video. */
  PeerConfiguration config = {
      .datachannel = DATA_CHANNEL_NONE,
      .video_codec = venc_type == VENC_TYPE_H265 ? CODEC_H265 : CODEC_H264,
      .audio_codec = CODEC_NONE};

  peer_init();
  g_pc = peer_connection_create(&config);
  peer_connection_oniceconnectionstatechange(g_pc, onconnectionstatechange);
  peer_connection_onicecandidate(g_pc, onicecandidate);

  /* ── Start worker threads ── */
  pthread_t peer_connection_thread;
  pthread_create(&peer_connection_thread, NULL, peer_connection_task, NULL);

  pthread_t venc_stream_thread;
  pthread_create(&venc_stream_thread, NULL, venc_stream_task, NULL);

  /* ── Start HTTP signaling server ── */
  http_server_start(port, on_http_request);

  /* Print access URL */
  char local_ip[64];
  get_local_ip(local_ip, sizeof(local_ip));

  printf("\n========================================\n");
  printf("  libpeer LAN Camera Demo\n");
  printf("========================================\n");
  printf("  CSI: %u, Connector: %d\n", csi_num, connector_type);
  printf("  Encode: %ux%u @ %ukbps %s\n", venc_width, venc_height, venc_bitrate,
         venc_type_name(venc_type));
  printf("  Open in browser:\n");
  printf("  http://%s:%d\n", local_ip, port);
  printf("========================================\n\n");

  /* ── Main loop: wait for SIGINT ──
   * This 100ms sleep loop is the signal-safe way to wait.
   * g_exit_requested is set by the signal handler. */
  while (!g_exit_requested) {
    usleep(100000);
  }

  /* ── Shutdown sequence ──
   * Order matters: we must stop producing data before stopping consumers.
   *
   * 1. Set g_interrupted to tell all worker threads to exit their loops
   * 2. Stop HTTP server (closes server socket, joins http thread)
   * 3. Join venc_stream thread (waits for it to finish current frame)
   * 4. Join peer_connection thread (waits for ICE loop to exit)
   * 5. Free cached SPS/PPS and offer SDP
   * 6. Destroy peer connection and deinit libpeer
   * 7. Deinit MPP pipeline (releases VB, VICAP, VENC, VO, connector) */
  g_interrupted = 1;

  http_server_stop();
  pthread_join(venc_stream_thread, NULL);
  pthread_join(peer_connection_thread, NULL);

  /* Free cached H.264 SPS/PPS */
  if (g_sps_pps_buf) {
    free(g_sps_pps_buf);
    g_sps_pps_buf = NULL;
  }

  /* Free any pending offer SDP */
  if (g_offer_sdp) {
    free(g_offer_sdp);
  }

  /* Destroy WebRTC peer connection and deinit libpeer */
  peer_connection_destroy(g_pc);
  peer_deinit();

  /* Deinit MPP pipeline (VB/VICAP/VENC/VO/connector) */
  mpp_pipeline_deinit();

  return 0;
}
