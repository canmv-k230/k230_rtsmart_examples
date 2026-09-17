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
 *   │  peer_connection_task ── ICE/DTLS ──> browsers             │
 *   └────────────────────────────────────────────────────────────┘
 *                                                       ▲
 *   ┌────────────────────────────────────────────────────────────┐
 *   │                    HTTP signaling                          │
 *   │  GET /offer  ──> create SDP offer ──> browser              │
 *   │  POST /answer?session=id <── remote SDP <── browser        │
 *   └────────────────────────────────────────────────────────────┘
 *
 * Thread model:
 *   - Main thread:      signal wait loop, then orchestrates shutdown
 *   - http_server:      accepts connections, parses HTTP, calls on_http_request
 *   - peer_connection:  runs all peer_connection_loop() calls at 1ms interval
 *   - venc_stream:      polls VENC and fans encoded frames out to all peers
 *
 * Shutdown sequence (triggered by SIGINT):
 *   1. g_exit_requested = 1  (signal handler, safe for async-signal-safe)
 *   2. Main loop exits, sets g_interrupted = 1  (tells threads to stop)
 *   3. http_server_stop()    (closes server fd, joins http thread)
 *   4. pthread_join(venc)    (waits for encode thread to finish)
 *   5. pthread_join(peer)    (waits for peer connection thread to finish)
 *   6. Free resources, destroy peer connections, deinit MPP pipeline
 */

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "peer.h"
#include "hal_netmgmt.h"
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

#define MAX_WEBRTC_CLIENTS 4
#define NEGOTIATION_TIMEOUT_MS 30000
#define SESSION_ID_HEX_LEN 32
#define PEER_NEGOTIATION_POLL_US 1000
#define PEER_CONNECTED_POLL_US 10000
#define PEER_IDLE_POLL_US 50000

typedef struct {
  PeerConnection* pc;
  PeerConnectionState state;
  char id[SESSION_ID_HEX_LEN + 1];
  uint64_t last_activity_ms;
  int reserved;
  char client_ip[INET_ADDRSTRLEN];
  char local_ip[INET_ADDRSTRLEN];
} WebRtcSession;

static WebRtcSession g_sessions[MAX_WEBRTC_CLIENTS];
static pthread_mutex_t g_sessions_mutex = PTHREAD_MUTEX_INITIALIZER;
static MediaCodec g_video_codec = CODEC_H265;

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

/** Called while g_sessions_mutex is held. */
static void onconnectionstatechange(PeerConnectionState state, void* data) {
  WebRtcSession* session = (WebRtcSession*)data;
  printf("Session %s state: %s\n", session->id,
         peer_connection_state_to_string(state));
  session->state = state;
}

/** SIGINT handler: sets flag for main loop to exit.
 *  Only sets g_exit_requested; does NOT set g_interrupted here
 *  because worker threads should continue running until main()
 *  orchestrates an orderly shutdown. */
static void signal_handler(int sig) {
  g_exit_requested = 1;
}

static uint64_t get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

static int session_is_terminal(PeerConnectionState state) {
  return state == PEER_CONNECTION_CLOSED || state == PEER_CONNECTION_FAILED ||
         state == PEER_CONNECTION_DISCONNECTED;
}

/* Read time only while holding g_sessions_mutex, after any activity update. */
static int session_negotiation_expired(const WebRtcSession* session, uint64_t now) {
  return (session->state == PEER_CONNECTION_NEW ||
          session->state == PEER_CONNECTION_CHECKING ||
          session->state == PEER_CONNECTION_CONNECTED) &&
         now >= session->last_activity_ms &&
         now - session->last_activity_ms > NEGOTIATION_TIMEOUT_MS;
}

/** Destroy a session. The caller must hold g_sessions_mutex. */
static void session_destroy_locked(WebRtcSession* session) {
  if (session->pc) {
    peer_connection_close(session->pc);
    peer_connection_destroy(session->pc);
    session->pc = NULL;
  }
  session->state = PEER_CONNECTION_CLOSED;
  session->id[0] = '\0';
  session->last_activity_ms = 0;
  session->reserved = 0;
  session->client_ip[0] = '\0';
  session->local_ip[0] = '\0';
}

/** Find a session by ID. The caller must hold g_sessions_mutex. */
static WebRtcSession* session_find(const char* id) {
  for (int i = 0; i < MAX_WEBRTC_CLIENTS; i++) {
    WebRtcSession* session = &g_sessions[i];
    if (session->pc && strcmp(session->id, id) == 0) {
      return session;
    }
  }
  return NULL;
}

/** Reserve a free or expired slot. The caller must hold g_sessions_mutex. */
static WebRtcSession* session_available(void) {
  uint64_t now = get_time_ms();

  for (int i = 0; i < MAX_WEBRTC_CLIENTS; i++) {
    WebRtcSession* session = &g_sessions[i];
    if ((!session->pc && !session->reserved) ||
        session_is_terminal(session->state) ||
        session_negotiation_expired(session, now)) {
      if (session->pc) {
        session_destroy_locked(session);
      }
      return session;
    }
  }
  return NULL;
}

static int session_matches_client(const WebRtcSession* session,
                                  const char* client_ip,
                                  const char* local_ip) {
  if (!client_ip || strcmp(session->client_ip, client_ip) != 0) {
    return 0;
  }
  if (local_ip && local_ip[0] != '\0' && strcmp(local_ip, "0.0.0.0") != 0 &&
      strcmp(session->local_ip, local_ip) != 0) {
    return 0;
  }
  return 1;
}

static int secure_random_hex(char* output, size_t hex_len) {
  static const char hex[] = "0123456789abcdef";
  uint8_t random[32];
  size_t byte_len = hex_len / 2;

  if (!output || hex_len == 0 || (hex_len & 1) != 0 || byte_len > sizeof(random)) {
    return -1;
  }
  if (peer_random_bytes(random, byte_len) != 0) {
    return -1;
  }
  for (size_t i = 0; i < byte_len; i++) {
    output[i * 2] = hex[random[i] >> 4];
    output[i * 2 + 1] = hex[random[i] & 0x0f];
  }
  output[hex_len] = '\0';
  return 0;
}

static int path_matches_route(const char* path, const char* route) {
  size_t route_len = strlen(route);
  return strncmp(path, route, route_len) == 0 &&
         (path[route_len] == '\0' || path[route_len] == '?');
}

static int query_parameter(const char* path, const char* key,
                           char* output, size_t output_size) {
  const char* item = strchr(path, '?');
  size_t key_len = strlen(key);

  if (!item || output_size == 0) return 0;
  item++;
  while (*item) {
    const char* item_end = strchr(item, '&');
    const char* equals;
    size_t value_len;
    if (!item_end) item_end = item + strlen(item);
    equals = memchr(item, '=', (size_t)(item_end - item));
    if (equals && (size_t)(equals - item) == key_len &&
        strncmp(item, key, key_len) == 0) {
      value_len = (size_t)(item_end - equals - 1);
      if (value_len == 0 || value_len >= output_size) return 0;
      memcpy(output, equals + 1, value_len);
      output[value_len] = '\0';
      return 1;
    }
    if (*item_end == '\0') break;
    item = item_end + 1;
  }
  return 0;
}

static int parse_session_id(const char* path, char* session_id) {
  if (!query_parameter(path, "session", session_id, SESSION_ID_HEX_LEN + 1) ||
      strlen(session_id) != SESSION_ID_HEX_LEN) {
    return 0;
  }
  for (size_t i = 0; i < SESSION_ID_HEX_LEN; i++) {
    if (!isxdigit((unsigned char)session_id[i])) return 0;
  }
  return 1;
}

/** Caller must hold g_sessions_mutex. */
static int session_generate_id(char* session_id) {
  for (int attempt = 0; attempt < 8; attempt++) {
    if (secure_random_hex(session_id, SESSION_ID_HEX_LEN) != 0) return -1;
    if (!session_find(session_id)) return 0;
  }
  return -1;
}

/** Replace mDNS host candidates with the HTTP client's IPv4 address.
 *
 * Browsers may hide their LAN address behind a .local mDNS name. libpeer
 * cannot resolve that name, but the signaling server already knows the
 * browser's address from the accepted TCP connection. Only the candidate
 * address field is changed; ICE credentials and candidate ports are kept.
 */
static int replace_mdns_candidates(char* sdp, size_t capacity, const char* client_ip) {
  static const char candidate_prefix[] = "a=candidate:";
  const size_t client_ip_len = client_ip ? strlen(client_ip) : 0;
  char* line = sdp;
  int replaced = 0;

  if (!sdp || !client_ip_len) return 0;

  while (*line) {
    char* line_end = strstr(line, "\r\n");
    if (!line_end) line_end = line + strlen(line);

    if ((size_t)(line_end - line) > sizeof(candidate_prefix) - 1 &&
        strncmp(line, candidate_prefix, sizeof(candidate_prefix) - 1) == 0) {
      char* address_start = line;

      /* Skip foundation, component, transport, and priority fields. */
      for (int field = 0; field < 4 && address_start < line_end; field++) {
        address_start = memchr(address_start, ' ', line_end - address_start);
        if (!address_start) break;
        while (address_start < line_end && *address_start == ' ') address_start++;
      }

      if (address_start && address_start < line_end) {
        char* address_end = memchr(address_start, ' ', line_end - address_start);
        if (!address_end) address_end = line_end;

        size_t address_len = address_end - address_start;
        if (address_len > 6 && strncasecmp(address_end - 6, ".local", 6) == 0) {
          size_t sdp_len = strlen(sdp);
          size_t tail_len = sdp_len - (address_end - sdp) + 1;
          size_t new_sdp_len = sdp_len - address_len + client_ip_len;

          if (new_sdp_len + 1 > capacity) return -1;

          memmove(address_start + client_ip_len, address_end, tail_len);
          memcpy(address_start, client_ip, client_ip_len);
          line_end += (ptrdiff_t)client_ip_len - (ptrdiff_t)address_len;
          replaced++;
        }
      }
    }

    if (!*line_end) break;
    line = line_end + 2;
  }

  return replaced;
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

  pthread_mutex_lock(&g_sessions_mutex);
  for (int i = 0; i < MAX_WEBRTC_CLIENTS; i++) {
    WebRtcSession* session = &g_sessions[i];
    if (session->pc && session->state == PEER_CONNECTION_COMPLETED) {
      /* Parameter sets and the frame are sent separately. Each peer owns an
       * independent RTP sequence/timestamp state and SRTP context. */
      if (type == K_VENC_I_FRAME && g_sps_pps_buf && g_sps_pps_size > 0) {
        peer_connection_send_video(session->pc, g_sps_pps_buf,
                                   g_sps_pps_size, pts);
      }
      peer_connection_send_video(session->pc, data, size, pts);
    }
  }
  pthread_mutex_unlock(&g_sessions_mutex);
}

/* ── Network utilities ───────────────────────────────────────────── */

/**
 * Format an RT-Smart network-management interface's IPv4 address.
 */
static int get_netif_ip(enum rt_netif_t netif, char* buf, int buf_len) {
  struct ifconfig_t config;
  struct in_addr address;
  char netdev_name[32];

  if (netmgmt_utils_get_netdev_name(netif, netdev_name) != 0 || netdev_name[0] == '\0') {
    return 0;
  }
  memset(&config, 0, sizeof(config));
  if (netmgmt_utils_get_ifconfig(netif, &config) != 0 || config.ip.addr == 0) {
    return 0;
  }

  address.s_addr = config.ip.addr;
  return inet_ntop(AF_INET, &address, buf, buf_len) != NULL;
}

static void print_network_devices(void) {
  int dev_num = 0;
  char names[NET_DEV_MAX_CNT][32];
  if (netmgmt_utils_get_dev_list(&dev_num, names) != 0) {
    printf("Network device list unavailable\n");
    return;
  }
  printf("Active network devices:");
  for (int i = 0; i < dev_num && i < NET_DEV_MAX_CNT; i++) {
    printf(" %s", names[i]);
  }
  printf("\n");
}

static int collect_http_local_ips(char local_ips[][INET_ADDRSTRLEN],
                                  const char* interface_names[], int max_ips) {
  static const struct {
    enum rt_netif_t netif;
    const char* name;
  } interfaces[] = {
    {RT_NET_DEV_WLAN_AP, "SoftAP"},
    {RT_NET_DEV_WLAN_STA, "Wi-Fi STA"},
    {RT_NET_DEV_LAN, "LAN"},
  };
  int count = 0;
  int ap_active = 0;
  for (size_t i = 0; i < sizeof(interfaces) / sizeof(interfaces[0]) && count < max_ips; i++) {
    if (interfaces[i].netif == RT_NET_DEV_WLAN_AP &&
        (netmgmt_wlan_ap_isactived(&ap_active) != 0 || !ap_active)) continue;
    char ip[INET_ADDRSTRLEN] = {0};
    if (!get_netif_ip(interfaces[i].netif, ip, sizeof(ip))) continue;
    int duplicate = 0;
    for (int j = 0; j < count; j++) if (strcmp(local_ips[j], ip) == 0) duplicate = 1;
    if (!duplicate) {
      snprintf(local_ips[count], INET_ADDRSTRLEN, "%s", ip);
      if (interface_names) interface_names[count] = interfaces[i].name;
      count++;
    }
  }
  return count;
}

static int get_http_local_ips(char local_ips[][INET_ADDRSTRLEN], int max_ips,
                              void* user_data) {
  (void)user_data;
  return collect_http_local_ips(local_ips, NULL, max_ips);
}

static const char* interface_name_for_ip(const char* ip) {
  char local_ips[HTTP_MAX_LISTENERS][INET_ADDRSTRLEN] = {{0}};
  const char* interface_names[HTTP_MAX_LISTENERS] = {0};
  int count = collect_http_local_ips(local_ips, interface_names,
                                     HTTP_MAX_LISTENERS);
  for (int i = 0; i < count; i++) {
    if (strcmp(local_ips[i], ip) == 0) {
      return interface_names[i];
    }
  }
  return "interface";
}

/* ── Worker threads ──────────────────────────────────────────────── */

/** Runs peer_connection_loop() frequently during negotiation, then reduces
 *  the polling rate once sessions are established or idle. */
static void* peer_connection_task(void* data) {
  (void)data;

  while (!g_interrupted) {
    unsigned int poll_delay_us = PEER_IDLE_POLL_US;

    pthread_mutex_lock(&g_sessions_mutex);
    for (int i = 0; i < MAX_WEBRTC_CLIENTS; i++) {
      WebRtcSession* session = &g_sessions[i];
      if (session->pc) {
        peer_connection_loop(session->pc);
        uint64_t now = get_time_ms();
        if (session_is_terminal(session->state) ||
            session_negotiation_expired(session, now)) {
          if (!session_is_terminal(session->state)) {
            printf("Session %s signaling timeout after %" PRIu64 " ms\n",
                   session->id, now - session->last_activity_ms);
          }
          session_destroy_locked(session);
          continue;
        }

        if (session->state == PEER_CONNECTION_CHECKING ||
            session->state == PEER_CONNECTION_CONNECTED) {
          poll_delay_us = PEER_NEGOTIATION_POLL_US;
        } else if (session->state == PEER_CONNECTION_COMPLETED) {
          if (poll_delay_us > PEER_CONNECTED_POLL_US) {
            poll_delay_us = PEER_CONNECTED_POLL_US;
          }
        }
      }
    }
    pthread_mutex_unlock(&g_sessions_mutex);

    usleep(poll_delay_us);
  }
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
  (void)data;

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
 *   GET  /offer      → Allocate a session and create its SDP offer
 *   POST /answer?session=<id> → Set that session's remote SDP answer
 *   POST /disconnect?session=<id> → Release a session immediately
 *
 * Signaling flow:
 *   1. Browser calls GET /offer
 *   2. Server creates PeerConnection offer + gathers ICE candidates
 *   3. Server returns the SDP and an X-WebRTC-Session response header
 *   4. Browser calls pc.setRemoteDescription(offer)
 *   5. Browser creates answer and POSTs it with the session ID
 *   6. Server calls peer_connection_set_remote_description(answer)
 *   7. DTLS/SRTP handshake completes → video frames start flowing
 */
static void on_http_request(const char* method, const char* path,
                            const char* body, int body_len,
                            const char* client_ip,
                            const char* local_ip,
                            http_response_t* response) {
  const char* request_local_ip =
      local_ip && local_ip[0] != '\0' && strcmp(local_ip, "0.0.0.0") != 0
          ? local_ip
          : NULL;
  if (strcmp(method, "OPTIONS") == 0) {
    response->status = 204;
    response->content_type = "text/plain";
    response->body = NULL;
    response->body_len = 0;
    return;
  }

  /* ── Serve web page ── */
  if (strcmp(method, "GET") == 0 &&
      (path_matches_route(path, "/") || path_matches_route(path, "/index.html"))) {
    response->status = 200;
    response->content_type = "text/html; charset=utf-8";
    response->body = WEB_PAGE_HTML;
    response->body_len = strlen(WEB_PAGE_HTML);

  /* ── Create SDP offer ── */
  } else if (strcmp(method, "GET") == 0 && path_matches_route(path, "/offer")) {
    WebRtcSession* session;
    PeerConnection* new_pc;
    PeerConfiguration config = {0};
    const char* offer;
    /* HTTP requests are serialized; keep the response independent of peer
     * lifetime once the session lock is released. */
    static char offer_response[8192];

    if (!client_ip || client_ip[0] == '\0' || !request_local_ip ||
        request_local_ip[0] == '\0' || strcmp(request_local_ip, "0.0.0.0") == 0) {
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Failed to determine signaling interface";
      response->body_len = strlen(response->body);
      return;
    }

    pthread_mutex_lock(&g_sessions_mutex);
    session = session_available();
    if (!session) {
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 503;
      response->content_type = "text/plain";
      response->body = "Maximum WebRTC clients reached";
      response->body_len = strlen(response->body);
      return;
    }

    if (session_generate_id(session->id) != 0) {
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Failed to create session ID";
      response->body_len = strlen(response->body);
      return;
    }
    session->last_activity_ms = get_time_ms();
    session->reserved = 1;
    session->state = PEER_CONNECTION_NEW;
    snprintf(session->client_ip, sizeof(session->client_ip), "%s", client_ip);
    snprintf(session->local_ip, sizeof(session->local_ip), "%s", request_local_ip);
    pthread_mutex_unlock(&g_sessions_mutex);

    config.datachannel = DATA_CHANNEL_NONE;
    config.video_codec = g_video_codec;
    config.audio_codec = CODEC_NONE;
    config.user_data = session;
    new_pc = peer_connection_create(&config);
    if (!new_pc) {
      pthread_mutex_lock(&g_sessions_mutex);
      session_destroy_locked(session);
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Failed to create WebRTC peer";
      response->body_len = strlen(response->body);
      return;
    }
    if (peer_connection_set_local_ip(new_pc, session->local_ip) != 0) {
      peer_connection_destroy(new_pc);
      pthread_mutex_lock(&g_sessions_mutex);
      session_destroy_locked(session);
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Failed to bind local ICE address";
      response->body_len = strlen(response->body);
      return;
    }

    /* Build the new peer before publishing it to the worker and encoder
     * threads, so existing sessions continue during certificate generation. */
    offer = peer_connection_create_offer(new_pc);
    if (!offer || strlen(offer) >= sizeof(offer_response)) {
      peer_connection_destroy(new_pc);
      pthread_mutex_lock(&g_sessions_mutex);
      session_destroy_locked(session);
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Failed to create offer";
      response->body_len = strlen(response->body);
      return;
    }

    memcpy(offer_response, offer, strlen(offer) + 1);

    pthread_mutex_lock(&g_sessions_mutex);
    peer_connection_oniceconnectionstatechange(new_pc,
                                               onconnectionstatechange);
    session->pc = new_pc;
    session->reserved = 0;
    session->last_activity_ms = get_time_ms();

    response->status = 200;
    response->content_type = "application/sdp";
    response->body = offer_response;
    response->body_len = strlen(offer_response);
    snprintf(response->extra_headers, sizeof(response->extra_headers),
             "X-WebRTC-Session: %s\r\n", session->id);
    printf("Session %s using local ICE address %s for HTTP client %s\n",
           session->id, session->local_ip, session->client_ip);
    pthread_mutex_unlock(&g_sessions_mutex);

  /* ── Set remote SDP answer ── */
  } else if (strcmp(method, "POST") == 0 &&
             strncmp(path, "/answer?", strlen("/answer?")) == 0) {
    char session_id[SESSION_ID_HEX_LEN + 1];
    WebRtcSession* session;

    if (!parse_session_id(path, session_id)) {
      response->status = 400;
      response->content_type = "text/plain";
      response->body = "Missing or invalid session ID";
      response->body_len = strlen(response->body);
      return;
    }
    if (!body || body_len <= 0) {
      response->status = 400;
      response->content_type = "text/plain";
      response->body = "Missing SDP body";
      response->body_len = strlen(response->body);
      return;
    }

    pthread_mutex_lock(&g_sessions_mutex);
    session = session_find(session_id);
    if (!session) {
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 404;
      response->content_type = "text/plain";
      response->body = "WebRTC session not found";
      response->body_len = strlen(response->body);
      return;
    }
    if (!session_matches_client(session, client_ip, local_ip)) {
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 403;
      response->content_type = "text/plain";
      response->body = "WebRTC session client mismatch";
      response->body_len = strlen(response->body);
      return;
    }

    /* Null-terminate the answer SDP for libpeer and leave enough room to
     * replace short mDNS hostnames with an IPv4 address if necessary.
     * body is not guaranteed to be null-terminated (it points into
     * the receive buffer at the start of the body section). */
    size_t candidate_count = 0;
    const char* candidate = body;
    const char* body_end = body + body_len;
    while (candidate < body_end &&
           (candidate = strstr(candidate, "a=candidate:")) != NULL &&
           candidate < body_end) {
      candidate_count++;
      candidate += strlen("a=candidate:");
    }

    size_t answer_capacity = body_len + candidate_count * INET_ADDRSTRLEN + 1;
    char* answer_copy = (char*)calloc(1, answer_capacity);
    if (!answer_copy) {
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Out of memory";
      response->body_len = strlen(response->body);
      return;
    }
    memcpy(answer_copy, body, body_len);
    answer_copy[body_len] = '\0';

    int mdns_replaced = replace_mdns_candidates(answer_copy, answer_capacity, client_ip);
    if (mdns_replaced < 0) {
      free(answer_copy);
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 500;
      response->content_type = "text/plain";
      response->body = "Failed to process SDP";
      response->body_len = strlen(response->body);
      return;
    }
    if (mdns_replaced > 0) {
      printf("Replaced %d mDNS ICE candidate(s) with HTTP client IP %s\n",
             mdns_replaced, client_ip);
    }

    peer_connection_set_remote_description(session->pc, answer_copy,
                                           SDP_TYPE_ANSWER);
    if (peer_connection_get_state(session->pc) == PEER_CONNECTION_FAILED) {
      free(answer_copy);
      session_destroy_locked(session);
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 400;
      response->content_type = "text/plain";
      response->body = "Invalid SDP answer";
      response->body_len = strlen(response->body);
      return;
    }
    session->last_activity_ms = get_time_ms();
    pthread_mutex_unlock(&g_sessions_mutex);

    response->status = 200;
    response->content_type = "text/plain";
    response->body = "OK";
    response->body_len = strlen(response->body);

    free(answer_copy);

  /* ── Release a WebRTC session ── */
  } else if (strcmp(method, "POST") == 0 &&
             strncmp(path, "/disconnect?", strlen("/disconnect?")) == 0) {
    char session_id[SESSION_ID_HEX_LEN + 1];
    WebRtcSession* session;

    if (!parse_session_id(path, session_id)) {
      response->status = 400;
      response->content_type = "text/plain";
      response->body = "Missing or invalid session ID";
      response->body_len = strlen(response->body);
      return;
    }
    pthread_mutex_lock(&g_sessions_mutex);
    session = session_find(session_id);
    if (!session) {
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 200;
      response->content_type = "text/plain";
      response->body = "OK";
      response->body_len = 2;
      return;
    }
    if (!session_matches_client(session, client_ip, local_ip)) {
      pthread_mutex_unlock(&g_sessions_mutex);
      response->status = 403;
      response->content_type = "text/plain";
      response->body = "WebRTC session client mismatch";
      response->body_len = strlen(response->body);
      return;
    }
    printf("Session %s disconnected by HTTP client %s\n",
           session->id, session->client_ip);
    session_destroy_locked(session);
    pthread_mutex_unlock(&g_sessions_mutex);
    response->status = 200;
    response->content_type = "text/plain";
    response->body = "OK";
    response->body_len = 2;

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
  printf("  -t type       Video encoder type: h264/h265 (default: h265)\n");
  printf("  -W width      VENC encode width (default: 1280)\n");
  printf("  -H height     VENC encode height (default: 720)\n");
  printf("  -b bitrate    VENC bitrate kbps (default: 512)\n");
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
  /* Default configuration */
  int port = 8080;
  k_u32 csi_num = 2;
  k_connector_type connector_type = ST7701_V1_MIPI_2LAN_480X800_30FPS;
  k_u32 venc_width = 1280;
  k_u32 venc_height = 720;
  k_u32 venc_bitrate = 512;
  VencType venc_type = VENC_TYPE_H265;

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
 *          VICAP (CHN0→VO display, CHN1→VENC encode) → VENC H.264/H.265
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

  /* Create each PeerConnection on demand so it can bind to the interface
   * used by that browser's HTTP connection. */
  print_network_devices();
  g_video_codec = venc_type == VENC_TYPE_H265 ? CODEC_H265 : CODEC_H264;
  for (int i = 0; i < MAX_WEBRTC_CLIENTS; i++) {
    g_sessions[i].state = PEER_CONNECTION_CLOSED;
  }
  if (peer_init() != 0) {
    printf("Failed to initialize libpeer\n");
    mpp_pipeline_deinit();
    return 1;
  }
  /* ── Start worker threads ── */
  pthread_t peer_connection_thread;
  pthread_create(&peer_connection_thread, NULL, peer_connection_task, NULL);

  pthread_t venc_stream_thread;
  pthread_create(&venc_stream_thread, NULL, venc_stream_task, NULL);

  /* ── Start HTTP signaling server ── */
  if (http_server_start(port, on_http_request, get_http_local_ips, NULL) != 0) {
    printf("Failed to start HTTP server\n");
    g_interrupted = 1;
    pthread_join(venc_stream_thread, NULL);
    pthread_join(peer_connection_thread, NULL);
    pthread_mutex_lock(&g_sessions_mutex);
    for (int i = 0; i < MAX_WEBRTC_CLIENTS; i++) {
      session_destroy_locked(&g_sessions[i]);
    }
    pthread_mutex_unlock(&g_sessions_mutex);
    peer_deinit();
    mpp_pipeline_deinit();
    return 1;
  }

  /* Print access URL */
  printf("\n========================================\n");
  printf("  libpeer LAN Camera Demo\n");
  printf("========================================\n");
  printf("  CSI: %u, Connector: %d\n", csi_num, connector_type);
  printf("  Encode: %ux%u @ %ukbps %s\n", venc_width, venc_height, venc_bitrate,
         venc_type_name(venc_type));
  printf("  Clients: up to %d concurrent sessions\n", MAX_WEBRTC_CLIENTS);
  printf("  Open in browser:\n");
  char access_ips[HTTP_MAX_LISTENERS][INET_ADDRSTRLEN] = {{0}};
  int access_ip_count = http_server_get_local_ips(access_ips,
                                                  HTTP_MAX_LISTENERS);
  if (access_ip_count == 1 && strcmp(access_ips[0], "0.0.0.0") == 0) {
    access_ip_count = collect_http_local_ips(access_ips, NULL,
                                             HTTP_MAX_LISTENERS);
  }
  if (access_ip_count == 0) {
    printf("  No configured IPv4 interface address is available yet\n");
  } else {
    for (int i = 0; i < access_ip_count; i++) {
      printf("  http://%s:%d/ (%s)\n", access_ips[i], port,
             interface_name_for_ip(access_ips[i]));
    }
  }
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
   * 5. Free cached SPS/PPS and all peer sessions
   * 6. Deinitialize libpeer
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

  pthread_mutex_lock(&g_sessions_mutex);
  for (int i = 0; i < MAX_WEBRTC_CLIENTS; i++) {
    session_destroy_locked(&g_sessions[i]);
  }
  pthread_mutex_unlock(&g_sessions_mutex);

  peer_deinit();

  /* Deinit MPP pipeline (VB/VICAP/VENC/VO/connector) */
  mpp_pipeline_deinit();

  return 0;
}
