/**
 * @file mpp_pipeline.c
 * @brief K230 MPP (Media Process Platform) video pipeline for WebRTC demo
 *
 * Pipeline topology:
 *
 *   ┌──────────────┐                 ┌──────────────┐
 *   │    VICAP     │── CHN0 (bind) ──>│    VO        │
 *   │  (Camera     │                 │ (LCD/HDMI    │
 *   │   Capture)   │── CHN1 (bind) ──>│  display)    │
 *   │              │                 └──────────────┘
 *   └──────────────┘
 *         │
 *         │ CHN1 (bind)
 *         ▼
 *   ┌──────────────┐
 *   │    VENC      │── H.264 bitstream ──> WebRTC (via main.c)
 *   │  (H.264      │
 *   │   encoder)   │
 *   └──────────────┘
 *
 * Init order (must be exactly this sequence):
 *   1. VB (Video Buffer) system init
 *   2. Connector (display panel: LCD or HDMI)
 *   3. VO Layer (video output overlay)
 *   4. VICAP (camera capture, 2 channels)
 *   5. VENC (H.264 encoder with dedicated VB pool)
 *   6. Bind VICAP-CHN0 → VO (display)
 *   7. Bind VICAP-CHN1 → VENC (encode)
 *
 * Deinit order (reverse of init, with VB release considerations):
 *   1. Stop VENC + VICAP streams
 *   2. Drain remaining VENC frames (release VB blocks)
 *   3. Unbind VICAP-VO and VICAP-VENC
 *   4. Disable VO layer (releases VB blocks held by display)
 *   5. Deinit VICAP
 *   6. Deinit VENC + destroy VB pool
 *   7. Deinit connector + VB system
 *
 * ⚠ VB (Video Buffer) release order is CRITICAL on K230:
 *   - VO layer holds references to VB blocks for display
 *   - If VICAP/VENC are deinitialized before VO layer is disabled,
 *     VB blocks remain locked and the system reports VB leaks
 *   - Therefore: layer_exit() MUST come before vicap_exit()/venc_exit()
 */

#include "mpp_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mpi_sensor_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_vo_api.h"
#include "mpi_connector_api.h"
#include "mpi_venc_api.h"
#include "kd_display.h"

/* ── Constants ───────────────────────────────────────────────────── */

#define DEFAULT_ISP_WIDTH   (1920)   /* Sensor input width (before scaling) */
#define DEFAULT_ISP_HEIGHT  (1080)   /* Sensor input height (before scaling) */
#define DEFAULT_FPS         (30)     /* Sensor frame rate */
#define VB_BUF_COUNT        (6)      /* VB buffers per VICAP channel */

/* ── Channel assignments ─────────────────────────────────────────── */

static k_vicap_dev g_vicap_dev = VICAP_DEV_ID_0;       /* VICAP device 0 */
static k_vicap_chn g_vicap_chn_vo = VICAP_CHN_ID_0;    /* CHN0 → VO display */
static k_vicap_chn g_vicap_chn_venc = VICAP_CHN_ID_1;  /* CHN1 → VENC encode */

static k_vo_layer_id g_vo_layer = K_VO_LAYER_VIDEO1;   /* VO overlay layer */

static k_u32 g_venc_chn = 0;                            /* Allocated VENC channel ID */
static k_u32 g_venc_attach_pool_id = VB_INVALID_POOLID; /* VENC's dedicated VB pool */

static k_u32 g_display_width = 0;   /* Actual display panel width (after rotation) */
static k_u32 g_display_height = 0;  /* Actual display panel height (after rotation) */

static int g_pipeline_initialized = 0;  /* 1 after mpp_pipeline_init succeeds */
static int g_pipeline_started = 0;      /* 1 after mpp_pipeline_start succeeds */

/* ── VB (Video Buffer) ───────────────────────────────────────────── */

/**
 * Initialize the VB (Video Buffer) subsystem.
 * VB provides shared memory pools that VICAP, VO, and VENC use to
 * exchange video frames without copying.
 */
static int vb_init(void) {
  k_vb_config config;
  k_vb_supplement_config supplement_config;
  k_s32 ret;

  memset(&config, 0, sizeof(config));
  config.max_pool_cnt = 64;  /* Max number of VB pools (global limit) */

  ret = kd_mpi_vb_set_config(&config);
  if (ret) {
    printf("[MPP] VB set_config failed, ret=%d\n", ret);
    return ret;
  }

  /* Enable JPEG supplement data (required by some VICAP operations) */
  memset(&supplement_config, 0, sizeof(supplement_config));
  supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;

  ret = kd_mpi_vb_set_supplement_config(&supplement_config);
  if (ret) {
    printf("[MPP] VB set_supplement_config failed, ret=%d\n", ret);
    return ret;
  }

  ret = kd_mpi_vb_init();
  if (ret) {
    printf("[MPP] VB init failed, ret=%d\n", ret);
    return ret;
  }

  return 0;
}

static void vb_exit(void) {
  kd_mpi_vb_exit();
}

/* ── Connector (Display Panel) ───────────────────────────────────── */

/**
 * Initialize the display connector (LCD panel or HDMI).
 *
 * For the ST7701 LCD panel (480x800), the physical orientation is
 * portrait (480 wide × 800 tall), but we use it in landscape mode
 * (800 wide × 480 tall), so we swap width/height.
 *
 * @param panel_id    Connector type (LCD or HDMI)
 * @param out_width   Output: actual display width in pixels
 * @param out_height  Output: actual display height in pixels
 * @return 0 on success, negative on failure
 */
static int connector_init(k_connector_type panel_id, k_u32* out_width, k_u32* out_height) {
  k_s32 ret;

  ret = kd_display_init(panel_id, 0, 0, GDMA_ROTATE_DEGREE_0);
  if (ret) {
    printf("[MPP] Display init failed, ret=%d\n", ret);
    return ret;
  }

  ret = kd_display_get_resolution(out_width, out_height);
  if (ret) {
    printf("[MPP] Get display resolution failed, ret=%d\n", ret);
    kd_display_deinit();
    return ret;
  }

  /* ST7701 LCD is portrait (480x800), swap to landscape (800x480) */
  if (panel_id == ST7701_V1_MIPI_2LAN_480X800_30FPS) {
    k_u32 temp = *out_width;
    *out_width = *out_height;
    *out_height = temp;
  }

  printf("[MPP] Display: %ux%u\n", *out_width, *out_height);
  return 0;
}

static void connector_exit(void) {
  kd_display_deinit();
}

/* ── VO Layer (Video Output) ─────────────────────────────────────── */

/**
 * Initialize a VO (Video Output) layer for displaying camera preview.
 *
 * The layer is configured with NV12 (YUV420 semi-planar) pixel format,
 * which matches the VICAP output format. A global alpha of 0xFF means
 * fully opaque.
 *
 * We first try to disable the layer (in case it was left enabled from
 * a previous run/crash), then configure and enable it.
 */
static int layer_init(k_vo_layer_id layer_id, k_u32 width, k_u32 height) {
  k_vo_layer_attr layer_attr;
  k_s32 ret;

  /* Disable first in case of leftover state from previous run */
  ret = kd_mpi_vo_disable_layer(layer_id);
  if (ret) {
    printf("[MPP] WARNING: VO disable layer failed, ret=%d\n", ret);
  }

  memset(&layer_attr, 0, sizeof(layer_attr));
  layer_attr.layer_id = layer_id;
  layer_attr.position.x = 0;
  layer_attr.position.y = 0;
  layer_attr.img_size.width = width;
  layer_attr.img_size.height = height;
  layer_attr.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;  /* NV12 */
  layer_attr.global_alpha = 0xFF;  /* Fully opaque */
  layer_attr.func = GDMA_ROTATE_DEGREE_0;
  layer_attr.rot_buf_nr = 0;
  layer_attr.rot_buf_bpp = 0;

  ret = kd_mpi_vo_set_layer_attr(layer_id, &layer_attr);
  if (ret != K_SUCCESS) {
    printf("[MPP] VO set_layer_attr failed, ret=%d\n", ret);
    return ret;
  }

  ret = kd_mpi_vo_enable_layer(layer_id);
  if (ret != K_SUCCESS) {
    printf("[MPP] VO enable_layer failed, ret=%d\n", ret);
    return ret;
  }

  printf("[MPP] VO Layer %d: %ux%u NV12 enabled\n", layer_id, width, height);
  return 0;
}

/**
 * Disable a VO layer. This releases VB blocks held by the display,
 * which is critical for clean shutdown (see file header comment).
 */
static void layer_exit(k_vo_layer_id layer_id) {
  kd_mpi_vo_disable_layer(layer_id);
}

/* ── VICAP (Camera Capture) ──────────────────────────────────────── */

/**
 * Initialize VICAP with two output channels:
 *   - CHN0: Scaled to display resolution → bound to VO (preview)
 *   - CHN1: Scaled to encode resolution → bound to VENC (WebRTC)
 *
 * The sensor is auto-detected via kd_mpi_sensor_adapt_get() using
 * the specified CSI number. Both channels output NV12 format.
 *
 * @param dev_id       VICAP device ID (typically VICAP_DEV_ID_0)
 * @param csi_num      CSI interface number (0-2, selects camera connector)
 * @param vo_width     Output width for display channel (CHN0)
 * @param vo_height    Output height for display channel (CHN0)
 * @param venc_width   Output width for encode channel (CHN1)
 * @param venc_height  Output height for encode channel (CHN1)
 */
static int vicap_init(k_vicap_dev dev_id, k_u32 csi_num,
                      k_u32 vo_width, k_u32 vo_height,
                      k_u32 venc_width, k_u32 venc_height) {
  k_vicap_dev_attr dev_attr;
  k_vicap_chn_attr chn_attr;
  k_vicap_sensor_info sensor_info;
  k_vicap_probe_config probe_cfg;
  k_vicap_sensor_type sensor_type;
  k_s32 ret;

  /* ── Sensor auto-detection ── */
  memset(&probe_cfg, 0, sizeof(probe_cfg));
  probe_cfg.csi_num = csi_num;
  probe_cfg.width = DEFAULT_ISP_WIDTH;
  probe_cfg.height = DEFAULT_ISP_HEIGHT;
  probe_cfg.fps = DEFAULT_FPS;

  ret = kd_mpi_sensor_adapt_get(&probe_cfg, &sensor_info);
  if (ret) {
    printf("[MPP] Sensor detect failed on CSI %u\n", csi_num);
    return ret;
  }

  sensor_type = sensor_info.sensor_type;

  ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
  if (ret) {
    printf("[MPP] Get sensor info failed, ret=%d\n", ret);
    return ret;
  }

  printf("[MPP] Sensor: %s (%ux%u@%ufps)\n",
         sensor_info.sensor_name, sensor_info.width, sensor_info.height, sensor_info.fps);

  /* ── Device attributes (ISP pipeline config) ── */
  memset(&dev_attr, 0, sizeof(dev_attr));
  dev_attr.acq_win.h_start = 0;
  dev_attr.acq_win.v_start = 0;
  dev_attr.acq_win.width = DEFAULT_ISP_WIDTH;
  dev_attr.acq_win.height = DEFAULT_ISP_HEIGHT;
  dev_attr.mode = VICAP_WORK_ONLINE_MODE;  /* Online mode: sensor → VICAP → output */
  dev_attr.pipe_ctrl.data = 0xFFFFFFFF;    /* Enable all ISP blocks */
  dev_attr.pipe_ctrl.bits.af_enable = 0;   /* Disable auto-focus */
  dev_attr.pipe_ctrl.bits.ahdr_enable = 0; /* Disable auto-HDR */
  dev_attr.pipe_ctrl.bits.dnr3_enable = 0; /* Disable 3D denoise */
  dev_attr.cpature_frame = 0;
  dev_attr.sensor_info = sensor_info;

  ret = kd_mpi_vicap_set_dev_attr(dev_id, dev_attr);
  if (ret) {
    printf("[MPP] VICAP set_dev_attr failed, ret=%d\n", ret);
    return ret;
  }

  /* ── Channel 0 → VO display ──
   * Scales sensor output to display resolution.
   * crop_enable=K_FALSE and scale_enable=K_FALSE means the ISP
   * hardware scaler handles the resize automatically. */
  memset(&chn_attr, 0, sizeof(chn_attr));
  chn_attr.out_win.width = vo_width;
  chn_attr.out_win.height = vo_height;
  chn_attr.crop_win = dev_attr.acq_win;
  chn_attr.scale_win = chn_attr.out_win;
  chn_attr.crop_enable = K_FALSE;
  chn_attr.scale_enable = K_FALSE;
  chn_attr.chn_enable = K_TRUE;
  chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;  /* NV12 */
  chn_attr.buffer_num = VB_BUF_COUNT;
  chn_attr.buffer_size = VB_ALIGN_UP(vo_width * vo_height * 3 / 2, 4096);  /* NV12 size, 4K aligned */
  chn_attr.buffer_pool_id = VB_INVALID_POOLID;  /* Let VICAP allocate from common pool */

  ret = kd_mpi_vicap_set_chn_attr(dev_id, g_vicap_chn_vo, chn_attr);
  if (ret) {
    printf("[MPP] VICAP set_chn_attr VO failed, ret=%d\n", ret);
    return ret;
  }

  /* ── Channel 2 → VENC encode ──
   * Scales sensor output to WebRTC encode resolution (typically 1280x720).
   * alignment=12 for VENC compatibility (H.264 requires 16-pixel
   * alignment, but VICAP channel alignment of 12 works with the
   * hardware scaler). */
  memset(&chn_attr, 0, sizeof(chn_attr));
  chn_attr.out_win.width = venc_width;
  chn_attr.out_win.height = venc_height;
  chn_attr.crop_win = dev_attr.acq_win;
  chn_attr.scale_win = chn_attr.out_win;
  chn_attr.crop_enable = K_FALSE;
  chn_attr.scale_enable = K_FALSE;
  chn_attr.chn_enable = K_TRUE;
  chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;  /* NV12 */
  chn_attr.buffer_num = VB_BUF_COUNT;
  chn_attr.buffer_size = VB_ALIGN_UP(venc_width * venc_height * 3 / 2, 4096);
  chn_attr.alignment = 12;  /* VENC-compatible alignment */
  chn_attr.buffer_pool_id = VB_INVALID_POOLID;

  ret = kd_mpi_vicap_set_chn_attr(dev_id, g_vicap_chn_venc, chn_attr);
  if (ret) {
    printf("[MPP] VICAP set_chn_attr VENC failed, ret=%d\n", ret);
    return ret;
  }

  /* Set database parse mode for ISP tuning parameters */
  ret = kd_mpi_vicap_set_database_parse_mode(dev_id, VICAP_DATABASE_PARSE_XML_JSON);
  if (ret) {
    printf("[MPP] VICAP set_database_parse_mode failed, ret=%d\n", ret);
    return ret;
  }

  ret = kd_mpi_vicap_init(dev_id);
  if (ret) {
    printf("[MPP] VICAP init failed, ret=%d\n", ret);
    return ret;
  }

  printf("[MPP] VICAP: CHN%d=%ux%u(VO), CHN%d=%ux%u(VENC)\n",
         g_vicap_chn_vo, vo_width, vo_height,
         g_vicap_chn_venc, venc_width, venc_height);

  return 0;
}

static void vicap_exit(k_vicap_dev dev_id) {
  kd_mpi_vicap_deinit(dev_id);
}

/* ── VENC (Video Encoder) ────────────────────────────────────────── */

/**
 * Create a dedicated VB pool for VENC output stream buffers.
 *
 * VENC needs its own VB pool because encoded H.264 data has different
 * size characteristics than raw YUV frames. The pool block size is
 * calculated as width*height/2 (rough upper bound for H.264 CBR output),
 * 4K-aligned.
 *
 * @return VB pool ID, or VB_INVALID_POOLID on failure
 */
static k_u32 venc_create_pool(k_u32 width, k_u32 height) {
  k_vb_pool_config pool_config;
  k_u64 stream_size = (k_u64)width * height / 2;  /* H.264 CBR worst case */

  memset(&pool_config, 0, sizeof(pool_config));
  pool_config.blk_cnt = 4;  /* 4 stream buffers (enough for 1 frame + lookahead) */
  pool_config.blk_size = ((stream_size + 0xfff) & ~0xfff);  /* 4K align */
  pool_config.mode = VB_REMAP_MODE_NOCACHE;  /* No CPU cache — DMA only */

  k_u32 pool_id = kd_mpi_vb_create_pool(&pool_config);
  printf("[MPP] VENC pool created: id=%d\n", pool_id);
  return pool_id;
}

/**
 * Initialize VENC H.264 encoder channel.
 *
 * Creates a dedicated VB pool, attaches it to the VENC channel,
 * then creates the channel with CBR rate control.
 *
 * @param chn_id         VENC channel ID (from kd_mpi_venc_request_chn)
 * @param width          Encode width in pixels
 * @param height         Encode height in pixels
 * @param bitrate_kbps   Target bitrate in kbps (CBR mode)
 * @return 0 on success, -1 on failure
 */
static int venc_init(k_u32 chn_id, k_u32 width, k_u32 height, k_u32 bitrate_kbps) {
  k_venc_chn_attr chn_attr;
  k_s32 ret;

  /* Create and attach a dedicated VB pool for this VENC channel */
  g_venc_attach_pool_id = venc_create_pool(width, height);
  if (g_venc_attach_pool_id == VB_INVALID_POOLID) {
    printf("[MPP] VENC VB create pool failed\n");
    return -1;
  }
  kd_mpi_venc_attach_vb_pool(chn_id, g_venc_attach_pool_id);

  /* Configure H.264 CBR encoding */
  memset(&chn_attr, 0, sizeof(chn_attr));
  chn_attr.venc_attr.pic_width = width;
  chn_attr.venc_attr.pic_height = height;
  chn_attr.rc_attr.rc_mode = K_VENC_RC_MODE_CBR;  /* Constant Bitrate */
  chn_attr.rc_attr.cbr.src_frame_rate = 30;         /* Input framerate */
  chn_attr.rc_attr.cbr.dst_frame_rate = 30;         /* Output framerate */
  chn_attr.rc_attr.cbr.bit_rate = bitrate_kbps;     /* Target bitrate */
  chn_attr.venc_attr.type = K_PT_H264;              /* H.264 codec */
  chn_attr.venc_attr.profile = VENC_PROFILE_H264_MAIN;  /* Main profile */

  ret = kd_mpi_venc_create_chn(chn_id, &chn_attr);
  if (ret != K_SUCCESS) {
    printf("[MPP] VENC create_chn failed: 0x%x\n", ret);
    kd_mpi_venc_detach_vb_pool(chn_id);
    kd_mpi_vb_destory_pool(g_venc_attach_pool_id);
    g_venc_attach_pool_id = VB_INVALID_POOLID;
    return -1;
  }

  /* Enable IDR (instantaneous decoder refresh) frames.
   * IDR frames allow the decoder to start from any I-frame,
   * which is essential for WebRTC where the browser may join
   * at any time. */
  ret = kd_mpi_venc_enable_idr(chn_id, K_TRUE);
  if (ret != K_SUCCESS) {
    printf("[MPP] VENC enable_idr failed: 0x%x\n", ret);
    kd_mpi_venc_destroy_chn(chn_id);
    kd_mpi_venc_detach_vb_pool(chn_id);
    kd_mpi_vb_destory_pool(g_venc_attach_pool_id);
    g_venc_attach_pool_id = VB_INVALID_POOLID;
    return -1;
  }

  printf("[MPP] VENC: chn=%d %ux%u H264 CBR %ukbps\n", chn_id, width, height, bitrate_kbps);
  return 0;
}

/**
 * Destroy VENC channel and its dedicated VB pool.
 * Must be called AFTER unbinding from VICAP.
 */
static void venc_exit(k_u32 chn_id) {
  kd_mpi_venc_detach_vb_pool(chn_id);
  kd_mpi_venc_destroy_chn(chn_id);
  if (g_venc_attach_pool_id != VB_INVALID_POOLID) {
    kd_mpi_vb_destory_pool(g_venc_attach_pool_id);
    g_venc_attach_pool_id = VB_INVALID_POOLID;
  }
}

/* ── Bind (VICAP → VO / VENC) ───────────────────────────────────── */

/** Bind VICAP channel to VO layer for display preview.
 *  After binding, VICAP frames flow automatically to VO without CPU intervention. */
static int vicap_bind_vo(k_vicap_dev dev_id, k_vicap_chn chn_id, k_vo_layer_id layer_id) {
  k_mpp_chn src, dst;

  memset(&src, 0, sizeof(src));
  src.mod_id = K_ID_VI;
  src.dev_id = dev_id;
  src.chn_id = chn_id;

  memset(&dst, 0, sizeof(dst));
  dst.mod_id = K_ID_VO;
  dst.dev_id = K_VO_DISPLAY_DEV_ID;
  dst.chn_id = layer_id;

  k_s32 ret = kd_mpi_sys_bind(&src, &dst);
  if (ret) {
    printf("[MPP] VICAP-VO bind failed, ret=0x%x\n", ret);
  }
  return ret;
}

static void vicap_unbind_vo(k_vicap_dev dev_id, k_vicap_chn chn_id, k_vo_layer_id layer_id) {
  k_mpp_chn src, dst;

  memset(&src, 0, sizeof(src));
  src.mod_id = K_ID_VI;
  src.dev_id = dev_id;
  src.chn_id = chn_id;

  memset(&dst, 0, sizeof(dst));
  dst.mod_id = K_ID_VO;
  dst.dev_id = K_VO_DISPLAY_DEV_ID;
  dst.chn_id = layer_id;

  kd_mpi_sys_unbind(&src, &dst);
}

/** Bind VICAP channel to VENC for H.264 encoding.
 *  After binding, VICAP frames flow automatically to VENC without CPU intervention. */
static int vicap_bind_venc(k_vicap_dev dev_id, k_vicap_chn chn_id, k_u32 venc_chn_id) {
  k_mpp_chn src, dst;

  memset(&src, 0, sizeof(src));
  src.mod_id = K_ID_VI;
  src.dev_id = dev_id;
  src.chn_id = chn_id;

  memset(&dst, 0, sizeof(dst));
  dst.mod_id = K_ID_VENC;
  dst.dev_id = 0;
  dst.chn_id = venc_chn_id;

  k_s32 ret = kd_mpi_sys_bind(&src, &dst);
  if (ret) {
    printf("[MPP] VICAP-VENC bind failed, ret=0x%x\n", ret);
  }
  return ret;
}

static void vicap_unbind_venc(k_vicap_dev dev_id, k_vicap_chn chn_id, k_u32 venc_chn_id) {
  k_mpp_chn src, dst;

  memset(&src, 0, sizeof(src));
  src.mod_id = K_ID_VI;
  src.dev_id = dev_id;
  src.chn_id = chn_id;

  memset(&dst, 0, sizeof(dst));
  dst.mod_id = K_ID_VENC;
  dst.dev_id = 0;
  dst.chn_id = venc_chn_id;

  kd_mpi_sys_unbind(&src, &dst);
}

/* ── Public API ──────────────────────────────────────────────────── */

/**
 * Initialize the complete MPP video pipeline.
 *
 * Sets up all hardware blocks in the correct order (see file header).
 * Each step has rollback cleanup for all previous steps on failure.
 *
 * @param config  Pipeline configuration (CSI, connector, encode params)
 * @return 0 on success, negative on failure
 */
int mpp_pipeline_init(const MppPipelineConfig* config) {
  k_s32 ret;

  if (g_pipeline_initialized) {
    printf("[MPP] Pipeline already initialized\n");
    return 0;
  }

  /* 1. VB (Video Buffer subsystem) */
  ret = vb_init();
  if (ret != 0) return ret;

  /* 2. Connector (display panel) */
  ret = connector_init(config->connector_type, &g_display_width, &g_display_height);
  if (ret != 0) { vb_exit(); return ret; }

  /* 3. VO Layer (video output overlay) */
  ret = layer_init(g_vo_layer, g_display_width, g_display_height);
  if (ret != 0) { connector_exit(); vb_exit(); return ret; }

  /* 4. VICAP (camera capture, 2 channels: CHN0→VO, CHN1→VENC) */
  ret = vicap_init(g_vicap_dev, config->csi_num,
                   g_display_width, g_display_height,
                   config->venc_width, config->venc_height);
  if (ret != 0) { layer_exit(g_vo_layer); connector_exit(); vb_exit(); return ret; }

  /* 5. VENC channel allocation + H.264 encoder init */
  ret = kd_mpi_venc_request_chn(&g_venc_chn);
  if (ret != 0) {
    printf("[MPP] VENC request_chn failed, ret=%d\n", ret);
    vicap_exit(g_vicap_dev); layer_exit(g_vo_layer); connector_exit(); vb_exit();
    return ret;
  }

  ret = venc_init(g_venc_chn, config->venc_width, config->venc_height, config->venc_bitrate_kbps);
  if (ret != 0) {
    kd_mpi_venc_release_chn(g_venc_chn);
    vicap_exit(g_vicap_dev); layer_exit(g_vo_layer); connector_exit(); vb_exit();
    return ret;
  }

  /* 6. Bind VICAP-CHN0 → VO (camera preview on LCD/HDMI) */
  ret = vicap_bind_vo(g_vicap_dev, g_vicap_chn_vo, g_vo_layer);
  if (ret != 0) {
    venc_exit(g_venc_chn); kd_mpi_venc_release_chn(g_venc_chn);
    vicap_exit(g_vicap_dev); layer_exit(g_vo_layer); connector_exit(); vb_exit();
    return ret;
  }

  /* 7. Bind VICAP-CHN1 → VENC (camera → H.264 encode → WebRTC) */
  ret = vicap_bind_venc(g_vicap_dev, g_vicap_chn_venc, g_venc_chn);
  if (ret != 0) {
    vicap_unbind_vo(g_vicap_dev, g_vicap_chn_vo, g_vo_layer);
    venc_exit(g_venc_chn); kd_mpi_venc_release_chn(g_venc_chn);
    vicap_exit(g_vicap_dev); layer_exit(g_vo_layer); connector_exit(); vb_exit();
    return ret;
  }

  g_pipeline_initialized = 1;
  printf("[MPP] Pipeline initialized: CSI%d, %ux%u encode, display %ux%u\n",
         config->csi_num, config->venc_width, config->venc_height,
         g_display_width, g_display_height);
  return 0;
}

/**
 * Start the pipeline: begin VICAP streaming and VENC encoding.
 * After this, H.264 frames are available via kd_mpi_venc_get_stream().
 */
int mpp_pipeline_start(void) {
  k_s32 ret;

  if (!g_pipeline_initialized) {
    printf("[MPP] Pipeline not initialized\n");
    return -1;
  }

  if (g_pipeline_started) {
    printf("[MPP] Pipeline already started\n");
    return 0;
  }

  ret = kd_mpi_vicap_start_stream(g_vicap_dev);
  if (ret != K_SUCCESS) {
    printf("[MPP] VICAP start_stream failed: 0x%x\n", ret);
    return ret;
  }

  ret = kd_mpi_venc_start_chn(g_venc_chn);
  if (ret != K_SUCCESS) {
    printf("[MPP] VENC start failed: 0x%x\n", ret);
    kd_mpi_vicap_stop_stream(g_vicap_dev);
    return ret;
  }

  g_pipeline_started = 1;
  printf("[MPP] Pipeline started: camera capture + LCD/HDMI display + H264 encode\n");
  return 0;
}

k_u32 mpp_pipeline_get_venc_chn(void) {
  return g_venc_chn;
}

/**
 * Stop and deinitialize the complete MPP pipeline.
 *
 * ⚠ CRITICAL: Resource release order matters!
 *
 * 1. Stop VENC first (stops producing encoded data)
 * 2. Stop VICAP (stops producing raw frames)
 * 3. Drain remaining VENC frames (releases VB blocks in encoder)
 * 4. Unbind VICAP-VO and VICAP-VENC (breaks data flow paths)
 * 5. Disable VO layer — THIS releases VB blocks held by the display.
 *    If this is done too late, VB blocks remain locked and the
 *    system reports "VB get block failed" errors.
 * 6. Deinit VICAP
 * 7. Deinit VENC + destroy its VB pool
 * 8. Deinit connector + VB system
 */
void mpp_pipeline_deinit(void) {
  if (!g_pipeline_initialized) return;

  if (g_pipeline_started) {
    /* Stop producing: VENC first, then VICAP */
    kd_mpi_venc_stop_chn(g_venc_chn);
    kd_mpi_vicap_stop_stream(g_vicap_dev);

    /* Drain remaining VENC frames to release their VB blocks.
     * Without this, VENC holds references to VB blocks that
     * would prevent clean deinit. */
    k_venc_stream output;
    k_venc_chn_status status;
    k_venc_pack packs[VENC_MAX_PACK_CNT];
    while (1) {
      kd_mpi_venc_query_status(g_venc_chn, &status);
      if (status.cur_packs <= 0) break;
      output.pack_cnt = status.cur_packs > VENC_MAX_PACK_CNT ? VENC_MAX_PACK_CNT : status.cur_packs;
      output.pack = packs;
      if (kd_mpi_venc_get_stream(g_venc_chn, &output, 1000) != K_SUCCESS) break;
      kd_mpi_venc_release_stream(g_venc_chn, &output);
    }

    g_pipeline_started = 0;
  }

  /* Unbind data flow paths */
  vicap_unbind_venc(g_vicap_dev, g_vicap_chn_venc, g_venc_chn);
  vicap_unbind_vo(g_vicap_dev, g_vicap_chn_vo, g_vo_layer);

  /* Brief delay to allow in-flight frames to complete after unbind.
   * One frame period at 30fps ≈ 33ms. */
  usleep(1000 * (1000 / 30));

  /* ⚠ CRITICAL: Disable VO layer BEFORE VICAP/VENC deinit.
   * VO holds VB blocks for display; disabling it releases them.
   * If VICAP/VENC are deinitialized first, their VB pools are
   * destroyed while VO still holds references → VB leak errors. */
  layer_exit(g_vo_layer);

  /* Now safe to deinit VICAP (releases its VB channels) */
  vicap_exit(g_vicap_dev);

  /* Deinit VENC and destroy its dedicated VB pool */
  venc_exit(g_venc_chn);
  kd_mpi_venc_release_chn(g_venc_chn);

  /* Finally, deinit display connector and VB system */
  connector_exit();
  vb_exit();

  g_pipeline_initialized = 0;
  printf("[MPP] Pipeline deinitialized\n");
}
