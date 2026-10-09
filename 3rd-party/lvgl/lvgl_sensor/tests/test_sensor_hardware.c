#include <stdbool.h>
#include <time.h>

#include "lvgl.h"
#include "mpi_vo_api.h"

static uint32_t sensor_test_timer_handler(void);
static void sensor_test_set_src(lv_obj_t *obj, const void *src);

/* Exercise the application unchanged, with bounded runtime and observations
 * at its image submission and timer calls. All media and drawing calls are real. */
#define main sensor_application_main
#define lv_timer_handler sensor_test_timer_handler
#define lv_image_set_src sensor_test_set_src
#include "../lvgl_sensor.c"
#undef lv_image_set_src
#undef lv_timer_handler
#undef main

static unsigned submitted_frames;
static unsigned display_flushes;
static bool checked_display;
static bool display_ok;
static uint64_t started_ms;

static uint64_t monotonic_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void count_flush(lv_event_t *event)
{
    (void)event;
    display_flushes++;
}

static void sensor_test_set_src(lv_obj_t *obj, const void *src)
{
    lv_image_set_src(obj, src);
    if (obj == img_widget && src == &sensor_dsc && sensor_dsc.data) submitted_frames++;
}

static uint32_t sensor_test_timer_handler(void)
{
    if (!checked_display && g_display) {
        k_vo_dev_attr attr = {0};
        k_gdma_rotation_e expected = GDMA_ROTATE_DEGREE_0;
        switch (g_rotation_degrees) {
        case 90: expected = GDMA_ROTATE_DEGREE_270; break;
        case 180: expected = GDMA_ROTATE_DEGREE_180; break;
        case 270: expected = GDMA_ROTATE_DEGREE_90; break;
        }
        int width = lv_display_get_horizontal_resolution(g_display);
        int height = lv_display_get_vertical_resolution(g_display);
        bool portrait = g_rotation_degrees == 0 || g_rotation_degrees == 180;
        display_ok = kd_mpi_vo_get_dev_attr(&attr) == 0 && attr.dev_rot_flg == expected &&
                     width == (portrait ? 480 : 800) && height == (portrait ? 800 : 480);
        printf("Sensor rotation: %s CLI=%d GDMA=%d expected=%d logical=%dx%d\n",
               display_ok ? "PASS" : "FAIL", g_rotation_degrees, attr.dev_rot_flg,
               expected, width, height);
        lv_display_add_event_cb(g_display, count_flush, LV_EVENT_FLUSH_FINISH, NULL);
        checked_display = true;
    }
    uint32_t delay = lv_timer_handler();
    if (submitted_frames >= 120 || monotonic_ms() - started_ms >= 10000) g_app_run = 0;
    return delay;
}

int main(int argc, char **argv)
{
    started_ms = monotonic_ms();
    int result = sensor_application_main(argc, argv);
    bool cleaned = !g_display;
    for (unsigned i = 0; i < FRAME_MAP_CACHE_SIZE; i++) cleaned &= !g_frame_maps[i].va;
    bool ok = result == 0 && display_ok && submitted_frames >= 120 &&
              display_flushes >= 10 && cleaned;
    printf("Sensor hardware: %s frames=%u flushes=%u cleanup=%d elapsed_ms=%llu\n",
           ok ? "PASS" : "FAIL", submitted_frames, display_flushes, cleaned,
           (unsigned long long)(monotonic_ms() - started_ms));
    return ok ? 0 : 1;
}
