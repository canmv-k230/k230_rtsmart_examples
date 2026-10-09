#include <assert.h>

#define main sensor_application_main
#include "../lvgl_sensor.c"
#undef main

int main(void)
{
    assert(g_rotation_degrees == 270);
    assert(g_display_rotation == GDMA_ROTATE_DEGREE_90);
    char *defaults[] = {"sensor", NULL};
    optind = 0;
    assert(parse_arguments(1, defaults) == 0);
    assert(g_display_rotation == GDMA_ROTATE_DEGREE_90);

    const char *angles[] = {"0", "90", "180", "270"};
    const k_gdma_rotation_e expected[] = {GDMA_ROTATE_DEGREE_0, GDMA_ROTATE_DEGREE_270,
                                         GDMA_ROTATE_DEGREE_180, GDMA_ROTATE_DEGREE_90};
    const char *options[] = {"-r", "--rotate", "-rotate"};
    for (unsigned option = 0; option < sizeof(options) / sizeof(options[0]); option++) {
        for (unsigned angle = 0; angle < sizeof(angles) / sizeof(angles[0]); angle++) {
            char *args[] = {"sensor", (char *)options[option], (char *)angles[angle], NULL};
            optind = 0;
            assert(parse_arguments(3, args) == 0);
            assert(g_rotation_degrees == atoi(angles[angle]));
            assert(g_display_rotation == expected[angle]);
        }
    }
    puts("sensor default and all rotation options: PASS (13 cases)");
    return 0;
}
