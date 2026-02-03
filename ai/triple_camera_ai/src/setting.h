
#define DISPLAY_TYPE 'lt9611'

#if DISPLAY_TYPE == 'st7701'
    #define ISP_WIDTH 1920
    #define ISP_HEIGHT 1080
    #define DISPLAY_MODE 1
    #define DISPLAY_WIDTH 400
    #define DISPLAY_HEIGHT 240
    #define DISPLAY_ROTATE 1
    #define AI_FRAME_WIDTH 640
    #define AI_FRAME_HEIGHT 360
    #define AI_FRAME_CHANNEL 3
    #define USE_OSD 1
    #define OSD_WIDTH 400
    #define OSD_HEIGHT 240
    #define OSD_CHANNEL 4
#elif DISPLAY_TYPE == 'lt9611'
    #define ISP_WIDTH 1920
    #define ISP_HEIGHT 1080
    #define DISPLAY_MODE 0
    #define DISPLAY_WIDTH 960
    #define DISPLAY_HEIGHT 540
    #define DISPLAY_ROTATE 0
    #define AI_FRAME_WIDTH 640
    #define AI_FRAME_HEIGHT 360
    #define AI_FRAME_CHANNEL 3
    #define USE_OSD 1
    #define OSD_WIDTH 960
    #define OSD_HEIGHT 540
    #define OSD_CHANNEL 4
 #endif