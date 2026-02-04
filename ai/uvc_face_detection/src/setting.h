
#define DISPLAY_TYPE 'st7701'

#if DISPLAY_TYPE == 'st7701' 
    #define UVC_WIDTH 640
    #define UVC_HEIGHT 480
    #define DISPLAY_MODE 1   
    #define DISPLAY_WIDTH 640
    #define DISPLAY_HEIGHT 480
    #define DISPLAY_ROTATE 1
    #define AI_FRAME_WIDTH 640
    #define AI_FRAME_HEIGHT 480
    #define AI_FRAME_CHANNEL 3
#elif DISPLAY_TYPE == 'lt9611'
    #define UVC_WIDTH 1920
    #define UVC_HEIGHT 1080
    #define DISPLAY_MODE 0   
    #define DISPLAY_WIDTH 1920
    #define DISPLAY_HEIGHT 1080
    #define DISPLAY_ROTATE 0
    #define AI_FRAME_WIDTH 1920
    #define AI_FRAME_HEIGHT 1080
    #define AI_FRAME_CHANNEL 3
 #endif