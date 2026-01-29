#define DISPLAY_TYPE 'st7701'   // Select display type: 'st7701' or 'lt9611'

#if DISPLAY_TYPE == 'st7701'

    // ISP (Image Signal Processor) output resolution
    #define ISP_WIDTH 1920
    #define ISP_HEIGHT 1080

    // Display configuration
    #define DISPLAY_MODE 1       // Display mode selector (platform-specific)
    #define DISPLAY_WIDTH 800    // Physical display width
    #define DISPLAY_HEIGHT 480   // Physical display height
    #define DISPLAY_ROTATE 1     // Display rotation enable (e.g., 90/180 degrees)

    // AI input frame configuration
    #define AI_FRAME_WIDTH 640
    #define AI_FRAME_HEIGHT 360
    #define AI_FRAME_CHANNEL 3   // Number of color channels (e.g., RGB)

    // OSD (On-Screen Display) configuration
    #define USE_OSD 1            // Enable OSD overlay
    #define OSD_WIDTH 800
    #define OSD_HEIGHT 480
    #define OSD_CHANNEL 4        // OSD color channels (e.g., ARGB)

#elif DISPLAY_TYPE == 'lt9611'

    // ISP (Image Signal Processor) output resolution
    #define ISP_WIDTH 1920
    #define ISP_HEIGHT 1080

    // Display configuration
    #define DISPLAY_MODE 0       // Display mode selector (platform-specific)
    #define DISPLAY_WIDTH 1920   // Physical display width
    #define DISPLAY_HEIGHT 1080  // Physical display height
    #define DISPLAY_ROTATE 0     // Display rotation disabled

    // AI input frame configuration
    #define AI_FRAME_WIDTH 1280
    #define AI_FRAME_HEIGHT 720
    #define AI_FRAME_CHANNEL 3   // Number of color channels (e.g., RGB)

    // OSD (On-Screen Display) configuration
    #define USE_OSD 1            // Enable OSD overlay
    #define OSD_WIDTH 1920
    #define OSD_HEIGHT 1080
    #define OSD_CHANNEL 4        // OSD color channels (e.g., ARGB)

#endif
