#ifndef _PIPELINE_TYPES_H
#define _PIPELINE_TYPES_H
#include <cstdint>

/**
 * @brief Structure used to dump frame buffer information
 * including virtual and physical addresses.
 */
struct DumpRes {
    uintptr_t virt_addr;
    uintptr_t phy_addr;
};

enum DisplayType {
    DISPLAY_LCD,  ///< LCD (Liquid Crystal Display) output interface
    DISPLAY_HDMI  ///< HDMI (High-Definition Multimedia Interface) output interface
};

#endif