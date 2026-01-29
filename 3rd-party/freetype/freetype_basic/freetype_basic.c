#include "freetype_wrap.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BMP Header structures (must be packed to avoid padding)
#pragma pack(push, 1)
typedef struct {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BMPHeader;

typedef struct {
    uint32_t size;
    int32_t  width;
    int32_t  height;
    uint16_t planes;
    uint16_t bits;
    uint32_t compression;
    uint32_t imagesize;
    int32_t  xresolution;
    int32_t  yresolution;
    uint32_t ncolours;
    uint32_t importantcolours;
} BMPInfoHeader;
#pragma pack(pop)

typedef struct {
    uint8_t* buffer;
    int      width;
    int      height;
} ImageCanvas;

/**
 * Callback function required by freetype_wrap.
 * It maps the FreeType bitmap onto our ImageCanvas.
 */
void my_draw_bitmap_callback(void* ctx, int color, int x, int y, FT_Bitmap* bitmap)
{
    ImageCanvas* canvas = (ImageCanvas*)ctx;

    for (int row = 0; row < bitmap->rows; row++) {
        for (int col = 0; col < bitmap->width; col++) {
            int canvas_x = x + col;
            int canvas_y = y + row;

            // Bounds checking
            if (canvas_x < 0 || canvas_x >= canvas->width || canvas_y < 0 || canvas_y >= canvas->height)
                continue;

            // FreeType provides an 8-bit grayscale buffer
            uint8_t alpha = bitmap->buffer[row * bitmap->pitch + col];
            if (alpha > 0) {
                // For this test, we treat 'color' as a simple RGB hex (e.g., 0xFFFFFF for white)
                // We calculate the pixel index in a 24-bit (3 bytes per pixel) buffer
                int idx = (canvas_y * canvas->width + canvas_x) * 3;

                // Simple alpha blending/replacement
                canvas->buffer[idx + 0] = (color & 0xFF); // Blue
                canvas->buffer[idx + 1] = ((color >> 8) & 0xFF); // Green
                canvas->buffer[idx + 2] = ((color >> 16) & 0xFF); // Red
            }
        }
    }
}

/**
 * Saves a 24-bit RGB buffer to a BMP file.
 * Note: BMP files are stored bottom-to-top.
 */
void save_bmp(const char* filename, ImageCanvas* canvas)
{
    FILE* f = fopen(filename, "wb");
    if (!f)
        return;

    int      row_padding = (4 - (canvas->width * 3) % 4) % 4;
    uint32_t filesize    = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + (canvas->width * 3 + row_padding) * canvas->height;

    BMPHeader     header = { 0x4D42, filesize, 0, 0, 54 };
    BMPInfoHeader info   = { 40, canvas->width, canvas->height, 1, 24, 0, 0, 0, 0, 0, 0 };

    fwrite(&header, sizeof(header), 1, f);
    fwrite(&info, sizeof(info), 1, f);

    uint8_t padding_bytes[3] = { 0, 0, 0 };
    // Write rows from bottom to top for BMP format
    for (int i = canvas->height - 1; i >= 0; i--) {
        fwrite(&canvas->buffer[i * canvas->width * 3], 3, canvas->width, f);
        fwrite(padding_bytes, 1, row_padding, f);
    }

    fclose(f);
    printf("Saved image to %s\n", filename);
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: %s <font_path> <output_bmp_path>\n", argv[0]);
        return -1;
    }

    const char* font_path = argv[1];
    const char* save_path = argv[2];
    const char* text      = "Hello Canaan!";

    // Create a 640x480 black canvas
    ImageCanvas canvas;
    canvas.width  = 640;
    canvas.height = 480;
    canvas.buffer = (uint8_t*)calloc(canvas.width * canvas.height * 3, 1);

    printf("Rendering text: '%s' using font: %s\n", text, font_path);

    // Render string using freetype_wrap
    // Parameters: x, y, font_size, string, color (White), font_path, callback, context
    int ret = freetype_wrap_draw_string(50, 200, 64, text, 0xFFFFFF, font_path, my_draw_bitmap_callback, &canvas);

    if (ret != FREETYPE_WRAP_ERR_NONE) {
        printf("Freetype wrap error: %d\n", ret);
    } else {
        save_bmp(save_path, &canvas);
    }

    // Cleanup
    freetype_wrap_deinit();
    free(canvas.buffer);

    return 0;
}
