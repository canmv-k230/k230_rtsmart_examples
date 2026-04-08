#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

#include "drv_input.h"

static int reconnect_input_device(drv_input_inst_t *inst)
{
    int ret = drv_input_inst_try_reconnect(inst);

    if (ret == 0)
        printf("Reconnected input device: %s (%s)\n", inst->path, inst->info.name);

    return ret;
}

/* Helper function to print key name */
static const char *key_name(uint16_t code)
{
    static char buf[32];

    switch (code)
    {
        /* Modifier keys */
        case KEY_LEFTCTRL: return "LEFTCTRL";
        case KEY_LEFTSHIFT: return "LEFTSHIFT";
        case KEY_LEFTALT: return "LEFTALT";
        case KEY_LEFTMETA: return "LEFTMETA";
        case KEY_RIGHTCTRL: return "RIGHTCTRL";
        case KEY_RIGHTSHIFT: return "RIGHTSHIFT";
        case KEY_RIGHTALT: return "RIGHTALT";
        case KEY_RIGHTMETA: return "RIGHTMETA";

        /* Letter keys */
        case KEY_A: return "A";
        case KEY_B: return "B";
        case KEY_C: return "C";
        case KEY_D: return "D";
        case KEY_E: return "E";
        case KEY_F: return "F";
        case KEY_G: return "G";
        case KEY_H: return "H";
        case KEY_I: return "I";
        case KEY_J: return "J";
        case KEY_K: return "K";
        case KEY_L: return "L";
        case KEY_M: return "M";
        case KEY_N: return "N";
        case KEY_O: return "O";
        case KEY_P: return "P";
        case KEY_Q: return "Q";
        case KEY_R: return "R";
        case KEY_S: return "S";
        case KEY_T: return "T";
        case KEY_U: return "U";
        case KEY_V: return "V";
        case KEY_W: return "W";
        case KEY_X: return "X";
        case KEY_Y: return "Y";
        case KEY_Z: return "Z";

        /* Number keys */
        case KEY_1: return "1";
        case KEY_2: return "2";
        case KEY_3: return "3";
        case KEY_4: return "4";
        case KEY_5: return "5";
        case KEY_6: return "6";
        case KEY_7: return "7";
        case KEY_8: return "8";
        case KEY_9: return "9";
        case KEY_0: return "0";

        /* Control keys */
        case KEY_ENTER: return "ENTER";
        case KEY_ESC: return "ESC";
        case KEY_BACKSPACE: return "BACKSPACE";
        case KEY_TAB: return "TAB";
        case KEY_SPACE: return "SPACE";
        case KEY_MINUS: return "MINUS";
        case KEY_EQUAL: return "EQUAL";

        /* Arrow keys */
        case KEY_RIGHT: return "RIGHT";
        case KEY_LEFT: return "LEFT";
        case KEY_DOWN: return "DOWN";
        case KEY_UP: return "UP";

        /* Navigation keys */
        case KEY_HOME: return "HOME";
        case KEY_END: return "END";
        case KEY_PAGEUP: return "PAGEUP";
        case KEY_PAGEDOWN: return "PAGEDOWN";
        case KEY_INSERT: return "INSERT";
        case KEY_DELETE: return "DELETE";

        /* Function keys */
        case KEY_F1: return "F1";
        case KEY_F2: return "F2";
        case KEY_F3: return "F3";
        case KEY_F4: return "F4";
        case KEY_F5: return "F5";
        case KEY_F6: return "F6";
        case KEY_F7: return "F7";
        case KEY_F8: return "F8";
        case KEY_F9: return "F9";
        case KEY_F10: return "F10";
        case KEY_F11: return "F11";
        case KEY_F12: return "F12";

        default:
            snprintf(buf, sizeof(buf), "KEY_%d", code);
            return buf;
    }
}

/* Test 1: Blocking read operation */
int test_blocking_read(char *dev_path, size_t dev_path_size, bool auto_detect)
{
    drv_input_inst_t *inst = NULL;
    struct drv_keyboard_frame frame;
    int count = 0;
    int max_events = 10;

    printf("\n=== Test 1: Blocking Read ===\n");
    printf("This tests the blocking read behavior and rx_indicate callback\n");
    printf("Opening %s in blocking mode...\n", dev_path);

    if (drv_input_inst_create_path(dev_path, &inst) != 0)
    {
        perror("Failed to open device");
        return -1;
    }

    if (auto_detect)
        drv_input_inst_set_auto_reconnect(inst, DRV_INPUT_DEV_KEYBOARD);

    printf("Device opened successfully (fd=%d)\n", inst->fd);
    printf("Press keys on USB keyboard (will read %d event frames)...\n", max_events);
    printf("The read() call will BLOCK until data is available\n");
    printf("Note: Each frame ends with EV_SYN event\n\n");

    while (count < max_events)
    {
        int ret = drv_input_poll(inst, -1);
        if (ret < 0)
        {
            if (!drv_input_inst_is_connected(inst))
            {
                printf("Keyboard disconnected, waiting for reconnect...\n");
                while (reconnect_input_device(inst) != 0)
                    usleep(200000);
                continue;
            }
            perror("Poll error");
            break;
        }

        ret = drv_input_read_keyboard_frame(inst, &frame);
        if (ret <= 0)
            continue;

        for (size_t index = 0; index < frame.count; index++)
        {
                printf("    EV_KEY: %s -> %s\n",
                       key_name(frame.keycodes[index]),
                       frame.values[index] == KEY_PRESSED ? "PRESSED" : "RELEASED");
        }

        if (frame.complete)
        {
            printf("    EV_SYN: --- frame %d end ---\n\n", count + 1);
            count++;
        }
    }

    drv_input_inst_destroy(&inst);
    printf("Test 1 completed: Read %d event frames\n", count);
    return 0;
}

/* Test 2: Non-blocking read operation */
int test_nonblocking_read(char *dev_path, size_t dev_path_size, bool auto_detect)
{
    drv_input_inst_t *inst = NULL;
    struct drv_keyboard_frame frame;
    int empty_reads = 0;
    int event_count = 0;
    int frame_count = 0;
    int loop_count = 0;
    int max_loops = 500; /* 5 seconds at 10ms per loop */

    printf("\n=== Test 2: Non-blocking Read ===\n");
    printf("This tests non-blocking read with O_NONBLOCK flag\n");

    if (drv_input_inst_create_path(dev_path, &inst) != 0)
    {
        perror("Failed to open device");
        return -1;
    }

    if (auto_detect)
        drv_input_inst_set_auto_reconnect(inst, DRV_INPUT_DEV_KEYBOARD);

    printf("Device opened in non-blocking mode (fd=%d)\n", inst->fd);
    printf("Press keys on USB keyboard (test will run for 5 seconds)...\n");
    printf("Non-blocking reads will return -EAGAIN when no data available\n\n");

    while (loop_count < max_loops)
    {
        int ret = drv_input_read_keyboard_frame(inst, &frame);

        if (ret <= 0)
        {
            empty_reads++;
        }
        else
        {
            for (size_t index = 0; index < frame.count; index++)
            {
                printf("    EV_KEY: %s -> %s\n",
                       key_name(frame.keycodes[index]),
                       frame.values[index] == KEY_PRESSED ? "PRESSED" : "RELEASED");
                event_count++;
            }

            if (frame.complete)
            {
                printf("    EV_SYN: --- frame end ---\n\n");
                frame_count++;
            }
        }

        usleep(10000); /* 10ms delay */
        loop_count++;
    }

    drv_input_inst_destroy(&inst);
    printf("Test 2 completed: %d empty reads (EAGAIN), %d events, %d frames\n",
           empty_reads, event_count, frame_count);
    return 0;
}

/* Test 3: Poll functionality */
int test_poll(char *dev_path, size_t dev_path_size, bool auto_detect)
{
    drv_input_inst_t *inst = NULL;
    struct drv_keyboard_frame frame;
    int event_count = 0;
    int poll_count = 0;
    int max_events = 10;

    printf("\n=== Test 3: Poll Functionality ===\n");
    printf("This tests poll() for event notification\n");

    if (drv_input_inst_create_path(dev_path, &inst) != 0)
    {
        perror("Failed to open device");
        return -1;
    }

    if (auto_detect)
        drv_input_inst_set_auto_reconnect(inst, DRV_INPUT_DEV_KEYBOARD);

    printf("Device opened (fd=%d)\n", inst->fd);
    printf("Press keys on USB keyboard (will read %d events)...\n", max_events);
    printf("Using poll() with 1 second timeout\n");

    while (event_count < max_events)
    {
        printf("  Polling for events (timeout=1000ms)...\n");

        int ret = drv_input_poll(inst, 1000);
        poll_count++;

        if (ret < 0)
        {
            if (!drv_input_inst_is_connected(inst))
            {
                printf("Keyboard disconnected, waiting for reconnect...\n");
                while (reconnect_input_device(inst) != 0)
                    usleep(200000);
                continue;
            }
            perror("Poll error");
            break;
        }
        else if (ret == 0)
        {
            printf("  Poll timeout (no events)\n");
        }
        else
        {
            printf("  Poll returned POLLIN - data available\n");

            while (1)
            {
                ret = drv_input_read_keyboard_frame(inst, &frame);
                if (ret <= 0)
                    break;

                for (size_t index = 0; index < frame.count; index++)
                {
                    printf("    [%02d] %s: %s\n",
                           event_count + 1,
                           key_name(frame.keycodes[index]),
                           frame.values[index] == KEY_PRESSED ? "PRESSED" : "RELEASED");
                    event_count++;

                    if (event_count >= max_events)
                        break;
                }

                if (event_count >= max_events)
                    break;
            }
        }
    }

    drv_input_inst_destroy(&inst);
    printf("Test 3 completed: %d poll calls, %d events read\n",
           poll_count, event_count);
    return 0;
}


int main(int argc, char **argv)
{
    const char *dev_path = "/dev/input/event0";
    char detected_path[DRV_INPUT_PATH_MAX];
    char active_path[DRV_INPUT_PATH_MAX];
    struct drv_input_info info;
    bool auto_detect = false;

    if (argc > 1)
    {
        dev_path = argv[1];
    }
    else if (drv_input_find_first_by_type(DRV_INPUT_DEV_KEYBOARD,
                                          detected_path,
                                          sizeof(detected_path),
                                          &info) == 0)
    {
        dev_path = detected_path;
        auto_detect = true;
    }

    snprintf(active_path, sizeof(active_path), "%s", dev_path);
    active_path[sizeof(active_path) - 1] = '\0';

    printf("========================================\n");
    printf("HID Keyboard Driver Comprehensive Test\n");
    printf("========================================\n");
    printf("Device: %s\n", dev_path);
    if (argc <= 1)
        printf("Detected device: %s (%s)\n", dev_path, info.name);
    printf("Event size: %zu bytes\n", sizeof(struct input_event));
    printf("\n");

    /* Run all tests */
    if (test_blocking_read(active_path, sizeof(active_path), auto_detect) != 0)
    {
        printf("\nTest 1 FAILED\n");
        return 1;
    }

    if (test_nonblocking_read(active_path, sizeof(active_path), auto_detect) != 0)
    {
        printf("\nTest 2 FAILED\n");
        return 1;
    }

    if (test_poll(active_path, sizeof(active_path), auto_detect) != 0)
    {
        printf("\nTest 3 FAILED\n");
        return 1;
    }

    printf("\n========================================\n");
    printf("All tests PASSED successfully!\n");
    printf("========================================\n");

    return 0;
}
