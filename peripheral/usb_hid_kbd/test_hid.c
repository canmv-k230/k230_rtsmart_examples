#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#include "hid_keyboard.h"

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
int test_blocking_read(const char *dev_path)
{
    int fd;
    struct hid_keyboard_event event;
    int count = 0;
    int max_events = 10;

    printf("\n=== Test 1: Blocking Read ===\n");
    printf("This tests the blocking read behavior and rx_indicate callback\n");
    printf("Opening %s in blocking mode...\n", dev_path);

    fd = open(dev_path, O_RDONLY);
    if (fd < 0)
    {
        perror("Failed to open device");
        return -1;
    }

    printf("Device opened successfully (fd=%d)\n", fd);
    printf("Press keys on USB keyboard (will read %d event frames)...\n", max_events);
    printf("The read() call will BLOCK until data is available\n");
    printf("Note: Each frame ends with EV_SYN event\n\n");

    while (count < max_events)
    {
        ssize_t ret = read(fd, &event, sizeof(event));

        if (ret < 0)
        {
            perror("Read error");
            break;
        }

        if (ret == sizeof(event))
        {
            if (event.type == EV_KEY)
            {
                printf("    EV_KEY: %s -> %s\n",
                       key_name(event.code),
                       event.value == KEY_PRESSED ? "PRESSED" : "RELEASED");
            }
            else if (event.type == EV_SYN)
            {
                printf("    EV_SYN: --- frame %d end ---\n\n", count + 1);
                count++;
            }
        }
    }

    close(fd);
    printf("Test 1 completed: Read %d event frames\n", count);
    return 0;
}

/* Test 2: Non-blocking read operation */
int test_nonblocking_read(const char *dev_path)
{
    int fd;
    struct hid_keyboard_event event;
    int empty_reads = 0;
    int event_count = 0;
    int frame_count = 0;
    int loop_count = 0;
    int max_loops = 500; /* 5 seconds at 10ms per loop */

    printf("\n=== Test 2: Non-blocking Read ===\n");
    printf("This tests non-blocking read with O_NONBLOCK flag\n");

    fd = open(dev_path, O_RDONLY | O_NONBLOCK);
    if (fd < 0)
    {
        perror("Failed to open device");
        return -1;
    }

    printf("Device opened in non-blocking mode (fd=%d)\n", fd);
    printf("Press keys on USB keyboard (test will run for 5 seconds)...\n");
    printf("Non-blocking reads will return -EAGAIN when no data available\n\n");

    while (loop_count < max_loops)
    {
        ssize_t ret = read(fd, &event, sizeof(event));

        if (ret < 0)
        {
            if (errno == EAGAIN)
            {
                empty_reads++;
            }
            else
            {
                perror("Unexpected read error");
                break;
            }
        }
        else if (ret == sizeof(event))
        {
            if (event.type == EV_KEY)
            {
                printf("    EV_KEY: %s -> %s\n",
                       key_name(event.code),
                       event.value == KEY_PRESSED ? "PRESSED" : "RELEASED");
                event_count++;
            }
            else if (event.type == EV_SYN)
            {
                printf("    EV_SYN: --- frame end ---\n\n");
                frame_count++;
            }
        }

        usleep(10000); /* 10ms delay */
        loop_count++;
    }

    close(fd);
    printf("Test 2 completed: %d empty reads (EAGAIN), %d events, %d frames\n",
           empty_reads, event_count, frame_count);
    return 0;
}

/* Test 3: Poll functionality */
int test_poll(const char *dev_path)
{
    int fd;
    struct hid_keyboard_event event;
    struct pollfd pfd;
    int event_count = 0;
    int poll_count = 0;
    int max_events = 10;

    printf("\n=== Test 3: Poll Functionality ===\n");
    printf("This tests poll() for event notification\n");

    fd = open(dev_path, O_RDONLY);
    if (fd < 0)
    {
        perror("Failed to open device");
        return -1;
    }

    printf("Device opened (fd=%d)\n", fd);
    printf("Press keys on USB keyboard (will read %d events)...\n", max_events);
    printf("Using poll() with 1 second timeout\n");

    pfd.fd = fd;
    pfd.events = POLLIN;

    while (event_count < max_events)
    {
        printf("  Polling for events (timeout=1000ms)...\n");

        int ret = poll(&pfd, 1, 1000);
        poll_count++;

        if (ret < 0)
        {
            perror("Poll error");
            break;
        }
        else if (ret == 0)
        {
            printf("  Poll timeout (no events)\n");
        }
        else if (pfd.revents & POLLIN)
        {
            printf("  Poll returned POLLIN - data available\n");

            /* Read all available events */
            while (1)
            {
                ssize_t rret = read(fd, &event, sizeof(event));
                if (rret < 0)
                {
                    if (errno != EAGAIN)
                        perror("Read error after poll");
                    break;
                }
                else if (rret == sizeof(event) && event.type == EV_KEY)
                {
                    printf("    [%02d] %s: %s\n",
                           event_count + 1,
                           key_name(event.code),
                           event.value == KEY_PRESSED ? "PRESSED" : "RELEASED");
                    event_count++;

                    if (event_count >= max_events)
                        break;
                }
            }
        }
    }

    close(fd);
    printf("Test 3 completed: %d poll calls, %d events read\n",
           poll_count, event_count);
    return 0;
}


int main(int argc, char **argv)
{
    const char *dev_path = "/dev/hidk0";

    if (argc > 1)
    {
        dev_path = argv[1];
    }

    printf("========================================\n");
    printf("HID Keyboard Driver Comprehensive Test\n");
    printf("========================================\n");
    printf("Device: %s\n", dev_path);
    printf("Event size: %zu bytes\n", sizeof(struct hid_keyboard_event));
    printf("\n");

    /* Run all tests */
    if (test_blocking_read(dev_path) != 0)
    {
        printf("\nTest 1 FAILED\n");
        return 1;
    }

    if (test_nonblocking_read(dev_path) != 0)
    {
        printf("\nTest 2 FAILED\n");
        return 1;
    }

    if (test_poll(dev_path) != 0)
    {
        printf("\nTest 3 FAILED\n");
        return 1;
    }

    printf("\n========================================\n");
    printf("All tests PASSED successfully!\n");
    printf("========================================\n");

    return 0;
}
