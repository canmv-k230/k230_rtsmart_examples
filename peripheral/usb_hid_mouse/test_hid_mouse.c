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

static const char *button_name_from_mask(uint32_t mask)
{
    switch (mask)
    {
        case 1u << 0: return "LEFT";
        case 1u << 1: return "RIGHT";
        case 1u << 2: return "MIDDLE";
        case 1u << 3: return "SIDE";
        case 1u << 4: return "EXTRA";
        case 1u << 5: return "FORWARD";
        case 1u << 6: return "BACK";
        case 1u << 7: return "TOUCH";
        default: return "BUTTON";
    }
}

static const char *axis_name(uint16_t code, uint16_t type)
{
    if (type == EV_REL)
    {
        switch (code)
        {
            case REL_X: return "REL_X";
            case REL_Y: return "REL_Y";
            case REL_WHEEL: return "REL_WHEEL";
            case REL_HWHEEL: return "REL_HWHEEL";
            default: return "REL_UNKNOWN";
        }
    }

    switch (code)
    {
        case ABS_X: return "ABS_X";
        case ABS_Y: return "ABS_Y";
        case ABS_PRESSURE: return "ABS_PRESSURE";
        default: return "ABS_UNKNOWN";
    }
}

static void print_pointer_frame(const struct drv_pointer_frame *frame)
{
    uint32_t changed_mask;

    changed_mask = frame->pressed_mask;
    while (changed_mask)
    {
        uint32_t bit = changed_mask & (~changed_mask + 1u);
        printf("    EV_KEY: %s -> PRESSED\n", button_name_from_mask(bit));
        changed_mask &= ~bit;
    }

    changed_mask = frame->released_mask;
    while (changed_mask)
    {
        uint32_t bit = changed_mask & (~changed_mask + 1u);
        printf("    EV_KEY: %s -> RELEASED\n", button_name_from_mask(bit));
        changed_mask &= ~bit;
    }

    if (frame->has_rel)
    {
        if (frame->rel_x != 0)
            printf("    EV_REL: %s -> %d\n", axis_name(REL_X, EV_REL), frame->rel_x);
        if (frame->rel_y != 0)
            printf("    EV_REL: %s -> %d\n", axis_name(REL_Y, EV_REL), frame->rel_y);
        if (frame->wheel != 0)
            printf("    EV_REL: %s -> %d\n", axis_name(REL_WHEEL, EV_REL), frame->wheel);
        if (frame->hwheel != 0)
            printf("    EV_REL: %s -> %d\n", axis_name(REL_HWHEEL, EV_REL), frame->hwheel);
    }

    if (frame->has_abs)
    {
        printf("    EV_ABS: %s -> %d\n", axis_name(ABS_X, EV_ABS), frame->abs_x);
        printf("    EV_ABS: %s -> %d\n", axis_name(ABS_Y, EV_ABS), frame->abs_y);
        if (frame->pressure != 0)
            printf("    EV_ABS: %s -> %d\n", axis_name(ABS_PRESSURE, EV_ABS), frame->pressure);
    }

    if (frame->complete)
        printf("    EV_SYN: --- frame end ---\n\n");
}

static int test_blocking_read(char *dev_path, size_t dev_path_size, bool auto_detect)
{
    drv_input_inst_t *inst = NULL;
    struct drv_pointer_frame frame;
    int frames = 0;

    printf("\n=== Test 1: Blocking Read ===\n");
    printf("Opening %s in blocking mode...\n", dev_path);

    if (drv_input_inst_create_path(dev_path, &inst) != 0)
    {
        perror("Failed to open device");
        return -1;
    }

    if (auto_detect)
        drv_input_inst_set_auto_reconnect(inst, DRV_INPUT_DEV_MOUSE);

    while (frames < 10)
    {
        int ret = drv_input_poll(inst, -1);
        if (ret < 0)
        {
            if (!drv_input_inst_is_connected(inst))
            {
                printf("Pointer device disconnected, waiting for reconnect...\n");
                while (reconnect_input_device(inst) != 0)
                    usleep(200000);
                continue;
            }
            perror("Poll error");
            break;
        }

        ret = drv_input_read_pointer_frame(inst, &frame);
        if (ret <= 0)
            continue;

        print_pointer_frame(&frame);
        if (frame.complete)
            frames++;
    }

    drv_input_inst_destroy(&inst);
    return 0;
}

static int test_nonblocking_read(char *dev_path, size_t dev_path_size, bool auto_detect)
{
    drv_input_inst_t *inst = NULL;
    struct drv_pointer_frame frame;
    int empty_reads = 0;
    int loops = 0;

    printf("\n=== Test 2: Non-blocking Read ===\n");

    if (drv_input_inst_create_path(dev_path, &inst) != 0)
    {
        perror("Failed to open device");
        return -1;
    }

    if (auto_detect)
        drv_input_inst_set_auto_reconnect(inst, DRV_INPUT_DEV_MOUSE);

    while (loops < 500)
    {
        int ret = drv_input_read_pointer_frame(inst, &frame);
        if (ret <= 0)
        {
            empty_reads++;
        }
        else
        {
            print_pointer_frame(&frame);
        }

        usleep(10000);
        loops++;
    }

    drv_input_inst_destroy(&inst);
    printf("Test 2 completed: %d empty reads\n", empty_reads);
    return 0;
}

static int test_poll_read(char *dev_path, size_t dev_path_size, bool auto_detect)
{
    drv_input_inst_t *inst = NULL;
    struct drv_pointer_frame frame;
    int frames = 0;

    printf("\n=== Test 3: Poll Read ===\n");

    if (drv_input_inst_create_path(dev_path, &inst) != 0)
    {
        perror("Failed to open device");
        return -1;
    }

    if (auto_detect)
        drv_input_inst_set_auto_reconnect(inst, DRV_INPUT_DEV_MOUSE);

    while (frames < 10)
    {
        int ret = drv_input_poll(inst, 1000);
        if (ret < 0)
        {
            if (!drv_input_inst_is_connected(inst))
            {
                printf("Pointer device disconnected, waiting for reconnect...\n");
                while (reconnect_input_device(inst) != 0)
                    usleep(200000);
                continue;
            }
            perror("Poll error");
            break;
        }
        if (ret == 0)
            continue;

        while (1)
        {
            ret = drv_input_read_pointer_frame(inst, &frame);
            if (ret <= 0)
                break;

            print_pointer_frame(&frame);
            if (frame.complete)
                frames++;
        }
    }

    drv_input_inst_destroy(&inst);
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
        dev_path = argv[1];
    else if (drv_input_find_first_by_type(DRV_INPUT_DEV_MOUSE,
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
    printf("USB HID Mouse/Touch Test\n");
    printf("========================================\n");
    printf("Device: %s\n", dev_path);
    if (argc <= 1)
        printf("Detected device: %s (%s)\n", dev_path, info.name);
    printf("Event size: %zu bytes\n", sizeof(struct input_event));

    if (test_blocking_read(active_path, sizeof(active_path), auto_detect) != 0)
        return 1;
    if (test_nonblocking_read(active_path, sizeof(active_path), auto_detect) != 0)
        return 1;
    if (test_poll_read(active_path, sizeof(active_path), auto_detect) != 0)
        return 1;

    return 0;
}