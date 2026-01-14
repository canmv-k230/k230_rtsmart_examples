#include "canmv_misc.h"
#include "drv_touch.h"
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Global flag for graceful shutdown */
static volatile bool g_running = true;

/* Signal handler for Ctrl-C */
void signal_handler(int sig)
{
    if (sig == SIGINT) {
        printf("\nReceived SIGINT (Ctrl-C), shutting down gracefully...\n");
        g_running = false;
    }
}

/* Setup signal handling */
void setup_signal_handler(void)
{
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
}

/* Print touch event as string */
const char* touch_event_to_string(uint8_t event)
{
    switch (event) {
    case DRV_TOUCH_EVENT_NONE:
        return "NONE";
    case DRV_TOUCH_EVENT_UP:
        return "UP";
    case DRV_TOUCH_EVENT_DOWN:
        return "DOWN";
    case DRV_TOUCH_EVENT_MOVE:
        return "MOVE";
    default:
        return "UNKNOWN";
    }
}

/* Dump touch data in readable format */
void dump_touch_data(const struct drv_touch_data* touch_data, int point_count, int read_cycle)
{
    printf("\n=== Touch Data Read Cycle #%d ===\n", read_cycle);
    printf("Points detected: %d\n", point_count);
    printf("----------------------------------------\n");

    if (point_count <= 0) {
        printf("No touch points detected\n");
        return;
    }

    for (int i = 0; i < point_count; i++) {
        const struct drv_touch_data* point = &touch_data[i];
        printf("Point %d:\n", i);
        printf("  Event:      %s (%d)\n", touch_event_to_string(point->event), point->event);
        printf("  Track ID:   %d\n", point->track_id);
        printf("  Position:   X=%d, Y=%d\n", point->x_coordinate, point->y_coordinate);
        printf("  Width:      %d\n", point->width);
        printf("  Timestamp:  %u ms\n", point->timestamp);
        printf("----------------------------------------\n");
    }
}

/* Dump touch configuration in readable format */
void dump_touch_config(const struct drv_touch_config_t* config)
{
    if (config == NULL) {
        printf("Touch Configuration: NULL\n");
        return;
    }

    printf("=== Touch Device Configuration ===\n");
    printf("Device Index:      %d\n", config->touch_dev_index);
    printf("Display Range:     %d x %d\n", config->range_x, config->range_y);
    printf("Interrupt Pin:     %d\n", config->pin_intr);
    printf("Interrupt Value:   %d\n", config->intr_value);
    printf("Reset Pin:         %d\n", config->pin_reset);
    printf("Reset Value:       %d\n", config->reset_value);
    printf("I2C Bus Index:     %d\n", config->i2c_bus_index);
    printf("I2C Bus Speed:     %d Hz\n", config->i2c_bus_speed);

    /* Additional analysis based on values */
    printf("\nConfiguration Analysis:\n");

    /* Display size analysis */
    if (config->range_x > 0 && config->range_y > 0) {
        float       aspect_ratio = (float)config->range_x / config->range_y;
        const char* aspect_desc;

        if (aspect_ratio > 1.7 && aspect_ratio < 1.8) {
            aspect_desc = "16:9 display";
        } else if (aspect_ratio > 1.3 && aspect_ratio < 1.4) {
            aspect_desc = "4:3 display";
        } else if (aspect_ratio > 1.5 && aspect_ratio < 1.6) {
            aspect_desc = "3:2 display";
        } else if (fabs(aspect_ratio - 1.0) < 0.1) {
            aspect_desc = "1:1 square display";
        } else {
            aspect_desc = "custom aspect ratio";
        }
        printf("  - Display: %dx%d (%.2f:1, %s)\n", config->range_x, config->range_y, aspect_ratio, aspect_desc);
    }

    /* I2C speed analysis */
    if (config->i2c_bus_speed > 0) {
        if (config->i2c_bus_speed <= 100000) {
            printf("  - I2C: Standard mode (100 kHz)\n");
        } else if (config->i2c_bus_speed <= 400000) {
            printf("  - I2C: Fast mode (400 kHz)\n");
        } else if (config->i2c_bus_speed <= 1000000) {
            printf("  - I2C: Fast mode plus (1 MHz)\n");
        } else {
            printf("  - I2C: High speed mode (%d kHz)\n", config->i2c_bus_speed / 1000);
        }
    }

    /* Pin configuration analysis */
    if (config->pin_intr >= 0 && config->pin_reset >= 0) {
        printf("  - Hardware: Both interrupt and reset pins configured\n");
    } else if (config->pin_intr >= 0) {
        printf("  - Hardware: Interrupt pin only (no reset)\n");
    } else if (config->pin_reset >= 0) {
        printf("  - Hardware: Reset pin only (no interrupt)\n");
    } else {
        printf("  - Hardware: No dedicated pins (polling mode)\n");
    }

    printf("==================================\n\n");
}

/* Print device information */
void print_device_info(drv_touch_inst_t* inst)
{
    struct drv_touch_info info;

    if (drv_touch_get_info(inst, &info) == 0) {
        printf("=== Touch Device Information ===\n");
        printf("Type:      %d\n", info.type);
        printf("Vendor:    %d\n", info.vendor);
        printf("Max Points: %d\n", info.point_num);
        printf("X Range:   0-%u\n", info.range_x);
        printf("Y Range:   0-%u\n", info.range_y);
        printf("===============================\n\n");
    } else {
        printf("Failed to get device information\n");
    }
}

void print_device_config(drv_touch_inst_t* inst)
{
    struct drv_touch_config_t cfg;

    if (drv_touch_get_config(inst, &cfg) == 0) {
        dump_touch_config(&cfg);
    } else {
        printf("Failed to get device config\n");
    }
}

/* Get rotation information */
void print_rotation_info(drv_touch_inst_t* inst)
{
    int rotate;

    if (drv_touch_get_default_rotate(inst, &rotate) == 0) {
        const char* rotation_str;
        switch (rotate) {
        case DRV_TOUCH_ROTATE_DEGREE_0:
            rotation_str = "0 degrees";
            break;
        case DRV_TOUCH_ROTATE_DEGREE_90:
            rotation_str = "90 degrees";
            break;
        case DRV_TOUCH_ROTATE_DEGREE_180:
            rotation_str = "180 degrees";
            break;
        case DRV_TOUCH_ROTATE_DEGREE_270:
            rotation_str = "270 degrees";
            break;
        case DRV_TOUCH_ROTATE_SWAP_XY:
            rotation_str = "Swap XY";
            break;
        default:
            rotation_str = "Unknown";
            break;
        }
        printf("Default Rotation: %s (%d)\n\n", rotation_str, rotate);
    } else {
        printf("Failed to get rotation information\n\n");
    }
}

/* Print usage information */
void print_usage(const char* program_name)
{
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Test touch device functionality\n\n");
    printf("Options:\n");
    printf("  -d, --device ID        Use existing touch device ID (default: 0)\n");
    printf("  -c, --create           Create new touch device with configuration\n");
    printf("  --index INDEX          Touch device index for new device (required with --create)\n");
    printf("  --i2c-bus BUS          I2C bus index (required with --create)\n");
    printf("  --i2c-speed SPEED      I2C bus speed (required with --create)\n");

    printf("  --range-x WIDTH        X-axis range\n");
    printf("  --range-y HEIGHT       Y-axis range\n");
    printf("  --int-pin PIN          Interrupt pin\n");
    printf("  --int-value VALUE      Interrupt pin value\n");
    printf("  --reset-pin PIN        Reset pin\n");
    printf("  --reset-value VALUE    Reset pin value\n");

    printf("  -h, --help             Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s -d 0                    # Test default touch device 0\n", program_name);
    printf("  %s -c --index 1 --range-x 480 --range-y 800 --int-pin 23 --int-value 1 --reset-pin 22 --reset-value 0 "
           "--i2c-bus 3 --i2c-speed 400000 # Create and test new touch device\n",
           program_name);
}

/* Create new touch device with configuration */
int create_touch_device(struct drv_touch_config_t* config)
{
    printf("Creating new touch device with configuration:\n");
    dump_touch_config(config);

    int ret = canmv_misc_create_touch_device(config);
    if (ret != 0) {
        fprintf(stderr, "Failed to create touch device (error: %d)\n", ret);
        return -1;
    }

    printf("Successfully created touch device %d\n", config->touch_dev_index);
    return 0;
}

/* Cleanup created touch device */
void cleanup_touch_device(int device_index)
{
    printf("Unregistering touch device %d...\n", device_index);
    int ret = canmv_misc_delete_touch_device(device_index);
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to unregister touch device %d (error: %d)\n", device_index, ret);
    } else {
        printf("Successfully unregistered touch device %d\n", device_index);
    }
}

int main(int argc, char* argv[])
{
    drv_touch_inst_t*     touch_inst = NULL;
    struct drv_touch_data touch_data[5]; // Buffer for max 5 points
    int                   read_cycle = 0;
    int                   ret;

    /* Command line options */
    int                       device_id         = 0;
    int                       create_new        = 0;
    struct drv_touch_config_t new_device_config = {
        .touch_dev_index = 1,
        .range_x         = 0, // use system default
        .range_y         = 0, // use system default
        .pin_intr        = -1,
        .intr_value      = 0,
        .pin_reset       = -1,
        .reset_value     = 0,
        .i2c_bus_index   = 3,
        .i2c_bus_speed   = 400000,
    };

    /* Initialize config with safe defaults */
    new_device_config.pin_intr  = -1;
    new_device_config.pin_reset = -1;

    /* Command line option definitions */
    static struct option long_options[] = { { "device", required_argument, 0, 'd' },
                                            { "create", no_argument, 0, 'c' },
                                            { "index", required_argument, 0, 1 },
                                            { "range-x", required_argument, 0, 2 },
                                            { "range-y", required_argument, 0, 3 },
                                            { "int-pin", required_argument, 0, 5 },
                                            { "int-value", required_argument, 0, 6 },
                                            { "reset-pin", required_argument, 0, 7 },
                                            { "reset-value", required_argument, 0, 8 },
                                            { "i2c-bus", required_argument, 0, 9 },
                                            { "i2c-speed", required_argument, 0, 10 },
                                            { "help", no_argument, 0, 'h' },
                                            { 0, 0, 0, 0 } };

    /* Parse command line arguments */
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "d:cDh", long_options, &option_index)) != -1) {
        switch (opt) {
        case 'd':
            device_id = atoi(optarg);
            break;
        case 'c':
            create_new = 1;
            break;
        case 1:
            new_device_config.touch_dev_index = atoi(optarg);
            break;
        case 2:
            new_device_config.range_x = atoi(optarg);
            break;
        case 3:
            new_device_config.range_y = atoi(optarg);
            break;
        case 5:
            new_device_config.pin_intr = atoi(optarg);
            break;
        case 6:
            new_device_config.intr_value = atoi(optarg);
            break;
        case 7:
            new_device_config.pin_reset = atoi(optarg);
            break;
        case 8:
            new_device_config.reset_value = atoi(optarg);
            break;
        case 9:
            new_device_config.i2c_bus_index = atoi(optarg);
            break;
        case 10:
            new_device_config.i2c_bus_speed = atoi(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        default:
            fprintf(stderr, "Unknown option. Use -h for help.\n");
            return EXIT_FAILURE;
        }
    }
    printf("Touch Device Test Application\n");
    printf("=============================\n");

    /* Setup signal handler for Ctrl-C */
    setup_signal_handler();

    if (create_new) {
        /* Create new touch device */
        printf("Mode: Creating new touch device\n");
        ret = create_touch_device(&new_device_config);
        if (ret != 0) {
            return EXIT_FAILURE;
        }
        device_id = new_device_config.touch_dev_index;
        printf("\n");
    } else {
        /* Use existing device */
        printf("Mode: Testing existing touch device %d\n", device_id);
    }

    /* Create touch instance */
    printf("Opening touch device %d...\n", device_id);
    ret = drv_touch_inst_create(device_id, &touch_inst);
    if (ret != 0 || touch_inst == NULL) {
        fprintf(stderr, "Failed to create touch instance for device %d (error: %d)\n", device_id, ret);
        if (create_new) {
            cleanup_touch_device(device_id);
        }
        return EXIT_FAILURE;
    }
    printf("Successfully opened touch device %d\n\n", device_id);

    /* Display device information */
    print_device_info(touch_inst);
    print_device_config(touch_inst);
    print_rotation_info(touch_inst);

    printf("Starting touch data reading loop...\n");
    printf("Press Ctrl-C to stop the application\n\n");

    /* Main reading loop */
    while (g_running) {
        /* Read touch data with max 5 points */
        int point_count = drv_touch_read(touch_inst, touch_data, 5);

        if (point_count > 0) {
            /* Successfully read touch data */
            dump_touch_data(touch_data, point_count, ++read_cycle);
        } else if (point_count == 0) {
            /* No data available (non-blocking read) */
            printf(".");
            fflush(stdout);
        } else {
            /* Error occurred */
            if (point_count == -1) {
                fprintf(stderr, "Error: Invalid parameters in drv_touch_read\n");
            } else if (point_count == -2) {
                /* This is normal for non-blocking read when no data is available */
                printf(".");
                fflush(stdout);
            } else {
                fprintf(stderr, "Error: Unknown error in drv_touch_read (%d)\n", point_count);
            }
        }

        /* Small delay to prevent excessive CPU usage */
        usleep(10000); // 10ms delay

        if (100 <= read_cycle) {
            g_running = 0;
            printf("reach 100 times test, exit\n");
        }
    }

    /* Cleanup */
    printf("\nCleaning up resources...\n");
    if (touch_inst != NULL) {
        drv_touch_inst_destroy(&touch_inst);
        printf("Touch instance destroyed\n");
    }

    if (create_new) {
        cleanup_touch_device(device_id);
    }

    printf("Application terminated gracefully\n");
    return EXIT_SUCCESS;
}
