#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <signal.h>
#include <unistd.h>

#include "drv_adc.h"

volatile sig_atomic_t stop_flag = 0;

// Signal handler for Ctrl+C
void handle_sigint(int sig)
{
    (void)sig;
    stop_flag = 1;
    printf("\nReceived Ctrl+C, shutting down...\n");
}

int main(int argc, char** argv)
{
    int      channel = 0; // Default channel
    uint32_t ref_uv  = DRV_ADC_DEFAULT_REF_UV;

    // Parse command line arguments
    if (argc > 1) {
        channel = atoi(argv[1]);
        if (channel < 0 || channel >= DRV_ADC_MAX_CHANNEL) {
            fprintf(stderr, "Error: Invalid channel (0-%d)\n", DRV_ADC_MAX_CHANNEL - 1);
            return 1;
        }
    }
    if (argc > 2) {
        ref_uv = atoi(argv[2]);
        if (ref_uv != 1800000 && ref_uv != 3600000) { // Reasonable range check
            fprintf(stderr, "Error: Reference voltage should be 1.8V or 3.6V (1800000, 3600000 uV)\n");
            return 1;
        }
    }

    // Register signal handler
    signal(SIGINT, handle_sigint);

    printf("ADC Test Application\n");
    printf("Reading channel %d with reference %.2fV\n", channel, (float)ref_uv / 1000000.0f);
    printf("Press Ctrl+C to stop...\n\n");

    // Initialize ADC
    if (drv_adc_init() != 0) {
        fprintf(stderr, "ADC initialization failed\n");
        return 1;
    }

    // Main measurement loop
    while (!stop_flag) {
        uint32_t raw     = drv_adc_read(channel);
        uint32_t uv      = drv_adc_read_uv(channel, ref_uv);
        float    voltage = (float)uv / 1000000.0f;

        printf("Channel %d: Raw=0x%03X (%4u), Voltage=%.4fV (%u uV)\n", channel, raw, raw, voltage, uv);

        sleep(1); // 1 second delay between readings
    }

    // Cleanup
    drv_adc_deinit();
    printf("ADC test completed\n");

    return 0;
}
