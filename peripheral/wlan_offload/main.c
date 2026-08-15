/*
 * Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wlan_offload_client.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>

static void print_mac(const uint8_t address[6])
{
    printf("%02x:%02x:%02x:%02x:%02x:%02x",
           address[0], address[1], address[2],
           address[3], address[4], address[5]);
}

static void print_ssid(const struct wlan_offload_ssid *ssid)
{
    size_t index;

    for (index = 0; index < ssid->length; index++)
    {
        unsigned char value = ssid->value[index];

        putchar(value >= 0x20 && value <= 0x7e ? value : '.');
    }
}

static void print_info(const struct wlan_offload_info *info)
{
    printf("address: ");
    print_mac(info->address);
    printf("\ncapabilities: 0x%08x\n", info->capabilities);
    printf("phy capabilities: 0x%08x\n", info->phy_capabilities);
    printf("ciphers: 0x%08x\n", info->cipher_mask);
    printf("interface types: 0x%08x\n", info->interface_type_mask);
    printf("bands: 0x%02x\n", info->band_mask);
    printf("maximum frame: %u\n", info->max_frame_size);
    printf("maximum scan SSIDs: %u\n", info->max_scan_ssids);
    printf("maximum scan IE: %u\n", info->max_scan_ie_length);
    printf("framework API: %u\n", info->framework_api_version);
    printf("firmware protocol: %u\n", info->firmware_protocol_version);
    printf("firmware version: 0x%08x\n", info->firmware_version);
    printf("firmware features: 0x%08x\n", info->firmware_features);
    printf("firmware generation: %u\n", info->firmware_generation);
    printf("firmware limits: vifs=%u stations=%u channel-contexts=%u\n",
           info->max_vifs, info->max_stations,
           info->max_channel_contexts);
}

static void print_event(const struct wlan_offload_event *event)
{
    if (event->overflow)
    {
        fputs("warning: control event queue overflowed\n", stderr);
    }
    switch (event->type)
    {
    case WLAN_OFFLOAD_EVENT_RADIO_ONLINE:
        puts("radio online");
        break;
    case WLAN_OFFLOAD_EVENT_RADIO_OFFLINE:
        puts("radio offline");
        break;
    case WLAN_OFFLOAD_EVENT_SCAN_RESULT:
        printf("scan: ");
        print_ssid(&event->data.network.ssid);
        printf("  ");
        print_mac(event->data.network.bssid);
        printf("  channel=%u frequency=%u MHz width=%u rssi=%d "
               "security=0x%08x%s\n",
               event->data.network.channel.primary_channel,
               event->data.network.channel.primary_frequency_mhz,
               event->data.network.channel.width,
               event->data.network.rssi,
               event->data.network.security,
               event->truncated ? " truncated" : "");
        break;
    case WLAN_OFFLOAD_EVENT_SCAN_DONE:
        printf("scan done: request=%u status=%d\n",
               event->request_id, event->status);
        break;
    case WLAN_OFFLOAD_EVENT_CONNECT_RESULT:
        printf("connect result: request=%u status=%d\n",
               event->request_id, event->status);
        break;
    case WLAN_OFFLOAD_EVENT_DISCONNECTED:
        printf("disconnected: reason=%u peer=",
               event->data.disconnected.reason);
        print_mac(event->data.disconnected.bssid);
        putchar('\n');
        break;
    case WLAN_OFFLOAD_EVENT_AUTH_RX:
        printf("authentication frame: %u bytes\n", event->data.frame.length);
        break;
    case WLAN_OFFLOAD_EVENT_ASSOC_RX:
        printf("association frame: %u bytes\n", event->data.frame.length);
        break;
    case WLAN_OFFLOAD_EVENT_MGMT_RX:
        printf("management frame: %u bytes\n", event->data.frame.length);
        break;
    case WLAN_OFFLOAD_EVENT_MGMT_TX_STATUS:
        printf("management TX: cookie=%llu acknowledged=%u\n",
               (unsigned long long)event->data.tx_status.cookie,
               event->data.tx_status.acknowledged);
        break;
    case WLAN_OFFLOAD_EVENT_EAPOL_RX:
        printf("EAPOL: %u bytes from ", event->data.eapol.length);
        print_mac(event->data.eapol.source);
        putchar('\n');
        break;
    case WLAN_OFFLOAD_EVENT_REGULATORY_CHANGED:
        puts("regulatory domain changed");
        break;
    case WLAN_OFFLOAD_EVENT_FIRMWARE_ERROR:
        printf("firmware error: reason=%u dump=%u bytes\n",
               event->data.firmware.reason,
               event->data.firmware.dump_length);
        break;
    case WLAN_OFFLOAD_EVENT_EXTERNAL_AUTH_REQUIRED:
        printf("external auth: request=%u akm=0x%08x ssid=",
               event->request_id, event->data.external_auth.akm_suite);
        print_ssid(&event->data.external_auth.ssid);
        printf(" bssid=");
        print_mac(event->data.external_auth.bssid);
        putchar('\n');
        break;
    }
}

static int enable_station(struct wlan_offload_handle *handle)
{
    int result = wlan_offload_set_interface(handle, WLAN_OFFLOAD_INTERFACE_STATION,
                                       1, NULL);

    if (result)
    {
        fprintf(stderr, "enable station: %s\n", strerror(-result));
    }
    return result;
}

static int run_scan(struct wlan_offload_handle *handle)
{
    struct wlan_offload_scan_params params;
    struct wlan_offload_event event;
    uint32_t request_id;
    int result;

    result = enable_station(handle);
    if (result)
    {
        return result;
    }
    memset(&params, 0, sizeof(params));
    result = wlan_offload_scan(handle, WLAN_OFFLOAD_INTERFACE_STATION,
                          &params, &request_id);
    if (result)
    {
        fprintf(stderr, "scan: %s\n", strerror(-result));
        return result;
    }
    for (;;)
    {
        result = wlan_offload_receive_event(handle, &event, 10000);
        if (result)
        {
            fprintf(stderr, "receive event: %s\n", strerror(-result));
            return result;
        }
        print_event(&event);
        if (event.type == WLAN_OFFLOAD_EVENT_SCAN_DONE &&
            event.request_id == request_id)
        {
            return event.status;
        }
    }
}

static int run_monitor(struct wlan_offload_handle *handle)
{
    struct wlan_offload_event event;
    int result;

    result = enable_station(handle);
    if (result)
    {
        return result;
    }
    for (;;)
    {
        result = wlan_offload_receive_event(handle, &event, -1);
        if (result)
        {
            fprintf(stderr, "receive event: %s\n", strerror(-result));
            return result;
        }
        print_event(&event);
    }
}

static int run_info(struct wlan_offload_handle *handle, int probe)
{
    struct wlan_offload_info info;
    int result;

    if (probe)
    {
        result = enable_station(handle);
        if (result)
        {
            return result;
        }
    }
    result = wlan_offload_get_info(handle, WLAN_OFFLOAD_INTERFACE_STATION, &info);
    if (!result)
    {
        print_info(&info);
    }
    return result;
}

static int run_names(struct wlan_offload_handle *handle)
{
    struct wlan_offload_names names;
    int result;

    result = wlan_offload_get_names(handle, WLAN_OFFLOAD_INTERFACE_STATION,
                                    &names);
    if (result)
    {
        return result;
    }
    printf("radio index      : %u\n", names.radio_index);
    printf("control device   : /dev/%s\n", names.control);
    printf("station device   : %s\n",
           names.station[0] ? names.station : "(none)");
    printf("ap device        : %s\n", names.ap[0] ? names.ap : "(none)");
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr, "usage: %s [device] info|names|probe|scan|monitor\n",
            program);
}

int main(int argc, char **argv)
{
    const char *device = "/dev/wlanctl0";
    const char *command;
    struct wlan_offload_handle *handle;
    int result;

    if (argc == 2)
    {
        command = argv[1];
    }
    else if (argc == 3)
    {
        device = argv[1];
        command = argv[2];
    }
    else
    {
        usage(argv[0]);
        return 2;
    }
    result = wlan_offload_open(&handle, device);
    if (result)
    {
        fprintf(stderr, "open %s: %s\n", device, strerror(-result));
        return 1;
    }
    if (!strcmp(command, "info") || !strcmp(command, "probe"))
    {
        result = run_info(handle, !strcmp(command, "probe"));
    }
    else if (!strcmp(command, "names"))
    {
        result = run_names(handle);
    }
    else if (!strcmp(command, "scan"))
    {
        result = run_scan(handle);
    }
    else if (!strcmp(command, "monitor"))
    {
        result = run_monitor(handle);
    }
    else
    {
        usage(argv[0]);
        result = -EINVAL;
    }
    if (result && (!strcmp(command, "info") || !strcmp(command, "probe") ||
                   !strcmp(command, "names")))
    {
        fprintf(stderr, "%s on %s: %s\n", command, device,
                strerror(-result));
    }
    wlan_offload_close(handle);
    return result ? 1 : 0;
}
