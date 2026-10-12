#include "hal_netmgmt.h"
#include <arpa/inet.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define TEST_SSID     "Xel"
#define TEST_PASSWORD "xel123456"
#define TEST_AP_SSID  "TestSoftAP"

#define TEST_LAN_STATIC_IP   0xC0A80164 // 192.168.1.100
#define TEST_LAN_STATIC_GW   0xC0A80101 // 192.168.1.1
#define TEST_LAN_STATIC_MASK 0xFFFFFF00
#define TEST_LAN_STATIC_DNS  0x08080808

#define DNS_PORT             53
#define DNS_RETRY_COUNT      5
#define DNS_REPLY_TIMEOUT_MS 2000
#define SOAK_DELAY_US        500000U
#define SOAK_REPORT_SECONDS  60U

static volatile sig_atomic_t soak_stop;

#define PASS(msg) printf(" [PASS] %s\n", msg)
#define FAIL(msg) printf(" [FAIL] %s\n", msg)
#define STEP(msg) printf("\n---- %s ----\n", msg)
#define DELAY_SEC(s)                                                                                                           \
    do {                                                                                                                       \
        sleep(s);                                                                                                              \
    } while (0)

#define CHECK_OK(expr, msg)                                                                                                    \
    do {                                                                                                                       \
        if ((expr) == 0)                                                                                                       \
            PASS(msg);                                                                                                         \
        else {                                                                                                                 \
            FAIL(msg);                                                                                                         \
            return;                                                                                                            \
        }                                                                                                                      \
    } while (0)

static void test_utils()
{
    STEP("UTILS: Default Device + Ifconfig + Probe");

    char preferred_dev[32] = "wlan0";
    char dev[32] = { 0 };
    CHECK_OK(netmgmt_utils_set_default_dev(preferred_dev), "Set default device");
    CHECK_OK(netmgmt_utils_get_default_dev(dev), "Get default device");
    printf("  Default device: %s\n", dev);

    int probe = 0;
    CHECK_OK(netmgmt_utils_probe_device(RT_NET_DEV_WLAN_STA, &probe), "Probe WLAN STA");

    struct ifconfig_t cfg = { 0 };
    CHECK_OK(netmgmt_utils_get_ifconfig(RT_NET_DEV_WLAN_STA, &cfg), "Get ifconfig");
    printf("  IP: 0x%X GW: 0x%X MASK: 0x%X DNS: 0x%X\n", cfg.ip.addr, cfg.gw.addr, cfg.netmask.addr, cfg.dns.addr);

    int  dev_num                       = 0;
    char dev_list[NET_DEV_MAX_CNT][32] = { 0 };
    CHECK_OK(netmgmt_utils_get_dev_list(&dev_num, dev_list), "Get device list");
    printf("  Device count: %d\n", dev_num);
    for (int i = 0; i < dev_num; ++i) {
        printf("   - %s\n", dev_list[i]);
    }
}

static void test_wlan_sta()
{
    STEP("WLAN STA: Auto Reconnect + Connect");

    CHECK_OK(netmgmt_wlan_sta_set_auto_reconnect(1), "Enable auto reconnect");

    int auto_reconnect = 0;
    CHECK_OK(netmgmt_wlan_sta_get_auto_reconnect(&auto_reconnect), "Read auto reconnect");
    printf("  Value: %d\n", auto_reconnect);

    CHECK_OK(netmgmt_wlan_sta_connect_with_ssid(TEST_SSID, TEST_PASSWORD), "Connect to SSID");
    DELAY_SEC(3);

    int connected = 0;
    CHECK_OK(netmgmt_wlan_sta_isconnected(&connected), "Check connection");
    if (!connected) {
        FAIL("STA not connected, aborting STA test");
        return;
    }

    STEP("WLAN STA: Info + MAC + RSSI");

    struct rt_wlan_info_t info = { 0 };
    CHECK_OK(netmgmt_wlan_sta_get_ap_info(&info), "Get AP info");
    printf("  SSID: %s\n", info.ssid.val);

    int rssi = 0;
    CHECK_OK(netmgmt_wlan_sta_get_rssi(&rssi), "Get RSSI");
    printf("  RSSI: %d\n", rssi);

    uint8_t mac[RT_WLAN_BSSID_MAX_LENGTH];
    CHECK_OK(netmgmt_wlan_sta_get_mac(mac), "Get MAC");
    printf("  MAC: %02X:%02X:%02X:%02X:%02X:%02X\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    CHECK_OK(netmgmt_wlan_sta_set_mac(mac), "Set MAC (same)");

    STEP("WLAN STA: Scan");

    struct rt_wlan_info_t scan_results[RT_WLAN_STA_SCAN_MAX_AP] = { 0 };
    int                   ap_num                                = 0;
    CHECK_OK(netmgmt_wlan_sta_scan(&ap_num, scan_results), "Scan for APs");
    for (int i = 0; i < ap_num; ++i) {
        printf("   - %s RSSI: %d\n", scan_results[i].ssid.val, scan_results[i].rssi);
    }

    STEP("WLAN STA: Scan with SSID + reconnect");

    struct rt_wlan_info_t specific_ap = { 0 };
    CHECK_OK(netmgmt_wlan_sta_scan_with_ssid(TEST_SSID, &specific_ap), "Scan with SSID");
    CHECK_OK(netmgmt_wlan_sta_disconnect_ap(), "Disconnect");
    DELAY_SEC(1);
    CHECK_OK(netmgmt_wlan_sta_connect_with_scan_info(&specific_ap, TEST_PASSWORD), "Reconnect with scan info");
    DELAY_SEC(2);

    STEP("WLAN STA: Final Disconnect");
    CHECK_OK(netmgmt_wlan_sta_disconnect_ap(), "Final disconnect");

    netmgmt_wlan_sta_set_auto_reconnect(0);
}

static void test_wlan_ap()
{
    STEP("WLAN AP: Start + Status");

    CHECK_OK(netmgmt_wlan_ap_start_with_ssid(TEST_AP_SSID, TEST_PASSWORD), "Start SoftAP");
    DELAY_SEC(2);

    int active = 0;
    CHECK_OK(netmgmt_wlan_ap_isactived(&active), "Check active");
    if (!active) {
        FAIL("AP not active, aborting");
        return;
    }

    STEP("WLAN AP: Info + Country");

    struct rt_wlan_info_t info = { 0 };
    CHECK_OK(netmgmt_wlan_ap_get_info(&info), "Get AP info");
    printf("  SSID: %s Channel: %d\n", info.ssid.val, info.channel);

    int country = 0;
    CHECK_OK(netmgmt_wlan_ap_get_country(&country), "Get country");
    printf("  Country: %d\n", country);

    int original_country = country;
    CHECK_OK(netmgmt_wlan_ap_set_country(original_country), "Set current country");
    country = -1;
    CHECK_OK(netmgmt_wlan_ap_get_country(&country), "Read country after setting");
    if (country != original_country) {
        FAIL("Country readback mismatch");
        return;
    }

    STEP("WLAN AP: STA List + Disconnect");

    struct rt_wlan_info_t sta_infos[RT_WLAN_STA_SCAN_MAX_AP] = { 0 };
    int                   sta_num                            = 0;
    netmgmt_wlan_ap_get_sta_info(&sta_num, sta_infos); // No fail-check to allow empty case
    printf("  Connected STAs: %d\n", sta_num);
    if (sta_num > 0) {
        CHECK_OK(netmgmt_wlan_ap_disconnect_sta(sta_infos[0].bssid), "Deauth STA[0]");
    }

    STEP("WLAN AP: Stop");
    CHECK_OK(netmgmt_wlan_ap_stop(), "Stop AP");
}

static void test_lan()
{
    STEP("LAN: Status + MAC");

    int conn = -1, link = -1;
    enum rt_netif_t lan_itf = RT_NET_DEV_USB_RTL8152; // Specify the LAN interface

    if (netmgmt_lan_get_isconnected(lan_itf, &conn) == 0) {
        if (conn)
            PASS("Cable is connected");
        else
            FAIL("Cable is not connected");
    } else {
        FAIL("Failed to get cable connection status");
    }

    if (netmgmt_lan_get_link_status(lan_itf, &link) == 0) {
        if (link)
            PASS("Link is up");
        else
            FAIL("Link is down");
    } else {
        FAIL("Failed to get link status");
    }

    printf("  [LAN Status] isconnected=%d, link=%d\n", conn, link);

    uint8_t mac[RT_WLAN_BSSID_MAX_LENGTH] = { 0 };
    CHECK_OK(netmgmt_lan_get_mac(lan_itf, mac), "Get LAN MAC");
    printf("  MAC: %02X:%02X:%02X:%02X:%02X:%02X\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    CHECK_OK(netmgmt_lan_set_mac(lan_itf, mac), "Set MAC");

    STEP("LAN: Static IP config");

    struct ifconfig_t cfg = {
        .ip.addr      = TEST_LAN_STATIC_IP,
        .gw.addr      = TEST_LAN_STATIC_GW,
        .netmask.addr = TEST_LAN_STATIC_MASK,
        .dns.addr     = TEST_LAN_STATIC_DNS,
    };
    // The original test used RT_NET_DEV_USB, which is now part of the LAN API.
    // The new API uses specific interfaces. We'll use RT_NET_DEV_USB_RTL8152 for this test.
    CHECK_OK(netmgmt_utils_set_ifconfig_static(lan_itf, &cfg), "Set static config");
    DELAY_SEC(1);

    CHECK_OK(netmgmt_utils_get_ifconfig(lan_itf, &cfg), "Verify static config");
    printf("  Static IP: 0x%X\n", cfg.ip.addr);

    STEP("LAN: DHCP config");
    CHECK_OK(netmgmt_utils_set_ifconfig_dhcp(lan_itf), "Enable DHCP");
    DELAY_SEC(2);
    CHECK_OK(netmgmt_utils_get_ifconfig(lan_itf, &cfg), "Verify DHCP IP");
    printf("  DHCP IP: 0x%X\n", cfg.ip.addr);
}

static void soak_signal_handler(int sig)
{
    (void)sig;
    soak_stop = 1;
}

static int monotonic_seconds(uint64_t *seconds)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    *seconds = (uint64_t)now.tv_sec;
    return 0;
}

static size_t build_dns_query(uint8_t *packet, size_t capacity, uint16_t id)
{
    static const uint8_t question[] = {
        4, 'p', 'o', 'o', 'l',
        3, 'n', 't', 'p',
        3, 'o', 'r', 'g',
        0,
        0, 1,
        0, 1,
    };

    if (capacity < 12U + sizeof(question)) {
        return 0;
    }
    memset(packet, 0, capacity);
    packet[0] = (uint8_t)(id >> 8);
    packet[1] = (uint8_t)id;
    packet[2] = 0x01;
    packet[5] = 0x01;
    memcpy(packet + 12, question, sizeof(question));
    return 12U + sizeof(question);
}

static int dns_query(int fd, uint16_t id, uint64_t *sent_bytes,
                     uint64_t *received_bytes, unsigned int *retries)
{
    uint8_t request[64];
    uint8_t response[512];
    size_t request_size = build_dns_query(request, sizeof(request), id);
    unsigned int attempt;

    for (attempt = 0; attempt < DNS_RETRY_COUNT && !soak_stop; attempt++) {
        struct pollfd pollfd = {
            .fd = fd,
            .events = POLLIN,
        };
        ssize_t sent;
        ssize_t received;
        int ready;

        sent = send(fd, request, request_size, 0);
        if (sent != (ssize_t)request_size) {
            if (sent < 0 && errno == EINTR) {
                attempt--;
                continue;
            }
            continue;
        }
        *sent_bytes += (uint64_t)sent;

        ready = poll(&pollfd, 1, DNS_REPLY_TIMEOUT_MS);
        if (ready <= 0 || !(pollfd.revents & POLLIN)) {
            if (ready < 0 && errno == EINTR) {
                attempt--;
                continue;
            }
            continue;
        }

        received = recv(fd, response, sizeof(response), 0);
        if (received < 12) {
            continue;
        }
        *received_bytes += (uint64_t)received;
        if (response[0] != (uint8_t)(id >> 8) ||
            response[1] != (uint8_t)id ||
            !(response[2] & 0x80) || (response[3] & 0x0f) != 0) {
            continue;
        }
        *retries += attempt;
        return 0;
    }

    return -1;
}

static int run_usb_soak(const char *dns_address, unsigned int seconds)
{
    struct sockaddr_in server;
    enum rt_netif_t lan_itf = RT_NET_DEV_USB_RTL8152;
    uint64_t start_seconds;
    uint64_t now_seconds;
    uint64_t deadline_seconds;
    uint64_t next_report_seconds;
    uint64_t sent_bytes = 0;
    uint64_t received_bytes = 0;
    unsigned int queries = 0;
    unsigned int retries = 0;
    int connected = 0;
    int link = 0;
    int fd = -1;

    if (netmgmt_lan_get_isconnected(lan_itf, &connected) != 0 || !connected ||
        netmgmt_lan_get_link_status(lan_itf, &link) != 0 || !link) {
        fprintf(stderr, "P10_USB_SOAK USB Ethernet link is not ready\n");
        return 1;
    }

    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(DNS_PORT);
    if (inet_pton(AF_INET, dns_address, &server.sin_addr) != 1) {
        fprintf(stderr, "P10_USB_SOAK invalid DNS address: %s\n", dns_address);
        return 1;
    }

    fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd < 0 || connect(fd, (struct sockaddr *)&server, sizeof(server)) != 0) {
        fprintf(stderr, "P10_USB_SOAK socket setup failed: %s\n", strerror(errno));
        if (fd >= 0) {
            close(fd);
        }
        return 1;
    }

    if (monotonic_seconds(&start_seconds) != 0) {
        fprintf(stderr, "P10_USB_SOAK clock_gettime failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    deadline_seconds = start_seconds + seconds;
    next_report_seconds = start_seconds + SOAK_REPORT_SECONDS;
    soak_stop = 0;
    signal(SIGINT, soak_signal_handler);
    signal(SIGTERM, soak_signal_handler);

    printf("P10_USB_SOAK start interface=eth0 dns=%s seconds=%u\n",
           dns_address, seconds);

    while (!soak_stop) {
        if (monotonic_seconds(&now_seconds) != 0) {
            fprintf(stderr, "P10_USB_SOAK clock_gettime failed: %s\n", strerror(errno));
            close(fd);
            return 1;
        }
        if (now_seconds >= deadline_seconds) {
            break;
        }
        if (dns_query(fd, (uint16_t)(queries + 1U), &sent_bytes,
                      &received_bytes, &retries) != 0) {
            fprintf(stderr, "P10_USB_SOAK DNS traffic failed at query %u\n",
                    queries + 1U);
            close(fd);
            return 1;
        }
        queries++;

        if (now_seconds >= next_report_seconds) {
            if (netmgmt_lan_get_isconnected(lan_itf, &connected) != 0 || !connected ||
                netmgmt_lan_get_link_status(lan_itf, &link) != 0 || !link) {
                fprintf(stderr, "P10_USB_SOAK USB Ethernet link dropped after %u queries\n",
                        queries);
                close(fd);
                return 1;
            }
            printf("P10_USB_SOAK progress queries=%u retries=%u tx=%llu rx=%llu\n",
                   queries, retries, (unsigned long long)sent_bytes,
                   (unsigned long long)received_bytes);
            next_report_seconds += SOAK_REPORT_SECONDS;
        }
        usleep(SOAK_DELAY_US);
    }

    close(fd);
    if (soak_stop) {
        fprintf(stderr, "P10_USB_SOAK interrupted after %u queries\n", queries);
        return 1;
    }

    printf("P10_USB_SOAK PASS seconds=%u queries=%u retries=%u tx=%llu rx=%llu\n",
           seconds, queries, retries, (unsigned long long)sent_bytes,
           (unsigned long long)received_bytes);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--usb-soak") == 0) {
        char *end = NULL;
        unsigned long seconds;

        if (argc != 4) {
            fprintf(stderr, "usage: %s --usb-soak <dns-ip> <seconds>\n", argv[0]);
            return 1;
        }
        seconds = strtoul(argv[3], &end, 10);
        if (end == argv[3] || *end != '\0' || seconds == 0 || seconds > UINT32_MAX) {
            fprintf(stderr, "invalid duration: %s\n", argv[3]);
            return 1;
        }
        return run_usb_soak(argv[2], (unsigned int)seconds);
    }

    printf("=== [FULL NETMGMT TEST] ===\n");

    test_utils();
    test_wlan_sta();
    test_wlan_ap();
    test_lan();

    printf("\n=== [TEST COMPLETE] ===\n");
    return 0;
}
