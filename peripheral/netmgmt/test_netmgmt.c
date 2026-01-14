#include "hal_netmgmt.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define TEST_SSID     "Xel"
#define TEST_PASSWORD "xel123456"
#define TEST_AP_SSID  "TestSoftAP"

#define TEST_LAN_STATIC_IP   0xC0A80164 // 192.168.1.100
#define TEST_LAN_STATIC_GW   0xC0A80101 // 192.168.1.1
#define TEST_LAN_STATIC_MASK 0xFFFFFF00
#define TEST_LAN_STATIC_DNS  0x08080808

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

    char dev[32] = { 0 };
    CHECK_OK(netmgmt_utils_set_defeault_dev("w0"), "Set default device");
    CHECK_OK(netmgmt_utils_get_defeault_dev(dev), "Get default device");
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

    CHECK_OK(netmgmt_wlan_ap_set_country(86), "Set country to 86");

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

int main()
{
    printf("=== [FULL NETMGMT TEST] ===\n");

    test_utils();
    test_wlan_sta();
    test_wlan_ap();
    test_lan();

    printf("\n=== [TEST COMPLETE] ===\n");
    return 0;
}
