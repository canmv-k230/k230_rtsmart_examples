/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "host/ble_gatt.h"
#include "host/ble_gap.h"
#include "host/ble_hs.h"
#include "host/ble_store.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "os/os_mbuf.h"
#include "rtsmart_nimble_hci.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "store/config/ble_store_config.h"

#include <stdio.h>
#include <string.h>

#define DEVICE_NAME "K230-NimBLE"
#define VALUE_MAX_LENGTH 20

void ble_store_config_init(void);

/* 6e400001-b5a3-f393-e0a9-e50e24dcca9e */
static const ble_uuid128_t peripheral_service_uuid =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
                     0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);

/* 6e400002-b5a3-f393-e0a9-e50e24dcca9e */
static const ble_uuid128_t peripheral_value_uuid =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
                     0x93, 0xf3, 0xa3, 0xb5, 0x02, 0x00, 0x40, 0x6e);

static uint8_t peripheral_value[VALUE_MAX_LENGTH] = "K230";
static uint16_t peripheral_value_length = 4;

static int peripheral_access(uint16_t conn_handle, uint16_t attr_handle,
                             struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    uint16_t length;
    int result;

    (void)conn_handle;
    (void)attr_handle;
    (void)arg;

    switch (ctxt->op) {
    case BLE_GATT_ACCESS_OP_READ_CHR:
        result = os_mbuf_append(ctxt->om, peripheral_value,
                                peripheral_value_length);
        return result == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;

    case BLE_GATT_ACCESS_OP_WRITE_CHR:
        if (OS_MBUF_PKTLEN(ctxt->om) > sizeof(peripheral_value)) {
            return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
        }
        result = ble_hs_mbuf_to_flat(ctxt->om, peripheral_value,
                                     sizeof(peripheral_value), &length);
        if (result != 0) {
            return BLE_ATT_ERR_UNLIKELY;
        }
        peripheral_value_length = length;
        printf("GATT value updated (%u bytes)\n", length);
        return 0;

    default:
        return BLE_ATT_ERR_UNLIKELY;
    }
}

static const struct ble_gatt_svc_def peripheral_services[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &peripheral_service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = &peripheral_value_uuid.u,
                .access_cb = peripheral_access,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_WRITE |
                         BLE_GATT_CHR_F_WRITE_NO_RSP,
            },
            { 0 },
        },
    },
    { 0 },
};

static void peripheral_advertise(void);

static int peripheral_gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;

    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            printf("BLE connected, handle=%u\n", event->connect.conn_handle);
        } else {
            printf("BLE connection failed, status=%d\n", event->connect.status);
            peripheral_advertise();
        }
        break;

    case BLE_GAP_EVENT_DISCONNECT:
        printf("BLE disconnected, reason=%d\n", event->disconnect.reason);
        peripheral_advertise();
        break;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        peripheral_advertise();
        break;

    default:
        break;
    }
    return 0;
}

static void peripheral_advertise(void)
{
    struct ble_gap_adv_params parameters;
    struct ble_hs_adv_fields fields;
    uint8_t own_address_type;
    int result;

    result = ble_hs_id_infer_auto(0, &own_address_type);
    if (result != 0) {
        printf("Cannot select BLE address type: %d\n", result);
        return;
    }

    memset(&fields, 0, sizeof(fields));
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.uuids128 = &peripheral_service_uuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;
    result = ble_gap_adv_set_fields(&fields);
    if (result != 0) {
        printf("Cannot set BLE advertising data: %d\n", result);
        return;
    }

    memset(&fields, 0, sizeof(fields));
    fields.name = (const uint8_t *)DEVICE_NAME;
    fields.name_len = strlen(DEVICE_NAME);
    fields.name_is_complete = 1;
    result = ble_gap_adv_rsp_set_fields(&fields);
    if (result != 0) {
        printf("Cannot set BLE scan response: %d\n", result);
        return;
    }

    memset(&parameters, 0, sizeof(parameters));
    parameters.conn_mode = BLE_GAP_CONN_MODE_UND;
    parameters.disc_mode = BLE_GAP_DISC_MODE_GEN;
    result = ble_gap_adv_start(own_address_type, NULL, BLE_HS_FOREVER,
                               &parameters, peripheral_gap_event, NULL);
    if (result != 0) {
        printf("Cannot start BLE advertising: %d\n", result);
    } else {
        printf("Advertising as %s\n", DEVICE_NAME);
    }
}

static void peripheral_on_reset(int reason)
{
    printf("NimBLE reset, reason=%d\n", reason);
}

static void peripheral_on_sync(void)
{
    int result = ble_hs_util_ensure_addr(0);

    if (result != 0) {
        printf("Cannot initialize BLE address: %d\n", result);
        return;
    }
    peripheral_advertise();
}

int main(int argc, char **argv)
{
    const char *user_hci_path = argc > 1 ? argv[1] : NULL;
    const char *selected_hci_path;
    int result;

    result = rtsmart_nimble_hci_set_device(user_hci_path);
    if (result != 0) {
        fprintf(stderr, "Invalid HCI device %s: %d\n",
                user_hci_path ? user_hci_path : "(auto)",
                result);
        return 1;
    }

    nimble_port_init();
    ble_store_config_init();
    ble_svc_gap_init();
    ble_svc_gatt_init();

    ble_hs_cfg.reset_cb = peripheral_on_reset;
    ble_hs_cfg.sync_cb = peripheral_on_sync;
    ble_hs_cfg.store_status_cb = ble_store_util_status_rr;

    result = ble_svc_gap_device_name_set(DEVICE_NAME);
    if (result == 0) {
        result = ble_gatts_count_cfg(peripheral_services);
    }
    if (result == 0) {
        result = ble_gatts_add_svcs(peripheral_services);
    }
    if (result != 0) {
        fprintf(stderr, "Cannot configure GATT services: %d\n", result);
        return 1;
    }

    /* NULL is expected when transport initialization found no usable HCI node. */
    selected_hci_path = rtsmart_nimble_hci_get_device();
    printf("NimBLE peripheral using %s\n",
           selected_hci_path ? selected_hci_path : "(no HCI controller)");
    nimble_port_run();
    return 0;
}
