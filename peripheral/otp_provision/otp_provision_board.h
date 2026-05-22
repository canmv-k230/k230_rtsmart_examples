#ifndef OTP_PROVISION_BOARD_H
#define OTP_PROVISION_BOARD_H

/*
 * Board-local status LED settings for otp_provision.
 *
 * Adjust these values per board.
 *
 * OTP_STATUS_LED_PIN:
 *   GPIO pin number used to report the final provisioning result.
 *   Set to -1 to disable LED indication.
 *
 * OTP_STATUS_LED_ACTIVE_LOW:
 *   1 when driving the LED low turns it on.
 *   0 when driving the LED high turns it on.
 */

#if defined(CONFIG_BOARD_K230_CANMV_01STUDIO)

#define OTP_STATUS_LED_PIN        52
#define OTP_STATUS_LED_ACTIVE_LOW 1

#elif defined(CONFIG_BOARD_K230_CANMV_LCKFB)

#define OTP_STATUS_LED_PIN        63 // BLUE
#define OTP_STATUS_LED_ACTIVE_LOW 1

#elif defined(CONFIG_BOARD_K230D_CANMV_LUSHANPI_LITE)

#define OTP_STATUS_LED_PIN        71 // BLUE
#define OTP_STATUS_LED_ACTIVE_LOW 1

#else

#define OTP_STATUS_LED_PIN        -1
#define OTP_STATUS_LED_ACTIVE_LOW 1

#endif

#endif
