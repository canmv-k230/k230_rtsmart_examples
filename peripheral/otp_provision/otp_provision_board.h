#ifndef OTP_PROVISION_BOARD_H
#define OTP_PROVISION_BOARD_H

/*
 * Board-local final status LED settings for otp_provision.
 *
 * Adjust these values per board.
 *
 * OTP_SUCCESS_LED_PIN:
 *   GPIO pin number used to report successful OTP provisioning.
 *   Set to -1 to disable the success LED.
 *
 * OTP_FAILURE_LED_PIN:
 *   GPIO pin number used to report failed OTP provisioning.
 *   Set to -1 to disable the failure LED.
 *
 * To use one LED for every final state, set only OTP_SUCCESS_LED_PIN
 * or set OTP_SUCCESS_LED_PIN and OTP_FAILURE_LED_PIN to the same GPIO.
 *
 * OTP_*_LED_ACTIVE_POL:
 *   GPIO output value that turns the LED on.
 *   1 when driving the pin high turns it on.
 *   0 when driving the pin low turns it on.
 */

#if defined(CONFIG_BOARD_K230_CANMV_01STUDIO)

#define OTP_DEFAULT_SUCCESS_LED_PIN        52
#define OTP_DEFAULT_SUCCESS_LED_ACTIVE_POL 0
#define OTP_DEFAULT_FAILURE_LED_PIN        -1
#define OTP_DEFAULT_FAILURE_LED_ACTIVE_POL 0

#elif defined(CONFIG_BOARD_K230_CANMV_LCKFB)

#define OTP_DEFAULT_SUCCESS_LED_PIN        63 // BLUE
#define OTP_DEFAULT_SUCCESS_LED_ACTIVE_POL 0
#define OTP_DEFAULT_FAILURE_LED_PIN        -1
#define OTP_DEFAULT_FAILURE_LED_ACTIVE_POL 0

#elif defined(CONFIG_BOARD_K230D_CANMV_LUSHANPI_LITE)

#define OTP_DEFAULT_SUCCESS_LED_PIN        66 // GREEN
#define OTP_DEFAULT_SUCCESS_LED_ACTIVE_POL 1
#define OTP_DEFAULT_FAILURE_LED_PIN        65 // RED
#define OTP_DEFAULT_FAILURE_LED_ACTIVE_POL 1

#else

#define OTP_DEFAULT_SUCCESS_LED_PIN        -1
#define OTP_DEFAULT_SUCCESS_LED_ACTIVE_POL 1
#define OTP_DEFAULT_FAILURE_LED_PIN        -1
#define OTP_DEFAULT_FAILURE_LED_ACTIVE_POL 1

#endif

#ifndef OTP_SUCCESS_LED_PIN
#define OTP_SUCCESS_LED_PIN OTP_DEFAULT_SUCCESS_LED_PIN
#endif

#ifndef OTP_SUCCESS_LED_ACTIVE_POL
#define OTP_SUCCESS_LED_ACTIVE_POL OTP_DEFAULT_SUCCESS_LED_ACTIVE_POL
#endif

#ifndef OTP_FAILURE_LED_PIN
#define OTP_FAILURE_LED_PIN OTP_DEFAULT_FAILURE_LED_PIN
#endif

#ifndef OTP_FAILURE_LED_ACTIVE_POL
#define OTP_FAILURE_LED_ACTIVE_POL OTP_DEFAULT_FAILURE_LED_ACTIVE_POL
#endif

#endif
