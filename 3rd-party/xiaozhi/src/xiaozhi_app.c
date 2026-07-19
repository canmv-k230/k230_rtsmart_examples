#include "xiaozhi_app.h"

#include "xiaozhi_activation.h"
#include "xiaozhi_audio.h"
#include "xiaozhi_light.h"
#include "xiaozhi_lvgl.h"
#include "xiaozhi_mcp.h"
#include "xiaozhi_protocol.h"
#include "xiaozhi_transport.h"
#include "xiaozhi_wake.h"

#include <cJSON.h>
#include <hal_netmgmt.h>
#include <hal_utils.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

struct xiaozhi_app {
	struct xiaozhi_app_config config;
	struct xiaozhi_transport *transport;
	struct xiaozhi_audio *audio;
	struct xiaozhi_wake *wake;
	struct xiaozhi_light *light;
	struct xiaozhi_lvgl *lvgl;
	char auth_header[XIAOZHI_MAX_AUTH_HEADER];
	char session_id[XIAOZHI_MAX_SESSION_ID];
	int audio_ready;
	int speaking;
	int wake_active;
	int wake_drop_audio_frames;
	int volume;
	struct xiaozhi_mcp_device mcp_device;
	struct xiaozhi_mcp_server mcp;
};

static void app_destroy_lvgl(struct xiaozhi_app *app)
{
	if (!app || !app->lvgl)
		return;
	xiaozhi_lvgl_destroy(app->lvgl);
	app->lvgl = NULL;
}

static void app_destroy_light(struct xiaozhi_app *app)
{
	if (!app || !app->light)
		return;
	xiaozhi_light_destroy(&app->light);
}

static int mac_is_zero(const uint8_t mac[RT_WLAN_BSSID_MAX_LENGTH])
{
	size_t i;

	for (i = 0; i < RT_WLAN_BSSID_MAX_LENGTH; i++)
		if (mac[i])
			return 0;
	return 1;
}

static int chip_id_is_invalid(const uint8_t chip_id[32])
{
	int all_zero = 1;
	int all_ff = 1;
	int i;

	for (i = 0; i < 32; i++) {
		if (chip_id[i] != 0)
			all_zero = 0;
		if (chip_id[i] != 0xff)
			all_ff = 0;
	}
	return all_zero || all_ff;
}

static int build_board_uuid(char *buffer, size_t buffer_size)
{
	uint8_t chip_id[32];
	uint8_t uuid[16];
	int written;
	int i;

	if (utils_read_chipid(chip_id) || chip_id_is_invalid(chip_id))
		return -1;

	/* Fold the complete chip ID into a stable UUID-shaped Client-Id. */
	for (i = 0; i < 16; i++)
		uuid[i] = chip_id[i] ^ chip_id[31 - i];
	uuid[6] = (uuid[6] & 0x0f) | 0x40;
	uuid[8] = (uuid[8] & 0x3f) | 0x80;

	written = snprintf(buffer, buffer_size,
			   "%02x%02x%02x%02x-%02x%02x-%02x%02x-"
			   "%02x%02x-%02x%02x%02x%02x%02x%02x",
			   uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5],
			   uuid[6], uuid[7], uuid[8], uuid[9], uuid[10], uuid[11],
			   uuid[12], uuid[13], uuid[14], uuid[15]);
	return written < 0 || (size_t)written >= buffer_size ? -1 : 0;
}

static int load_identity(struct xiaozhi_app *app)
{
	uint8_t mac[RT_WLAN_BSSID_MAX_LENGTH];
	int have_mac;
	int have_board_uuid = 0;

	memset(mac, 0, sizeof(mac));
	have_mac = netmgmt_wlan_sta_get_mac(mac) == 0 && !mac_is_zero(mac);
	if (!app->config.device_id_explicit && have_mac) {
		if (snprintf(app->config.device_id, sizeof(app->config.device_id),
			     "%02x:%02x:%02x:%02x:%02x:%02x", mac[0], mac[1], mac[2],
			     mac[3], mac[4], mac[5]) >=
		    (int)sizeof(app->config.device_id))
			return -1;
	}
	if (!app->config.client_id_explicit) {
		have_board_uuid = !build_board_uuid(app->config.client_id,
						     sizeof(app->config.client_id));
		if (!have_board_uuid && have_mac) {
			if (snprintf(app->config.client_id,
				     sizeof(app->config.client_id),
				     "00000000-0000-4000-8000-%02x%02x%02x%02x%02x%02x",
				     mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]) >=
			    (int)sizeof(app->config.client_id))
				return -1;
		}
	}
	if (!have_mac)
		printf("xiaozhi: WLAN MAC unavailable; using configured Device-Id\n");
	if (have_board_uuid)
		printf("xiaozhi: Client-Id derived from K230 chip ID: %s\n",
		       app->config.client_id);
	return 0;
}

static int build_auth_header(struct xiaozhi_app *app)
{
	int written;

	if (!app->config.token[0]) {
		app->auth_header[0] = '\0';
		return 0;
	}
	if (strchr(app->config.token, ' ')) {
		written = snprintf(app->auth_header, sizeof(app->auth_header), "%s",
				   app->config.token);
	} else {
		written = snprintf(app->auth_header, sizeof(app->auth_header),
				   "Bearer %s", app->config.token);
	}
	return written < 0 || (size_t)written >= sizeof(app->auth_header) ? -1 : 0;
}

static int wait_for_activation_retry(volatile sig_atomic_t *interrupted)
{
	int remaining_ms;

	for (remaining_ms = XIAOZHI_ACTIVATION_RETRY_DELAY_MS;
	     remaining_ms > 0 && (!interrupted || !*interrupted);
	     remaining_ms -= 100)
		usleep(100000);
	return interrupted && *interrupted ? -1 : 0;
}

static int activate_device(struct xiaozhi_app *app,
			   volatile sig_atomic_t *interrupted)
{
	struct xiaozhi_activation_options options;
	struct xiaozhi_activation_result result;
	char last_code[XIAOZHI_MAX_ACTIVATION_CODE];
	char last_message[XIAOZHI_MAX_ACTIVATION_MESSAGE];
	int ret;

	memset(&options, 0, sizeof(options));
	options.url = app->config.activation_url;
	options.device_id = app->config.device_id;
	options.client_id = app->config.client_id;
	options.timeout_secs = app->config.timeout_secs;
	options.allow_insecure = app->config.allow_insecure;
	options.log_level = app->config.log_level;
	last_code[0] = '\0';
	last_message[0] = '\0';
	app->config.token[0] = '\0';

	printf("xiaozhi: requesting device activation\n");
	xiaozhi_lvgl_set_connection(app->lvgl, "activation",
					    "Requesting device activation");
	for (;;) {
		if (interrupted && *interrupted)
			return -1;
		ret = xiaozhi_activation_request(&options, &result);
		if (ret == 0) {
			if (result.token[0] &&
			    snprintf(app->config.token, sizeof(app->config.token),
				     "%s", result.token) >=
					(int)sizeof(app->config.token))
					return -1;
			printf("xiaozhi: device activation accepted\n");
			xiaozhi_lvgl_set_activation(app->lvgl, "", "");
			xiaozhi_lvgl_set_connection(app->lvgl, "connecting",
						    "Device activation accepted");
			return build_auth_header(app);
		}
		if (ret == 1) {
			if (strcmp(last_code, result.code)) {
				snprintf(last_code, sizeof(last_code), "%s", result.code);
				printf("xiaozhi: device activation code: %s\n",
				       result.code);
			}
			xiaozhi_lvgl_set_activation(app->lvgl, result.code,
						    result.message);
			if (result.message[0] && strcmp(last_message, result.message)) {
				snprintf(last_message, sizeof(last_message), "%s",
					 result.message);
				printf("xiaozhi: activation message: %s\n",
				       result.message);
			}
		} else {
			printf("xiaozhi: device activation request failed; retrying\n");
			xiaozhi_lvgl_set_connection(app->lvgl, "activation",
						    "Activation request failed; retrying");
		}
		if (wait_for_activation_retry(interrupted))
			return -1;
	}
}

static int app_send_text(struct xiaozhi_app *app, const char *message)
{
	if (xiaozhi_transport_send_text(app->transport, message,
					strlen(message))) {
		printf("xiaozhi: failed to queue text message\n");
		return -1;
	}
	return 0;
}

static int app_send_listen_start(struct xiaozhi_app *app)
{
	char message[1024];
	const char *mode;

	if (app->config.mode == XIAOZHI_MODE_REALTIME)
		mode = "realtime";
	else if (app->config.mode == XIAOZHI_MODE_MANUAL)
		mode = "manual";
	else
		mode = "auto";
	if (xiaozhi_build_listen_message(message, sizeof(message),
					 app->session_id, mode))
		return -1;
	return app_send_text(app, message);
}

static int app_is_tool_call_text(const char *text)
{
	return text && text[0] == '%';
}

static int app_on_capture(void *opaque, const int16_t *samples,
				  size_t sample_count)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;
	int ret;
	char message[1024];

	if (!app->config.wake_word_enabled || app->wake_active)
		return 0;
	ret = xiaozhi_wake_process(app->wake, samples, sample_count);
	if (ret < 0)
		return ret;
	if (ret > 0) {
		app->wake_active = 1;
		app->wake_drop_audio_frames = 1;
		xiaozhi_lvgl_set_connection(app->lvgl, "ready",
					    "Listening for speech");
		printf("xiaozhi: wake word detected; entering realtime listening\n");
		if (xiaozhi_build_wake_detect_message(
				message, sizeof(message), app->session_id,
				app->config.wake_word_text) ||
		    app_send_text(app, message) || app_send_listen_start(app)) {
			app->wake_active = 0;
			app->wake_drop_audio_frames = 0;
			printf("xiaozhi: failed to start realtime listening after wake word\n");
		}
	}
	return 0;
}

static int app_send_mcp_payload(void *opaque, const char *payload, size_t len)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;
	char message[XIAOZHI_MAX_JSON];

	if (xiaozhi_build_mcp_envelope(message, sizeof(message), app->session_id,
				       payload, len))
		return -1;
	return app_send_text(app, message);
}

static int app_audio_send(void *opaque, const void *data, size_t len)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	if (app->config.wake_word_enabled && !app->wake_active)
		return 0;
	if (app->wake_drop_audio_frames) {
		app->wake_drop_audio_frames--;
		return 0;
	}
	return xiaozhi_transport_send_binary(app->transport, data, len);
}

static int app_set_volume(void *opaque, int volume)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	if (!app->audio || !app->audio_ready)
		return -1;
	if (xiaozhi_audio_set_volume(app->audio, volume)) {
		printf("xiaozhi: failed to set speaker volume to %d\n", volume);
		return -1;
	}
	app->volume = volume;
	xiaozhi_lvgl_set_audio(app->lvgl, 1, volume);
	printf("xiaozhi: speaker volume set to %d\n", volume);
	return 0;
}

static int app_get_volume(void *opaque, int *volume)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	if (!volume)
		return -1;
	if (app->audio_ready && !xiaozhi_audio_get_volume(app->audio, volume)) {
		app->volume = *volume;
		return 0;
	}
	*volume = app->volume;
	return 0;
}

static int app_set_light(void *opaque, int on)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	if (!app || !app->light || xiaozhi_light_set(app->light, on)) {
		printf("xiaozhi: failed to set GPIO light %s\n",
		       on ? "on" : "off");
		return -1;
	}
	printf("xiaozhi: GPIO light %s\n", on ? "on" : "off");
	return 0;
}

static int app_get_light(void *opaque, int *on)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	if (!app || !app->light || !on)
		return -1;
	return xiaozhi_light_get(app->light, on);
}

static int app_on_connected(void *opaque)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;
	char hello[XIAOZHI_MAX_JSON];

	app->session_id[0] = '\0';
	app->wake_active = 0;
	app->wake_drop_audio_frames = 0;
	xiaozhi_lvgl_set_connection(app->lvgl, "connected",
					    "WebSocket connected; sending hello");
	if (xiaozhi_build_hello(hello, sizeof(hello)))
		return -1;
	printf("xiaozhi: connected; sending hello\n");
	return app_send_text(app, hello);
}

static void print_server_hello(const struct xiaozhi_server_hello *hello)
{
	if (hello->session_id[0])
		printf("xiaozhi: session_id=%s\n", hello->session_id);
	else
		printf("xiaozhi: server hello has no session_id\n");
	printf("xiaozhi: server audio parameters: sample_rate=%d channels=%d frame_duration=%d; decoder/playback=%d Hz\n",
	       hello->sample_rate, hello->channels, hello->frame_duration,
	       XIAOZHI_AUDIO_SAMPLE_RATE);
}

static int app_handle_server_hello(struct xiaozhi_app *app,
					   const char *data, size_t len)
{
	struct xiaozhi_server_hello hello;

	if (xiaozhi_parse_server_hello(data, len, &hello)) {
		printf("xiaozhi: invalid server hello\n");
		return -1;
	}
	memcpy(app->session_id, hello.session_id, sizeof(app->session_id));
	print_server_hello(&hello);
	xiaozhi_lvgl_set_session(app->lvgl, app->session_id);
	xiaozhi_lvgl_set_connection(app->lvgl, "ready", "Server hello accepted");
	xiaozhi_lvgl_set_audio(app->lvgl, 0, app->volume);

	if (app->config.audio_enabled && app->audio && !app->audio_ready) {
		/* Match CanMV: let the Opus decoder emit the local 16 kHz PCM rate. */
		if (xiaozhi_audio_set_decode_format(
				app->audio, XIAOZHI_AUDIO_SAMPLE_RATE,
				XIAOZHI_AUDIO_CHANNELS,
				hello.frame_duration) ||
		    xiaozhi_audio_set_output_format(
				app->audio, XIAOZHI_AUDIO_SAMPLE_RATE,
				XIAOZHI_AUDIO_CHANNELS, hello.frame_duration) ||
		    xiaozhi_audio_initialize(app->audio)) {
			printf("xiaozhi: audio initialization failed; continuing without audio\n");
		} else {
			app->audio_ready = 1;
			xiaozhi_lvgl_set_audio(app->lvgl, 1, app->volume);
			printf("xiaozhi: K230 audio pipeline ready (server %d Hz -> decoder/playback %d Hz; direct Opus decode)\n",
			       hello.sample_rate, XIAOZHI_AUDIO_SAMPLE_RATE);
		}
	}

	xiaozhi_transport_mark_ready(app->transport);

	if (app->audio_ready && app->config.mode != XIAOZHI_MODE_MANUAL) {
		if (app->config.wake_word_enabled) {
			if (!app->wake ||
			    xiaozhi_audio_set_capture_callback(app->audio,
								app_on_capture, app))
				return -1;
			printf("xiaozhi: waiting for local wake word\n");
		} else if (app_send_listen_start(app)) {
			return -1;
		}
		if (xiaozhi_audio_start_capture(app->audio)) {
			printf("xiaozhi: failed to start microphone capture\n");
			app->audio_ready = 0;
			xiaozhi_lvgl_set_audio(app->lvgl, 0, app->volume);
		}
	}
	if (app->audio_ready && app->config.mode != XIAOZHI_MODE_MANUAL) {
		if (app->config.wake_word_enabled) {
			xiaozhi_lvgl_set_wake_prompt(
				app->lvgl, app->config.wake_word_text);
		} else {
			xiaozhi_lvgl_set_connection(app->lvgl, "ready",
						    "Listening for speech");
		}
	}

	printf("xiaozhi: session is ready\n");
	return 0;
}

static int app_on_text(void *opaque, const char *data, size_t len)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;
	cJSON *root;
	cJSON *type;
	cJSON *payload;
	cJSON *state;
	cJSON *text;

	root = cJSON_ParseWithLength(data, len);
	if (!root) {
		printf("xiaozhi: invalid JSON message\n");
		return 0;
	}
	type = cJSON_GetObjectItem(root, "type");
	if (!cJSON_IsString(type)) {
		cJSON_Delete(root);
		return 0;
	}
	text = cJSON_GetObjectItem(root, "text");

	if (!strcmp(type->valuestring, "hello")) {
		int ret = app_handle_server_hello(app, data, len);
		cJSON_Delete(root);
		return ret;
	}

	if (!strcmp(type->valuestring, "mcp")) {
		payload = cJSON_GetObjectItem(root, "payload");
		if (cJSON_IsObject(payload) &&
		    xiaozhi_mcp_handle(&app->mcp, payload, app_send_mcp_payload, app))
			printf("xiaozhi: ignored invalid MCP request\n");
		cJSON_Delete(root);
		return 0;
	}

	if (!strcmp(type->valuestring, "tts")) {
		state = cJSON_GetObjectItem(root, "state");
		text = cJSON_GetObjectItem(root, "text");
		if (cJSON_IsString(state) && !strcmp(state->valuestring, "start")) {
			app->speaking = 1;
			xiaozhi_lvgl_set_speaking(app->lvgl, 1);
			if (app->audio_ready) {
				/* Realtime mode keeps the microphone open for barge-in. */
				if (app->config.mode != XIAOZHI_MODE_REALTIME)
					xiaozhi_audio_stop_capture(app->audio);
				xiaozhi_audio_clear_playback(app->audio);
			}
			printf("xiaozhi: TTS started\n");
		} else if (cJSON_IsString(state) &&
			   !strcmp(state->valuestring, "stop")) {
			app->speaking = 0;
			xiaozhi_lvgl_set_speaking(app->lvgl, 0);
			printf("xiaozhi: TTS stopped\n");
			if (app->audio_ready && app->config.mode != XIAOZHI_MODE_MANUAL) {
				/* Realtime keeps one continuous listen session; restarting it
				 * here can discard the speech that interrupted TTS. */
				if ((app->config.mode != XIAOZHI_MODE_REALTIME &&
				     app_send_listen_start(app)) ||
				    xiaozhi_audio_start_capture(app->audio))
					printf("xiaozhi: failed to resume microphone capture\n");
			}
		} else if (cJSON_IsString(state) &&
			   !strcmp(state->valuestring, "sentence_start") &&
			   cJSON_IsString(text)) {
			if (app_is_tool_call_text(text->valuestring)) {
				printf("xiaozhi: assistant tool call: %s\n",
				       text->valuestring);
			} else {
				xiaozhi_lvgl_set_assistant_text(app->lvgl,
							text->valuestring);
				printf("xiaozhi: assistant: %s\n", text->valuestring);
			}
		}
		cJSON_Delete(root);
		return 0;
	}

	if (!strcmp(type->valuestring, "stt") && cJSON_IsString(text)) {
		xiaozhi_lvgl_set_user_text(app->lvgl, text->valuestring);
		printf("xiaozhi: user: %s\n", text->valuestring);
	} else if (!strcmp(type->valuestring, "llm")) {
		cJSON *emotion = cJSON_GetObjectItem(root, "emotion");
		const char *emotion_name = "neutral";
		const char *emoji = "";

		if (cJSON_IsString(emotion) && emotion->valuestring[0])
			emotion_name = emotion->valuestring;
		if (cJSON_IsString(text))
			emoji = text->valuestring;
		if (emotion_name[0] && (emoji[0] || cJSON_IsString(emotion))) {
			xiaozhi_lvgl_set_emotion(app->lvgl, emotion_name, emoji);
			printf("xiaozhi: emotion: %s\n", emotion_name);
		}
	} else if (!strcmp(type->valuestring, "system") ||
		   !strcmp(type->valuestring, "alert")) {
		printf("xiaozhi: received %s message\n", type->valuestring);
	} else {
		printf("xiaozhi: received JSON message type=%s\n", type->valuestring);
	}

	cJSON_Delete(root);
	return 0;
}

static int app_on_binary(void *opaque, const unsigned char *data, size_t len)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	if (app->audio_ready && xiaozhi_audio_queue_opus(app->audio, data, len))
		printf("xiaozhi: dropped incoming Opus packet (%zu bytes)\n", len);
	return 0;
}

static void app_on_closed(void *opaque)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	xiaozhi_lvgl_set_connection(app->lvgl, "offline", "WebSocket connection closed");
	printf("xiaozhi: WebSocket connection closed\n");
}

static void app_on_error(void *opaque, const char *reason)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	xiaozhi_lvgl_set_connection(app->lvgl, "error", reason ? reason : "Transport error");
	printf("xiaozhi: transport error: %s\n", reason ? reason : "unknown");
}

static void app_on_stopping(void *opaque)
{
	struct xiaozhi_app *app = (struct xiaozhi_app *)opaque;

	xiaozhi_lvgl_set_connection(app->lvgl, "stopping", "Stopping audio and transport");
	app->speaking = 0;
	app->wake_active = 0;
	app->wake_drop_audio_frames = 0;
	xiaozhi_lvgl_set_speaking(app->lvgl, 0);
	if (app->audio) {
		xiaozhi_audio_deinitialize(app->audio);
		app->audio_ready = 0;
		xiaozhi_lvgl_set_audio(app->lvgl, 0, app->volume);
	}
	if (app->wake && xiaozhi_wake_reset(app->wake))
		printf("xiaozhi: failed to reset wake-word detector\n");
}

void xiaozhi_app_config_init(struct xiaozhi_app_config *config)
{
	memset(config, 0, sizeof(*config));
	snprintf(config->address, sizeof(config->address), "%s",
		 XIAOZHI_DEFAULT_ADDRESS);
	snprintf(config->path, sizeof(config->path), "%s", XIAOZHI_DEFAULT_PATH);
	snprintf(config->activation_url, sizeof(config->activation_url), "%s",
		 XIAOZHI_DEFAULT_OTA_URL);
	config->port = XIAOZHI_DEFAULT_PORT;
	config->use_ssl = 1;
	config->allow_insecure = 1;
	snprintf(config->token, sizeof(config->token), "%s",
		 XIAOZHI_DEFAULT_TOKEN);
	snprintf(config->device_id, sizeof(config->device_id),
		 "00:00:00:00:00:00");
	snprintf(config->client_id, sizeof(config->client_id),
		 "canmv-k230-xiaozhi");
	config->timeout_secs = XIAOZHI_DEFAULT_TIMEOUT_SECS;
	config->max_attempts = XIAOZHI_DEFAULT_MAX_ATTEMPTS;
	config->log_level = XIAOZHI_DEFAULT_LOG_LEVEL;
	config->activation_enabled = 1;
	config->audio_enabled = 1;
	config->audio_input_device = 0;
	config->audio_input_channel = 0;
	config->audio_output_device = 0;
	config->audio_output_channel = 0;
	config->audio_internal_codec = 1;
	config->mode = XIAOZHI_MODE_REALTIME;
	config->wake_word_enabled = 1;
	snprintf(config->wake_word_model, sizeof(config->wake_word_model), "%s",
		 XIAOZHI_DEFAULT_WAKE_WORD_MODEL);
	snprintf(config->wake_word_task, sizeof(config->wake_word_task), "%s",
		 XIAOZHI_DEFAULT_WAKE_WORD_TASK);
	snprintf(config->wake_word_text, sizeof(config->wake_word_text), "%s",
		 XIAOZHI_DEFAULT_WAKE_WORD_TEXT);
	config->wake_word_keywords = XIAOZHI_DEFAULT_WAKE_WORD_KEYWORDS;
	config->wake_word_threshold = XIAOZHI_DEFAULT_WAKE_WORD_THRESHOLD;
	config->lvgl_enabled = XIAOZHI_HAS_LVGL;
	config->lvgl_connector = 605274512;
	config->lvgl_layer = 0;
	config->lvgl_touch_id = 0;
	snprintf(config->lvgl_resource_dir, sizeof(config->lvgl_resource_dir), "%s",
		 XIAOZHI_DEFAULT_LVGL_RESOURCE_DIR);
}

int xiaozhi_app_run(const struct xiaozhi_app_config *config,
		   volatile sig_atomic_t *interrupted)
{
	struct xiaozhi_app app;
	struct xiaozhi_audio_config audio_config;
	struct xiaozhi_wake_config wake_config;
	struct xiaozhi_light_config light_config;
	struct xiaozhi_lvgl_config lvgl_config;
	struct xiaozhi_transport_options transport_options;
	struct xiaozhi_transport_events transport_events;
	int attempt;
	int ret = 1;

	if (!config)
		return 1;
	memset(&app, 0, sizeof(app));
	app.config = *config;
	app.volume = 90;
	app.mcp_device.name = "canmv-k230";
	app.mcp_device.version = "1.0.0";
	app.mcp_device.opaque = &app;

	xiaozhi_lvgl_config_init(&lvgl_config);
	lvgl_config.enabled = app.config.lvgl_enabled;
	if (app.config.lvgl_connector)
		lvgl_config.connector = app.config.lvgl_connector;
	if (app.config.lvgl_layer)
		lvgl_config.layer = app.config.lvgl_layer;
	lvgl_config.touch_id = app.config.lvgl_touch_id;
	lvgl_config.resource_dir = app.config.lvgl_resource_dir;
	if (app.config.audio_enabled && app.config.audio_internal_codec) {
		lvgl_config.set_volume = app_set_volume;
		lvgl_config.volume_opaque = &app;
	}
	app.lvgl = xiaozhi_lvgl_create(&lvgl_config);
	if (app.config.lvgl_enabled && !app.lvgl)
		printf("xiaozhi: failed to allocate LVGL UI; continuing headless\n");
	if (app.lvgl) {
		if (xiaozhi_lvgl_start(app.lvgl)) {
			printf("xiaozhi: LVGL UI unavailable; continuing headless\n");
			app_destroy_lvgl(&app);
		} else {
			xiaozhi_lvgl_set_connection(app.lvgl, "starting",
						    "Loading board identity");
			xiaozhi_lvgl_set_audio(app.lvgl, 0, app.volume);
		}
	}

	if (load_identity(&app)) {
		app_destroy_lvgl(&app);
		return 1;
	}
	if (app.config.activation_enabled && !app.config.token_explicit) {
		if (activate_device(&app, interrupted)) {
			ret = interrupted && *interrupted ? 0 : 1;
			app_destroy_lvgl(&app);
			return ret;
		}
	} else if (build_auth_header(&app)) {
		app_destroy_lvgl(&app);
		return 1;
	} else {
		xiaozhi_lvgl_set_activation(app.lvgl, "", "");
		xiaozhi_lvgl_set_connection(app.lvgl, "connecting",
					    "Using configured WebSocket token");
	}

	if (app.config.wake_word_enabled) {
		memset(&wake_config, 0, sizeof(wake_config));
		wake_config.model_path = app.config.wake_word_model;
		wake_config.task_name = app.config.wake_word_task;
		wake_config.keyword_count = app.config.wake_word_keywords;
		wake_config.threshold = app.config.wake_word_threshold;
		app.wake = xiaozhi_wake_create(&wake_config);
		if (!app.wake) {
			printf("xiaozhi: failed to load wake-word model: %s\n",
			       app.config.wake_word_model);
			app_destroy_lvgl(&app);
			return 1;
		}
	}

	memset(&audio_config, 0, sizeof(audio_config));
	audio_config.enabled = app.config.audio_enabled;
	audio_config.input_device = app.config.audio_input_device;
	audio_config.input_channel = app.config.audio_input_channel;
	audio_config.output_device = app.config.audio_output_device;
	audio_config.output_channel = app.config.audio_output_channel;
	audio_config.internal_codec = app.config.audio_internal_codec;
	audio_config.sample_rate = XIAOZHI_AUDIO_SAMPLE_RATE;
	audio_config.channels = XIAOZHI_AUDIO_CHANNELS;
	audio_config.frame_samples = XIAOZHI_AUDIO_FRAME_SAMPLES;
	audio_config.bitrate = XIAOZHI_AUDIO_BITRATE;
	audio_config.audio3a_mask = app.config.audio3a_mask;
	audio_config.input_sample_rate = XIAOZHI_AUDIO_SAMPLE_RATE;
	audio_config.input_frame_samples = XIAOZHI_AUDIO_FRAME_SAMPLES;
	audio_config.decode_sample_rate = XIAOZHI_AUDIO_SAMPLE_RATE;
	audio_config.decode_channels = XIAOZHI_AUDIO_CHANNELS;
	audio_config.decode_frame_samples = XIAOZHI_AUDIO_FRAME_SAMPLES;
	audio_config.output_sample_rate = XIAOZHI_AUDIO_SAMPLE_RATE;
	audio_config.output_channels = XIAOZHI_AUDIO_CHANNELS;
	audio_config.output_frame_samples = XIAOZHI_AUDIO_FRAME_SAMPLES;
	app.audio = xiaozhi_audio_create(&audio_config, app_audio_send, &app);
	if (!app.audio && app.config.audio_enabled)
		printf("xiaozhi: failed to allocate audio manager; audio disabled\n");
	if (app.config.audio_enabled && app.config.audio_internal_codec && app.audio) {
		app.mcp_device.get_volume = app_get_volume;
		app.mcp_device.set_volume = app_set_volume;
	}
	xiaozhi_light_config_init(&light_config);
	app.light = xiaozhi_light_create(&light_config);
	if (light_config.enabled && !app.light)
		printf("xiaozhi: GPIO light unavailable; continuing without light MCP tool\n");
	if (app.light) {
		app.mcp_device.light_available = 1;
		app.mcp_device.set_light = app_set_light;
		app.mcp_device.get_light = app_get_light;
	}
	if (xiaozhi_mcp_server_init(&app.mcp, &app.mcp_device) ||
	    xiaozhi_mcp_register_default_tools(&app.mcp)) {
		printf("xiaozhi: failed to initialize MCP server\n");
		app_destroy_lvgl(&app);
		xiaozhi_wake_destroy(app.wake);
		xiaozhi_audio_destroy(app.audio);
		app_destroy_light(&app);
		return 1;
	}
	printf("xiaozhi: MCP server ready (%zu tools)\n", app.mcp.tool_count);

	memset(&transport_options, 0, sizeof(transport_options));
	transport_options.address = app.config.address;
	transport_options.path = app.config.path;
	transport_options.port = app.config.port;
	transport_options.use_ssl = app.config.use_ssl;
	transport_options.allow_insecure = app.config.allow_insecure;
	transport_options.auth_header = app.auth_header;
	transport_options.device_id = app.config.device_id;
	transport_options.client_id = app.config.client_id;
	transport_options.timeout_secs = app.config.timeout_secs;
	transport_options.log_level = app.config.log_level;

	memset(&transport_events, 0, sizeof(transport_events));
	transport_events.opaque = &app;
	transport_events.on_connected = app_on_connected;
	transport_events.on_text = app_on_text;
	transport_events.on_binary = app_on_binary;
	transport_events.on_closed = app_on_closed;
	transport_events.on_error = app_on_error;
	transport_events.on_stopping = app_on_stopping;
	app.transport = xiaozhi_transport_create(&transport_options,
							 &transport_events);
	if (!app.transport) {
		printf("xiaozhi: failed to create WebSocket transport\n");
		app_destroy_lvgl(&app);
		xiaozhi_wake_destroy(app.wake);
		xiaozhi_audio_destroy(app.audio);
		app_destroy_light(&app);
		return 1;
	}

	printf("xiaozhi: TLS certificate verification %s\n",
	       app.config.allow_insecure ? "disabled" : "enabled");
	for (attempt = 1;
	     !interrupted || !*interrupted;
	     attempt++) {
		if (app.config.max_attempts && attempt > app.config.max_attempts)
			break;
		if (app.config.max_attempts)
			printf("xiaozhi: connection attempt %d/%d\n", attempt,
			       app.config.max_attempts);
		else
			printf("xiaozhi: connection attempt %d\n", attempt);
		xiaozhi_lvgl_set_connection(app.lvgl, "connecting",
					    "Opening WebSocket connection");

		ret = xiaozhi_transport_run(app.transport, app.config.duration_secs,
					    interrupted);
		if (!ret || (interrupted && *interrupted))
			break;
		if (app.config.max_attempts && attempt >= app.config.max_attempts)
			break;
		printf("xiaozhi: retrying connection in %d ms\n",
		       XIAOZHI_RETRY_DELAY_MS);
		{
			int remaining_ms;

			for (remaining_ms = XIAOZHI_RETRY_DELAY_MS;
			     remaining_ms > 0 && (!interrupted || !*interrupted);
			     remaining_ms -= 100)
				usleep(100000);
		}
	}

	xiaozhi_transport_destroy(app.transport);
	xiaozhi_audio_destroy(app.audio);
	xiaozhi_wake_destroy(app.wake);
	app_destroy_light(&app);
	app_destroy_lvgl(&app);
	if (interrupted && *interrupted)
		return 0;
	if (app.config.max_attempts && attempt > app.config.max_attempts)
		printf("xiaozhi: unable to establish a session after %d attempts\n",
		       app.config.max_attempts);
	return ret ? 1 : 0;
}
