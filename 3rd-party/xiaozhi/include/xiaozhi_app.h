#ifndef XIAOZHI_APP_H
#define XIAOZHI_APP_H

#include <signal.h>

#include "xiaozhi_config.h"

struct xiaozhi_app_config {
	char address[XIAOZHI_MAX_ADDRESS];
	char path[XIAOZHI_MAX_PATH];
	int port;
	int use_ssl;
	int allow_insecure;
	int ssl_explicit;

	char token[XIAOZHI_MAX_TOKEN];
	char activation_url[XIAOZHI_MAX_URL];
	char device_id[XIAOZHI_MAX_DEVICE_ID];
	char client_id[XIAOZHI_MAX_CLIENT_ID];
	int device_id_explicit;
	int client_id_explicit;
	int token_explicit;
	int activation_enabled;

	int timeout_secs;
	int duration_secs;
	int max_attempts;
	int log_level;

	int audio_enabled;
	int audio_input_device;
	int audio_input_channel;
	int audio_output_device;
	int audio_output_channel;
	int audio_internal_codec;
	int audio3a_mask;
	int audio3a_explicit;
	int mode;
	int wake_word_enabled;
	char wake_word_model[XIAOZHI_MAX_PATH];
	char wake_word_task[XIAOZHI_MAX_WAKE_WORD_TASK];
	char wake_word_text[XIAOZHI_MAX_WAKE_WORD_TEXT];
	int wake_word_keywords;
	float wake_word_threshold;
	int lvgl_enabled;
	int lvgl_connector;
	int lvgl_layer;
	int lvgl_touch_id;
	char lvgl_resource_dir[XIAOZHI_MAX_RESOURCE_DIR];
};

void xiaozhi_app_config_init(struct xiaozhi_app_config *config);

int xiaozhi_app_run(const struct xiaozhi_app_config *config,
		   volatile sig_atomic_t *interrupted);

#endif
