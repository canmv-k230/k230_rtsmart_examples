#ifndef XIAOZHI_LVGL_H
#define XIAOZHI_LVGL_H

#include <stddef.h>

#include "xiaozhi_config.h"

struct xiaozhi_lvgl;

struct xiaozhi_lvgl_config {
	int enabled;
	int connector;
	int layer;
	int touch_id;
	const char *resource_dir;
	int (*set_volume)(void *opaque, int volume);
	void *volume_opaque;
};

void xiaozhi_lvgl_config_init(struct xiaozhi_lvgl_config *config);

struct xiaozhi_lvgl *xiaozhi_lvgl_create(
	const struct xiaozhi_lvgl_config *config);
int xiaozhi_lvgl_start(struct xiaozhi_lvgl *ui);
void xiaozhi_lvgl_stop(struct xiaozhi_lvgl *ui);
void xiaozhi_lvgl_destroy(struct xiaozhi_lvgl *ui);

void xiaozhi_lvgl_set_connection(struct xiaozhi_lvgl *ui,
					 const char *state, const char *detail);
void xiaozhi_lvgl_set_wake_prompt(struct xiaozhi_lvgl *ui,
					 const char *wake_word);
void xiaozhi_lvgl_set_session(struct xiaozhi_lvgl *ui, const char *session_id);
void xiaozhi_lvgl_set_activation(struct xiaozhi_lvgl *ui,
					 const char *code, const char *message);
void xiaozhi_lvgl_set_audio(struct xiaozhi_lvgl *ui, int available,
					int volume);
void xiaozhi_lvgl_set_speaking(struct xiaozhi_lvgl *ui, int speaking);
void xiaozhi_lvgl_set_user_text(struct xiaozhi_lvgl *ui, const char *text);
void xiaozhi_lvgl_set_assistant_text(struct xiaozhi_lvgl *ui,
					     const char *text);
void xiaozhi_lvgl_set_emotion(struct xiaozhi_lvgl *ui,
				      const char *emotion, const char *emoji);

#endif
