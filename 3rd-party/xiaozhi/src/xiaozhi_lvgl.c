#include "xiaozhi_lvgl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if XIAOZHI_HAS_LVGL

#include "lv_k230_display.h"
#include "lv_k230_input_touch.h"
#include "lvgl.h"

#include "k_connector_compat.h"
#include "k_vo_comm.h"
#include "kd_display.h"

#include <pthread.h>
#include <unistd.h>

#define XIAOZHI_LVGL_DEFAULT_CONNECTOR ST7701_V1_MIPI_2LAN_480X800_30FPS
#define XIAOZHI_LVGL_DEFAULT_LAYER K_VO_LAYER_OSD0
#define XIAOZHI_LVGL_DEFAULT_TOUCH_ID 0
/* The K230 GDMA direction is opposite to LVGL's 90-degree enum. */
#define XIAOZHI_LVGL_DEFAULT_ROTATION LV_DISPLAY_ROTATION_270

#define XIAOZHI_LVGL_CONNECTION_SIZE 32
#define XIAOZHI_LVGL_DETAIL_SIZE 160
#define XIAOZHI_LVGL_ACTIVATION_SIZE XIAOZHI_MAX_ACTIVATION_CODE
#define XIAOZHI_LVGL_EMOTION_SIZE 64
#define XIAOZHI_LVGL_EMOJI_SIZE 16
#define XIAOZHI_LVGL_LINE_SIZE 320
#define XIAOZHI_LVGL_PATH_SIZE (XIAOZHI_MAX_RESOURCE_DIR + 96)
#define XIAOZHI_LVGL_FONT_SIZE 20
#define XIAOZHI_LVGL_EMOTION_PANEL_WIDTH 148
#define XIAOZHI_LVGL_EMOTION_IMAGE_SIZE 128
#define XIAOZHI_LVGL_EMOTION_IMAGE_INNER_SIZE 112
#define XIAOZHI_LVGL_FS_PREFIX "S:"

struct xiaozhi_lvgl_snapshot {
	char connection[XIAOZHI_LVGL_CONNECTION_SIZE];
	char detail[XIAOZHI_LVGL_DETAIL_SIZE];
	char session[XIAOZHI_MAX_SESSION_ID];
	char activation[XIAOZHI_LVGL_ACTIVATION_SIZE];
	char activation_message[XIAOZHI_MAX_ACTIVATION_MESSAGE];
	char log[XIAOZHI_LVGL_LOG_SIZE];
	char emotion[XIAOZHI_LVGL_EMOTION_SIZE];
	char emoji[XIAOZHI_LVGL_EMOJI_SIZE];
	int activation_complete;
	int audio_available;
	int speaking;
	int volume;
};

struct xiaozhi_lvgl {
	struct xiaozhi_lvgl_config config;
	pthread_mutex_t lock;
	pthread_t thread;
	int thread_started;
	int running;

	struct xiaozhi_lvgl_snapshot state;
	struct xiaozhi_lvgl_snapshot rendered;

	lv_display_t *display;
	lv_indev_t *touch;
	lv_obj_t *connection_label;
	lv_obj_t *activation_label;
	lv_obj_t *message_label;
	lv_obj_t *log_panel;
	lv_obj_t *emotion_image;
#if LV_USE_GIF
	lv_obj_t *emotion_gif;
#endif
	lv_obj_t *emotion_label;
	lv_obj_t *session_label;
	lv_obj_t *audio_label;
	lv_obj_t *volume_label;
	lv_obj_t *volume_slider;
	lv_obj_t *detail_label;
	lv_font_t *text_font;
	char emotion_image_path[XIAOZHI_LVGL_PATH_SIZE];
	char emotion_gif_path[XIAOZHI_LVGL_PATH_SIZE];
	int applying_state;
};

static void copy_text(char *dst, size_t dst_size, const char *src)
{
	if (!dst || !dst_size)
		return;
	snprintf(dst, dst_size, "%s", src ? src : "");
}

static void clamp_snapshot(struct xiaozhi_lvgl_snapshot *state)
{
	if (state->volume < 0)
		state->volume = 0;
	if (state->volume > 100)
		state->volume = 100;
}

static void append_log(char *log, size_t log_size, const char *prefix,
			       const char *value)
{
	char line[XIAOZHI_LVGL_LINE_SIZE];
	size_t current;
	size_t line_size;
	int written;

	if (!log || !log_size || !value || !value[0])
		return;
	written = snprintf(line, sizeof(line), "%s%s\n", prefix ? prefix : "",
				   value);
	if (written < 0)
		return;
	line[sizeof(line) - 1] = '\0';
	line_size = strlen(line);
	if (line_size >= log_size) {
		memcpy(log, line, log_size - 1);
		log[log_size - 1] = '\0';
		return;
	}

	current = strlen(log);
	while (current + line_size >= log_size) {
		char *line_end = strchr(log, '\n');
		size_t removed;

		if (!line_end) {
			log[0] = '\0';
			current = 0;
			break;
		}
		removed = (size_t)(line_end - log) + 1;
		memmove(log, log + removed, current - removed + 1);
		current -= removed;
	}
	memcpy(log + current, line, line_size + 1);
}

static void volume_slider_event(lv_event_t *event)
{
	struct xiaozhi_lvgl *ui = lv_event_get_user_data(event);
	lv_obj_t *slider = lv_event_get_target(event);
	int volume;

	if (!ui || ui->applying_state || !ui->config.set_volume)
		return;
	volume = (int)lv_slider_get_value(slider);
	if (ui->config.set_volume(ui->config.volume_opaque, volume)) {
		printf("xiaozhi: LVGL volume update failed\n");
		return;
	}
	xiaozhi_lvgl_set_audio(ui, 1, volume);
}

static void apply_font(struct xiaozhi_lvgl *ui, lv_obj_t *obj)
{
	if (ui->text_font)
		lv_obj_set_style_text_font(obj, ui->text_font, 0);
}

static void create_ui(struct xiaozhi_lvgl *ui)
{
	lv_obj_t *screen;
	lv_obj_t *header;
	lv_obj_t *title;
	lv_obj_t *content;
	lv_obj_t *emotion_panel;
	lv_obj_t *emotion_holder;
	lv_obj_t *controls;
	lv_obj_t *volume_title;
	lv_obj_t *footer;

	screen = lv_obj_create(NULL);
	lv_obj_remove_flag(screen, LV_OBJ_FLAG_SCROLLABLE);
	lv_obj_set_style_bg_color(screen, lv_color_hex(0x101820), 0);
	lv_obj_set_style_bg_opa(screen, LV_OPA_COVER, 0);
	lv_obj_set_style_pad_all(screen, 12, 0);
	lv_obj_set_style_pad_row(screen, 8, 0);
	lv_obj_set_flex_flow(screen, LV_FLEX_FLOW_COLUMN);
	lv_obj_set_style_text_font(screen,
				   ui->text_font ? ui->text_font : LV_FONT_DEFAULT, 0);

	header = lv_obj_create(screen);
	lv_obj_remove_flag(header, LV_OBJ_FLAG_SCROLLABLE);
	lv_obj_set_width(header, lv_pct(100));
	lv_obj_set_height(header, LV_SIZE_CONTENT);
	lv_obj_set_style_pad_all(header, 8, 0);
	lv_obj_set_style_radius(header, 8, 0);
	lv_obj_set_style_bg_color(header, lv_color_hex(0x1d2b36), 0);
	lv_obj_set_style_border_width(header, 0, 0);
	lv_obj_set_flex_flow(header, LV_FLEX_FLOW_ROW);
	lv_obj_set_flex_align(header, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER,
				      LV_FLEX_ALIGN_CENTER);

	title = lv_label_create(header);
	lv_label_set_text(title, "XiaoZhi");
	lv_obj_set_style_text_color(title, lv_color_hex(0x7dd3fc), 0);
	apply_font(ui, title);
	lv_obj_set_flex_grow(title, 1);

	ui->connection_label = lv_label_create(header);
	lv_label_set_text(ui->connection_label, "starting");
	lv_obj_set_style_text_color(ui->connection_label, lv_color_hex(0xfbbf24), 0);
	apply_font(ui, ui->connection_label);

	ui->activation_label = lv_label_create(screen);
	lv_obj_set_width(ui->activation_label, lv_pct(100));
	lv_label_set_long_mode(ui->activation_label, LV_LABEL_LONG_WRAP);
	lv_label_set_text(ui->activation_label, "Waiting for device activation.");
	lv_obj_set_style_text_color(ui->activation_label, lv_color_hex(0xfde68a), 0);
	apply_font(ui, ui->activation_label);

	content = lv_obj_create(screen);
	lv_obj_remove_flag(content, LV_OBJ_FLAG_SCROLLABLE);
	lv_obj_set_width(content, lv_pct(100));
	lv_obj_set_style_pad_all(content, 10, 0);
	lv_obj_set_style_pad_column(content, 10, 0);
	lv_obj_set_style_radius(content, 8, 0);
	lv_obj_set_style_bg_color(content, lv_color_hex(0x17232d), 0);
	lv_obj_set_style_border_width(content, 0, 0);
	lv_obj_set_flex_grow(content, 1);
	lv_obj_set_flex_flow(content, LV_FLEX_FLOW_ROW);
	lv_obj_set_flex_align(content, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START,
				      LV_FLEX_ALIGN_START);

	emotion_panel = lv_obj_create(content);
	lv_obj_remove_flag(emotion_panel, LV_OBJ_FLAG_SCROLLABLE);
	lv_obj_set_width(emotion_panel, XIAOZHI_LVGL_EMOTION_PANEL_WIDTH);
	lv_obj_set_height(emotion_panel, lv_pct(100));
	lv_obj_set_style_pad_all(emotion_panel, 8, 0);
	lv_obj_set_style_pad_row(emotion_panel, 8, 0);
	lv_obj_set_style_radius(emotion_panel, 8, 0);
	lv_obj_set_style_bg_color(emotion_panel, lv_color_hex(0x263746), 0);
	lv_obj_set_style_border_width(emotion_panel, 0, 0);
	lv_obj_set_flex_flow(emotion_panel, LV_FLEX_FLOW_COLUMN);
	lv_obj_set_flex_align(emotion_panel, LV_FLEX_ALIGN_CENTER,
				      LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);

	emotion_holder = lv_obj_create(emotion_panel);
	lv_obj_remove_flag(emotion_holder, LV_OBJ_FLAG_SCROLLABLE);
	lv_obj_set_size(emotion_holder, XIAOZHI_LVGL_EMOTION_IMAGE_SIZE,
				XIAOZHI_LVGL_EMOTION_IMAGE_SIZE);
	lv_obj_set_style_pad_all(emotion_holder, 0, 0);
	lv_obj_set_style_radius(emotion_holder, 12, 0);
	/* The supplied GIFs use transparent backgrounds and dark artwork. */
	lv_obj_set_style_bg_color(emotion_holder, lv_color_hex(0x38bdf8), 0);
	lv_obj_set_style_bg_opa(emotion_holder, LV_OPA_COVER, 0);
	lv_obj_set_style_border_color(emotion_holder, lv_color_hex(0x7dd3fc), 0);
	lv_obj_set_style_border_width(emotion_holder, 2, 0);

	ui->emotion_image = lv_image_create(emotion_holder);
	lv_obj_set_size(ui->emotion_image, XIAOZHI_LVGL_EMOTION_IMAGE_INNER_SIZE,
				XIAOZHI_LVGL_EMOTION_IMAGE_INNER_SIZE);
	lv_obj_set_style_opa(ui->emotion_image, LV_OPA_COVER, LV_PART_MAIN);
	lv_obj_set_style_image_opa(ui->emotion_image, LV_OPA_COVER,
					LV_PART_MAIN);
	lv_image_set_inner_align(ui->emotion_image, LV_IMAGE_ALIGN_CONTAIN);
	lv_obj_align(ui->emotion_image, LV_ALIGN_CENTER, 0, 0);
	lv_obj_add_flag(ui->emotion_image, LV_OBJ_FLAG_HIDDEN);
#if LV_USE_GIF
	ui->emotion_gif = lv_gif_create(emotion_holder);
	lv_obj_set_size(ui->emotion_gif, XIAOZHI_LVGL_EMOTION_IMAGE_INNER_SIZE,
				XIAOZHI_LVGL_EMOTION_IMAGE_INNER_SIZE);
	lv_obj_set_style_opa(ui->emotion_gif, LV_OPA_COVER, LV_PART_MAIN);
	lv_obj_set_style_image_opa(ui->emotion_gif, LV_OPA_COVER, LV_PART_MAIN);
	lv_image_set_inner_align(ui->emotion_gif, LV_IMAGE_ALIGN_CONTAIN);
	lv_obj_align(ui->emotion_gif, LV_ALIGN_CENTER, 0, 0);
	lv_obj_add_flag(ui->emotion_gif, LV_OBJ_FLAG_HIDDEN);
#endif

	ui->emotion_label = lv_label_create(emotion_panel);
	lv_obj_set_width(ui->emotion_label, lv_pct(100));
	lv_label_set_long_mode(ui->emotion_label, LV_LABEL_LONG_WRAP);
	lv_obj_set_style_text_align(ui->emotion_label, LV_TEXT_ALIGN_CENTER, 0);
	lv_obj_set_style_text_color(ui->emotion_label, lv_color_hex(0xa5b4fc), 0);
	lv_obj_set_style_text_opa(ui->emotion_label, LV_OPA_COVER, LV_PART_MAIN);
	lv_label_set_text(ui->emotion_label, "");
	apply_font(ui, ui->emotion_label);

	ui->log_panel = lv_obj_create(content);
	lv_obj_set_height(ui->log_panel, lv_pct(100));
	lv_obj_set_width(ui->log_panel, 0);
	lv_obj_set_style_pad_all(ui->log_panel, 8, 0);
	lv_obj_set_style_radius(ui->log_panel, 6, 0);
	lv_obj_set_style_bg_color(ui->log_panel, lv_color_hex(0x101820), 0);
	lv_obj_set_style_border_width(ui->log_panel, 0, 0);
	lv_obj_set_flex_grow(ui->log_panel, 1);
	lv_obj_set_scroll_dir(ui->log_panel, LV_DIR_VER);
	lv_obj_set_scrollbar_mode(ui->log_panel, LV_SCROLLBAR_MODE_AUTO);

	ui->message_label = lv_label_create(ui->log_panel);
	lv_obj_set_width(ui->message_label, lv_pct(100));
	lv_label_set_long_mode(ui->message_label, LV_LABEL_LONG_WRAP);
	lv_obj_set_style_text_align(ui->message_label, LV_TEXT_ALIGN_LEFT, 0);
	lv_obj_set_style_text_color(ui->message_label, lv_color_hex(0xe2e8f0), 0);
	lv_obj_set_style_text_opa(ui->message_label, LV_OPA_COVER, LV_PART_MAIN);
	lv_label_set_text(ui->message_label, "Say something to XiaoZhi");
	apply_font(ui, ui->message_label);

	controls = lv_obj_create(screen);
	lv_obj_remove_flag(controls, LV_OBJ_FLAG_SCROLLABLE);
	lv_obj_set_width(controls, lv_pct(100));
	lv_obj_set_height(controls, LV_SIZE_CONTENT);
	lv_obj_set_style_pad_all(controls, 8, 0);
	lv_obj_set_style_pad_column(controls, 10, 0);
	lv_obj_set_style_radius(controls, 8, 0);
	lv_obj_set_style_bg_color(controls, lv_color_hex(0x1d2b36), 0);
	lv_obj_set_style_border_width(controls, 0, 0);
	lv_obj_set_flex_flow(controls, LV_FLEX_FLOW_ROW);
	lv_obj_set_flex_align(controls, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER,
				      LV_FLEX_ALIGN_CENTER);

	volume_title = lv_label_create(controls);
	lv_label_set_text(volume_title, "Volume");
	lv_obj_set_width(volume_title, 72);
	lv_label_set_long_mode(volume_title, LV_LABEL_LONG_MODE_CLIP);
	lv_obj_set_style_text_color(volume_title, lv_color_hex(0xcbd5e1), 0);
	apply_font(ui, volume_title);

	ui->volume_slider = lv_slider_create(controls);
	lv_obj_set_width(ui->volume_slider, 0);
	lv_obj_set_flex_grow(ui->volume_slider, 1);
	lv_slider_set_range(ui->volume_slider, 0, 100);
	lv_slider_set_value(ui->volume_slider, 70, LV_ANIM_OFF);
	lv_obj_add_event_cb(ui->volume_slider, volume_slider_event,
				     LV_EVENT_VALUE_CHANGED, ui);

	ui->volume_label = lv_label_create(controls);
	lv_label_set_text(ui->volume_label, "70%");
	lv_obj_set_width(ui->volume_label, 48);
	lv_label_set_long_mode(ui->volume_label, LV_LABEL_LONG_MODE_CLIP);
	lv_obj_set_style_text_align(ui->volume_label, LV_TEXT_ALIGN_CENTER, 0);
	lv_obj_set_style_text_color(ui->volume_label, lv_color_hex(0x7dd3fc), 0);
	apply_font(ui, ui->volume_label);

	ui->audio_label = lv_label_create(controls);
	lv_label_set_text(ui->audio_label, "Audio: waiting");
	lv_obj_set_width(ui->audio_label, 220);
	lv_label_set_long_mode(ui->audio_label, LV_LABEL_LONG_MODE_CLIP);
	lv_obj_set_style_text_color(ui->audio_label, lv_color_hex(0x94a3b8), 0);
	apply_font(ui, ui->audio_label);

	footer = lv_obj_create(screen);
	lv_obj_remove_flag(footer, LV_OBJ_FLAG_SCROLLABLE);
	lv_obj_set_width(footer, lv_pct(100));
	lv_obj_set_height(footer, LV_SIZE_CONTENT);
	lv_obj_set_style_pad_all(footer, 2, 0);
	lv_obj_set_style_pad_column(footer, 10, 0);
	lv_obj_set_flex_flow(footer, LV_FLEX_FLOW_ROW);
	lv_obj_set_flex_align(footer, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER,
				      LV_FLEX_ALIGN_CENTER);

	ui->detail_label = lv_label_create(footer);
	lv_obj_set_width(ui->detail_label, 0);
	lv_obj_set_flex_grow(ui->detail_label, 1);
	lv_label_set_long_mode(ui->detail_label, LV_LABEL_LONG_WRAP);
	lv_label_set_text(ui->detail_label, "Starting...");
	lv_obj_set_style_text_color(ui->detail_label, lv_color_hex(0x94a3b8), 0);
	apply_font(ui, ui->detail_label);

	ui->session_label = lv_label_create(footer);
	lv_label_set_text(ui->session_label, "Session: -");
	lv_obj_set_style_text_color(ui->session_label, lv_color_hex(0x64748b), 0);
	apply_font(ui, ui->session_label);

	lv_screen_load(screen);
}

static const char *emotion_asset_base(const char *emotion)
{
	if (emotion && (!strcmp(emotion, "thinking") ||
			       !strcmp(emotion, "confused")))
		return "think";
	if (emotion && (!strcmp(emotion, "neutral") ||
			       !strcmp(emotion, "surprised") ||
			       !strcmp(emotion, "shocked") ||
			       !strcmp(emotion, "crying") ||
			       !strcmp(emotion, "angry") ||
			       !strcmp(emotion, "sad") ||
			       !strcmp(emotion, "embarrassed") ||
			       !strcmp(emotion, "silly")))
		return "naughty";

	/* The supplied CanMV resources use the joke face for positive moods. */
	return "joke";
}

static void sanitize_emotion(char *dst, size_t dst_size, const char *emotion)
{
	size_t i;
	size_t out = 0;

	if (!dst || !dst_size)
		return;
	for (i = 0; emotion && emotion[i] && out + 1 < dst_size; i++) {
		char c = emotion[i];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		    (c >= '0' && c <= '9') || c == '_' || c == '-')
			dst[out++] = c;
	}
	dst[out] = '\0';
}

static int try_asset_in_dir(const char *resource_dir, const char *name,
				    const char *extension, char *path,
				    size_t path_size)
{
	int written;

	written = snprintf(path, path_size, "%s/%s.%s", resource_dir, name,
			   extension);
	if (written < 0 || (size_t)written >= path_size)
		return 0;
	return access(path, R_OK) == 0;
}

static int try_asset(const char *resource_dir, const char *name,
			     const char *extension, char *path, size_t path_size)
{
	char gifs_dir[XIAOZHI_LVGL_PATH_SIZE];
	int written;

	if (try_asset_in_dir(resource_dir, name, extension, path, path_size))
		return 1;
	written = snprintf(gifs_dir, sizeof(gifs_dir), "%s/gifs", resource_dir);
	if (written < 0 || (size_t)written >= sizeof(gifs_dir))
		return 0;
	return try_asset_in_dir(gifs_dir, name, extension, path, path_size);
}

static int find_emotion_asset(const struct xiaozhi_lvgl *ui,
			      const char *emotion, char *path, size_t path_size,
			      int allow_gif, int *is_gif)
{
	char safe_emotion[XIAOZHI_LVGL_EMOTION_SIZE];
	char exact_name[XIAOZHI_LVGL_EMOTION_SIZE + 8];
	char emotion_name[XIAOZHI_LVGL_EMOTION_SIZE + 8];
	char fallback_name[32];
	const char *resource_dir;
	const char *base;
	const char *asset_emotion;
	int written;

	if (!ui || !path || !path_size || !is_gif)
		return 0;
	resource_dir = ui->config.resource_dir;
	if (!resource_dir || !resource_dir[0])
		resource_dir = XIAOZHI_DEFAULT_LVGL_RESOURCE_DIR;
	asset_emotion = emotion;
	if (emotion && !strcmp(emotion, "smile"))
		asset_emotion = "happy";
	sanitize_emotion(safe_emotion, sizeof(safe_emotion), asset_emotion);
	base = emotion_asset_base(emotion);
	written = snprintf(exact_name, sizeof(exact_name), "emoji_%s",
			   safe_emotion);
	if (written < 0 || (size_t)written >= sizeof(exact_name))
		exact_name[0] = '\0';
	written = snprintf(emotion_name, sizeof(emotion_name), "img_%s",
			   safe_emotion);
	if (written < 0 || (size_t)written >= sizeof(emotion_name))
		emotion_name[0] = '\0';
	written = snprintf(fallback_name, sizeof(fallback_name), "img_%s", base);
	if (written < 0 || (size_t)written >= sizeof(fallback_name))
		return 0;

	*is_gif = 0;
#if LV_USE_GIF
	if (allow_gif && safe_emotion[0] &&
	    try_asset(resource_dir, safe_emotion, "gif", path, path_size)) {
		*is_gif = 1;
		return 1;
	}
#endif
	if (safe_emotion[0] &&
	    try_asset(resource_dir, safe_emotion, "png", path, path_size))
		return 1;
#if LV_USE_GIF
	if (allow_gif && exact_name[0] &&
	    try_asset(resource_dir, exact_name, "gif", path, path_size)) {
		*is_gif = 1;
		return 1;
	}
#else
	(void)allow_gif;
#endif
	if (exact_name[0] &&
	    try_asset(resource_dir, exact_name, "png", path, path_size))
		return 1;
#if LV_USE_GIF
	if (allow_gif && emotion_name[0] &&
	    try_asset(resource_dir, emotion_name, "gif", path, path_size)) {
		*is_gif = 1;
		return 1;
	}
#endif
	if (emotion_name[0] &&
	    try_asset(resource_dir, emotion_name, "png", path, path_size))
		return 1;
#if LV_USE_GIF
	if (allow_gif &&
	    try_asset(resource_dir, fallback_name, "gif", path, path_size)) {
		*is_gif = 1;
		return 1;
	}
#endif
	return try_asset(resource_dir, fallback_name, "png", path, path_size);
}

static int make_lvgl_path(char *dst, size_t dst_size, const char *path)
{
	int written;

	written = snprintf(dst, dst_size, "%s%s", XIAOZHI_LVGL_FS_PREFIX,
			   path ? path : "");
	return written < 0 || (size_t)written >= dst_size ? -1 : 0;
}

static void hide_emotion_assets(struct xiaozhi_lvgl *ui)
{
	lv_obj_add_flag(ui->emotion_image, LV_OBJ_FLAG_HIDDEN);
#if LV_USE_GIF
	lv_obj_add_flag(ui->emotion_gif, LV_OBJ_FLAG_HIDDEN);
#endif
}

static void show_emotion_asset(struct xiaozhi_lvgl *ui, const char *emotion)
{
	char raw_path[XIAOZHI_LVGL_PATH_SIZE];
	int is_gif;
	int found;

	hide_emotion_assets(ui);
	if (!emotion || !emotion[0])
		return;
	found = find_emotion_asset(ui, emotion, raw_path, sizeof(raw_path), 1,
					   &is_gif);
	if (!found) {
		printf("xiaozhi: no LVGL emotion asset for %s\n", emotion);
		return;
	}

#if LV_USE_GIF
	if (is_gif) {
		if (make_lvgl_path(ui->emotion_gif_path,
				   sizeof(ui->emotion_gif_path), raw_path) == 0) {
			lv_gif_set_src(ui->emotion_gif, ui->emotion_gif_path);
			if (lv_gif_is_loaded(ui->emotion_gif)) {
				lv_gif_restart(ui->emotion_gif);
				lv_obj_remove_flag(ui->emotion_gif, LV_OBJ_FLAG_HIDDEN);
				printf("xiaozhi: LVGL emotion GIF: %s\n", raw_path);
				return;
			}
		}
		/* A bad or unsupported GIF should fall back to the PNG resource. */
		found = find_emotion_asset(ui, emotion, raw_path, sizeof(raw_path),
					   0, &is_gif);
		if (!found)
			return;
	}
#else
	if (is_gif)
		return;
#endif

	if (make_lvgl_path(ui->emotion_image_path,
			   sizeof(ui->emotion_image_path), raw_path))
		return;
	lv_image_set_src(ui->emotion_image, ui->emotion_image_path);
	lv_obj_remove_flag(ui->emotion_image, LV_OBJ_FLAG_HIDDEN);
	printf("xiaozhi: LVGL emotion image: %s\n", raw_path);
}

static void load_text_font(struct xiaozhi_lvgl *ui)
{
#if LV_USE_FREETYPE
	char path[XIAOZHI_LVGL_PATH_SIZE];
	const char *resource_dir = ui->config.resource_dir;
	int written;

	if (!resource_dir || !resource_dir[0])
		resource_dir = XIAOZHI_DEFAULT_LVGL_RESOURCE_DIR;
	written = snprintf(path, sizeof(path), "%s/font/%s", resource_dir,
			   XIAOZHI_DEFAULT_LVGL_FONT_FILE);
	if (written < 0 || (size_t)written >= sizeof(path) ||
	    access(path, R_OK) != 0)
		path[0] = '\0';
	if (!path[0]) {
		written = snprintf(path, sizeof(path), "%s/%s", resource_dir,
				   XIAOZHI_DEFAULT_LVGL_FONT_FILE);
		if (written < 0 || (size_t)written >= sizeof(path) ||
		    access(path, R_OK) != 0)
			path[0] = '\0';
	}
	if (!path[0]) {
		printf("xiaozhi: LVGL Chinese font not found; using built-in font\n");
		return;
	}
	ui->text_font = lv_freetype_font_create(
		path, LV_FREETYPE_FONT_RENDER_MODE_BITMAP, XIAOZHI_LVGL_FONT_SIZE,
		LV_FREETYPE_FONT_STYLE_NORMAL);
	if (ui->text_font)
		printf("xiaozhi: LVGL FreeType font loaded: %s\n", path);
	else
		printf("xiaozhi: LVGL FreeType font load failed; using built-in font\n");
#else
	(void)ui;
	printf("xiaozhi: LVGL FreeType support unavailable; using built-in font\n");
#endif
}

static void release_text_font(struct xiaozhi_lvgl *ui)
{
#if LV_USE_FREETYPE
	if (ui->text_font) {
		lv_freetype_font_delete(ui->text_font);
		ui->text_font = NULL;
	}
#else
	(void)ui;
#endif
}

static void update_ui(struct xiaozhi_lvgl *ui,
			      const struct xiaozhi_lvgl_snapshot *state)
{
	char text[XIAOZHI_MAX_ACTIVATION_CODE +
		  XIAOZHI_MAX_ACTIVATION_MESSAGE + 64];
	lv_color_t connection_color;
	int log_changed = strcmp(state->log, ui->rendered.log) != 0;

	if (state->log[0])
		lv_label_set_text(ui->message_label, state->log);
	else if (!strcmp(state->connection, "ready") && state->detail[0])
		lv_label_set_text(ui->message_label, state->detail);
	else
		lv_label_set_text(ui->message_label, "Say something to XiaoZhi");
	if (log_changed) {
		lv_obj_update_layout(ui->log_panel);
		lv_obj_scroll_to_y(ui->log_panel, LV_COORD_MAX, LV_ANIM_OFF);
	}

	if (state->activation[0]) {
		if (state->activation_message[0])
			snprintf(text, sizeof(text), "Activation code: %s\n%s",
				 state->activation, state->activation_message);
		else
			snprintf(text, sizeof(text), "Activation code: %s",
				 state->activation);
	} else if (state->activation_complete) {
		snprintf(text, sizeof(text), "Device activation complete");
	} else if (state->activation_message[0]) {
		snprintf(text, sizeof(text), "Activation: %s",
			 state->activation_message);
	} else {
		snprintf(text, sizeof(text), "Waiting for device activation");
	}
	lv_label_set_text(ui->activation_label, text);

	lv_label_set_text(ui->connection_label,
				 state->connection[0] ? state->connection : "starting");
	lv_label_set_text(ui->detail_label, state->detail[0] ? state->detail : "");
	if (state->session[0])
		snprintf(text, sizeof(text), "Session: %s", state->session);
	else
		snprintf(text, sizeof(text), "Session: -");
	lv_label_set_text(ui->session_label, text);

	if (state->emotion[0])
		snprintf(text, sizeof(text), "Emotion: %s", state->emotion);
	else
		text[0] = '\0';
	lv_label_set_text(ui->emotion_label, text);
	if (strcmp(state->emotion, ui->rendered.emotion) ||
	    strcmp(state->emoji, ui->rendered.emoji))
		show_emotion_asset(ui, state->emotion);

	snprintf(text, sizeof(text), "%d%%", state->volume);
	lv_label_set_text(ui->volume_label, text);
	snprintf(text, sizeof(text), "Audio: %s%s",
		 state->audio_available ? "ready" : "unavailable",
		 state->speaking ? " / TTS" : "");
	lv_label_set_text(ui->audio_label, text);

	ui->applying_state = 1;
	lv_slider_set_value(ui->volume_slider, state->volume, LV_ANIM_OFF);
	if (state->audio_available && ui->config.set_volume)
		lv_obj_remove_state(ui->volume_slider, LV_STATE_DISABLED);
	else
		lv_obj_add_state(ui->volume_slider, LV_STATE_DISABLED);
	ui->applying_state = 0;

	if (!strcmp(state->connection, "ready"))
		connection_color = lv_color_hex(0x4ade80);
	else if (!strcmp(state->connection, "error"))
		connection_color = lv_color_hex(0xfb7185);
	else
		connection_color = lv_color_hex(0xfbbf24);
	lv_obj_set_style_text_color(ui->connection_label, connection_color, 0);
}

static void *lvgl_thread_main(void *opaque)
{
	struct xiaozhi_lvgl *ui = opaque;

	for (;;) {
		struct xiaozhi_lvgl_snapshot snapshot;
		uint32_t delay_ms;
		int running;

		pthread_mutex_lock(&ui->lock);
		running = ui->running;
		snapshot = ui->state;
		pthread_mutex_unlock(&ui->lock);
		if (!running)
			break;
		if (memcmp(&snapshot, &ui->rendered, sizeof(snapshot))) {
			update_ui(ui, &snapshot);
			ui->rendered = snapshot;
		}
		delay_ms = lv_timer_handler();
		if (delay_ms < 10)
			delay_ms = 10;
		if (delay_ms > 50)
			delay_ms = 50;
		usleep(delay_ms * 1000);
	}
	return NULL;
}

void xiaozhi_lvgl_config_init(struct xiaozhi_lvgl_config *config)
{
	if (!config)
		return;
	memset(config, 0, sizeof(*config));
	config->enabled = 1;
	config->connector = XIAOZHI_LVGL_DEFAULT_CONNECTOR;
	config->layer = XIAOZHI_LVGL_DEFAULT_LAYER;
	config->touch_id = XIAOZHI_LVGL_DEFAULT_TOUCH_ID;
	config->resource_dir = XIAOZHI_DEFAULT_LVGL_RESOURCE_DIR;
}

struct xiaozhi_lvgl *xiaozhi_lvgl_create(
	const struct xiaozhi_lvgl_config *config)
{
	struct xiaozhi_lvgl *ui;

	if (!config || !config->enabled)
		return NULL;
	ui = calloc(1, sizeof(*ui));
	if (!ui)
		return NULL;
	ui->config = *config;
	if (pthread_mutex_init(&ui->lock, NULL)) {
		free(ui);
		return NULL;
	}
	copy_text(ui->state.connection, sizeof(ui->state.connection), "starting");
	copy_text(ui->state.detail, sizeof(ui->state.detail), "Starting LVGL");
	ui->state.volume = 70;
	return ui;
}

int xiaozhi_lvgl_start(struct xiaozhi_lvgl *ui)
{
	if (!ui || ui->thread_started)
		return ui && ui->thread_started ? 0 : -1;
	if (kd_display_init((k_connector_type)ui->config.connector)) {
		printf("xiaozhi: LVGL display initialization failed\n");
		return -1;
	}
	lv_init();
	load_text_font(ui);
	ui->display = lv_k230_display_create((k_vo_layer_id)ui->config.layer,
					     255);
	if (!ui->display) {
		printf("xiaozhi: LVGL display creation failed\n");
		release_text_font(ui);
		lv_deinit();
		kd_display_deinit();
		return -1;
	}
	/* Rotate the 480x800 panel into a 800x480 landscape workspace. */
	lv_display_set_rotation(ui->display, XIAOZHI_LVGL_DEFAULT_ROTATION);
	ui->touch = lv_k230_touch_init(ui->config.touch_id);
	if (!ui->touch)
		printf("xiaozhi: LVGL touch initialization failed; continuing without touch\n");
	create_ui(ui);
	pthread_mutex_lock(&ui->lock);
	ui->running = 1;
	pthread_mutex_unlock(&ui->lock);
	if (pthread_create(&ui->thread, NULL, lvgl_thread_main, ui)) {
		pthread_mutex_lock(&ui->lock);
		ui->running = 0;
		pthread_mutex_unlock(&ui->lock);
		release_text_font(ui);
		lv_deinit();
		ui->display = NULL;
		kd_display_deinit();
		return -1;
	}
	ui->thread_started = 1;
	printf("xiaozhi: LVGL UI ready\n");
	return 0;
}

void xiaozhi_lvgl_stop(struct xiaozhi_lvgl *ui)
{
	if (!ui)
		return;
	if (ui->thread_started) {
		pthread_mutex_lock(&ui->lock);
		ui->running = 0;
		pthread_mutex_unlock(&ui->lock);
		pthread_join(ui->thread, NULL);
		ui->thread_started = 0;
	}
	if (ui->display) {
		release_text_font(ui);
		lv_deinit();
		ui->display = NULL;
		ui->touch = NULL;
		kd_display_deinit();
	}
}

void xiaozhi_lvgl_destroy(struct xiaozhi_lvgl *ui)
{
	if (!ui)
		return;
	xiaozhi_lvgl_stop(ui);
	pthread_mutex_destroy(&ui->lock);
	free(ui);
}

void xiaozhi_lvgl_set_connection(struct xiaozhi_lvgl *ui,
					 const char *state, const char *detail)
{
	if (!ui)
		return;
	pthread_mutex_lock(&ui->lock);
	copy_text(ui->state.connection, sizeof(ui->state.connection), state);
	copy_text(ui->state.detail, sizeof(ui->state.detail), detail);
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_wake_prompt(struct xiaozhi_lvgl *ui,
					 const char *wake_word)
{
	if (!ui)
		return;
	pthread_mutex_lock(&ui->lock);
	ui->state.log[0] = '\0';
	ui->state.emotion[0] = '\0';
	ui->state.emoji[0] = '\0';
	snprintf(ui->state.detail, sizeof(ui->state.detail),
		 "Say %s to wake XiaoZhi", wake_word && wake_word[0] ?
		 wake_word : "the wake word");
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_session(struct xiaozhi_lvgl *ui, const char *session_id)
{
	if (!ui)
		return;
	pthread_mutex_lock(&ui->lock);
	copy_text(ui->state.session, sizeof(ui->state.session), session_id);
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_activation(struct xiaozhi_lvgl *ui,
					 const char *code, const char *message)
{
	if (!ui)
		return;
	pthread_mutex_lock(&ui->lock);
	copy_text(ui->state.activation, sizeof(ui->state.activation), code);
	copy_text(ui->state.activation_message,
			  sizeof(ui->state.activation_message), message);
	ui->state.activation_complete = !code || !code[0];
	if (message && message[0])
		ui->state.activation_complete = 0;
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_audio(struct xiaozhi_lvgl *ui, int available,
				    int volume)
{
	if (!ui)
		return;
	pthread_mutex_lock(&ui->lock);
	ui->state.audio_available = available != 0;
	ui->state.volume = volume;
	clamp_snapshot(&ui->state);
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_speaking(struct xiaozhi_lvgl *ui, int speaking)
{
	if (!ui)
		return;
	pthread_mutex_lock(&ui->lock);
	ui->state.speaking = speaking != 0;
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_user_text(struct xiaozhi_lvgl *ui, const char *text)
{
	if (!ui || !text || !text[0])
		return;
	pthread_mutex_lock(&ui->lock);
	append_log(ui->state.log, sizeof(ui->state.log), "You: ", text);
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_assistant_text(struct xiaozhi_lvgl *ui,
					     const char *text)
{
	if (!ui || !text || !text[0])
		return;
	pthread_mutex_lock(&ui->lock);
	append_log(ui->state.log, sizeof(ui->state.log), "XiaoZhi: ", text);
	pthread_mutex_unlock(&ui->lock);
}

void xiaozhi_lvgl_set_emotion(struct xiaozhi_lvgl *ui, const char *emotion,
				      const char *emoji)
{
	int changed;

	if (!ui)
		return;
	pthread_mutex_lock(&ui->lock);
	changed = strcmp(ui->state.emotion, emotion ? emotion : "") != 0;
	copy_text(ui->state.emotion, sizeof(ui->state.emotion), emotion);
	copy_text(ui->state.emoji, sizeof(ui->state.emoji), emoji);
	if (changed && ui->state.emotion[0])
		append_log(ui->state.log, sizeof(ui->state.log), "Emotion: ",
				   ui->state.emotion);
	pthread_mutex_unlock(&ui->lock);
}

#else

struct xiaozhi_lvgl {
	int unused;
};

void xiaozhi_lvgl_config_init(struct xiaozhi_lvgl_config *config)
{
	if (config)
		memset(config, 0, sizeof(*config));
}

struct xiaozhi_lvgl *xiaozhi_lvgl_create(
	const struct xiaozhi_lvgl_config *config)
{
	(void)config;
	return NULL;
}

int xiaozhi_lvgl_start(struct xiaozhi_lvgl *ui)
{
	(void)ui;
	return -1;
}

void xiaozhi_lvgl_stop(struct xiaozhi_lvgl *ui)
{
	(void)ui;
}

void xiaozhi_lvgl_destroy(struct xiaozhi_lvgl *ui)
{
	(void)ui;
}

void xiaozhi_lvgl_set_connection(struct xiaozhi_lvgl *ui,
					 const char *state, const char *detail)
{
	(void)ui;
	(void)state;
	(void)detail;
}

void xiaozhi_lvgl_set_wake_prompt(struct xiaozhi_lvgl *ui,
					 const char *wake_word)
{
	(void)ui;
	(void)wake_word;
}

void xiaozhi_lvgl_set_session(struct xiaozhi_lvgl *ui, const char *session_id)
{
	(void)ui;
	(void)session_id;
}

void xiaozhi_lvgl_set_activation(struct xiaozhi_lvgl *ui,
					 const char *code, const char *message)
{
	(void)ui;
	(void)code;
	(void)message;
}

void xiaozhi_lvgl_set_audio(struct xiaozhi_lvgl *ui, int available,
				    int volume)
{
	(void)ui;
	(void)available;
	(void)volume;
}

void xiaozhi_lvgl_set_speaking(struct xiaozhi_lvgl *ui, int speaking)
{
	(void)ui;
	(void)speaking;
}

void xiaozhi_lvgl_set_user_text(struct xiaozhi_lvgl *ui, const char *text)
{
	(void)ui;
	(void)text;
}

void xiaozhi_lvgl_set_assistant_text(struct xiaozhi_lvgl *ui,
					     const char *text)
{
	(void)ui;
	(void)text;
}

void xiaozhi_lvgl_set_emotion(struct xiaozhi_lvgl *ui, const char *emotion,
				      const char *emoji)
{
	(void)ui;
	(void)emotion;
	(void)emoji;
}

#endif
