#ifndef XIAOZHI_AUDIO_H
#define XIAOZHI_AUDIO_H

#include <stddef.h>
#include <stdint.h>

#include "xiaozhi_config.h"

struct xiaozhi_audio;

struct xiaozhi_audio_config {
	int enabled;
	int input_device;
	int input_channel;
	int output_device;
	int output_channel;
	int internal_codec;
	int sample_rate;
	int channels;
	int frame_samples;
	int bitrate;
	int audio3a_mask;
	int input_sample_rate;
	int input_frame_samples;
	int decode_sample_rate;
	int decode_channels;
	int decode_frame_samples;
	int output_sample_rate;
	int output_channels;
	int output_frame_samples;
};

typedef int (*xiaozhi_audio_send_fn)(void *opaque, const void *data,
					     size_t len);

typedef int (*xiaozhi_audio_capture_fn)(void *opaque,
						const int16_t *samples,
						size_t sample_count);

struct xiaozhi_audio *xiaozhi_audio_create(
	const struct xiaozhi_audio_config *config,
	xiaozhi_audio_send_fn send, void *send_opaque);

void xiaozhi_audio_destroy(struct xiaozhi_audio *audio);

int xiaozhi_audio_initialize(struct xiaozhi_audio *audio);
void xiaozhi_audio_deinitialize(struct xiaozhi_audio *audio);

int xiaozhi_audio_set_output_format(struct xiaozhi_audio *audio,
					    int sample_rate, int channels,
					    int frame_duration);
int xiaozhi_audio_set_decode_format(struct xiaozhi_audio *audio,
					    int sample_rate, int channels,
					    int frame_duration);

int xiaozhi_audio_start_capture(struct xiaozhi_audio *audio);
void xiaozhi_audio_stop_capture(struct xiaozhi_audio *audio);

int xiaozhi_audio_set_capture_callback(struct xiaozhi_audio *audio,
					       xiaozhi_audio_capture_fn callback,
					       void *opaque);

int xiaozhi_audio_queue_opus(struct xiaozhi_audio *audio,
				     const void *data, size_t len);
void xiaozhi_audio_clear_playback(struct xiaozhi_audio *audio);

int xiaozhi_audio_is_ready(const struct xiaozhi_audio *audio);
int xiaozhi_audio_set_volume(struct xiaozhi_audio *audio, int volume);
int xiaozhi_audio_get_volume(struct xiaozhi_audio *audio, int *volume);

#endif
