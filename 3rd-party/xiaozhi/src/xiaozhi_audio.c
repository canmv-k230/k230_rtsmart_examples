#include "xiaozhi_audio.h"

#include <k_acodec_comm.h>
#include <k_audio_comm.h>
#include <k_adec_comm.h>
#include <k_aenc_comm.h>
#include <k_payload_comm.h>
#include <k_vb_comm.h>
#include <mpi_adec_api.h>
#include <mpi_aenc_api.h>
#include <mpi_ai_api.h>
#include <mpi_ao_api.h>
#include <mpi_sys_api.h>
#include <mpi_vb_api.h>

#include <fcntl.h>
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

typedef struct SpeexResamplerState_ SpeexResamplerState;

extern SpeexResamplerState *speex_resampler_init(
	unsigned int nb_channels, unsigned int in_rate, unsigned int out_rate,
	int quality, int *err);
extern void speex_resampler_destroy(SpeexResamplerState *st);
extern int speex_resampler_process_int(
	SpeexResamplerState *st, unsigned int channel_index,
	const short *in, unsigned int *in_len, short *out, unsigned int *out_len);
extern int speex_resampler_reset_mem(SpeexResamplerState *st);

struct xiaozhi_audio_packet {
	size_t len;
	unsigned char data[XIAOZHI_MAX_OPUS_PACKET];
};

struct xiaozhi_audio {
	struct xiaozhi_audio_config config;
	xiaozhi_audio_send_fn send;
	void *send_opaque;
	xiaozhi_audio_capture_fn capture_callback;
	void *capture_opaque;

	pthread_mutex_t lock;
	pthread_cond_t playback_cond;
	struct xiaozhi_audio_packet playback_queue[XIAOZHI_AUDIO_QUEUE_DEPTH];
	int playback_head;
	int playback_tail;
	int playback_count;

	pthread_t capture_thread;
	pthread_t playback_thread;
	int capture_thread_started;
	int playback_thread_started;
	volatile int capture_running;
	volatile int playback_running;

	int initialized;
	int ai_enabled;
	int ai_chn_enabled;
	int ao_enabled;
	int ao_chn_enabled;
	int aenc_created;
	int adec_created;
	k_aenc_chn aenc_chn;
	k_adec_chn adec_chn;

	k_s32 stream_pool_id;
	k_vb_blk_handle stream_handle;
	void *stream_virt;
	k_u64 stream_phys;

	SpeexResamplerState *playback_resampler;
	pthread_mutex_t playback_resampler_lock;
	k_s32 playback_pool_id;
	k_u32 playback_buffer_size;
	k_u32 playback_capacity_samples;

	int volume;
	int codec_fd;
};

static void init_audio_attr(struct xiaozhi_audio *audio,
				    k_aio_dev_attr *attr, int input)
{
	memset(attr, 0, sizeof(*attr));
	attr->audio_type = input ?
		(audio->config.input_device == 1 ? KD_AUDIO_INPUT_TYPE_PDM :
		 KD_AUDIO_INPUT_TYPE_I2S) : KD_AUDIO_OUTPUT_TYPE_I2S;

	if (input && audio->config.input_device == 1) {
		attr->kd_audio_attr.pdm_attr.sample_rate =
			audio->config.sample_rate;
		attr->kd_audio_attr.pdm_attr.bit_width = KD_AUDIO_BIT_WIDTH_16;
		attr->kd_audio_attr.pdm_attr.chn_cnt = 1;
		attr->kd_audio_attr.pdm_attr.snd_mode = KD_AUDIO_SOUND_MODE_MONO;
		attr->kd_audio_attr.pdm_attr.frame_num = 5;
		attr->kd_audio_attr.pdm_attr.point_num_per_frame =
			audio->config.frame_samples;
		attr->kd_audio_attr.pdm_attr.pdm_oversample =
			KD_AUDIO_PDM_INPUT_OVERSAMPLE_64;
		return;
	}

	attr->kd_audio_attr.i2s_attr.sample_rate = input ?
		audio->config.sample_rate : audio->config.output_sample_rate;
	attr->kd_audio_attr.i2s_attr.bit_width = KD_AUDIO_BIT_WIDTH_16;
	attr->kd_audio_attr.i2s_attr.chn_cnt = 2;
	attr->kd_audio_attr.i2s_attr.snd_mode = KD_AUDIO_SOUND_MODE_MONO;
	attr->kd_audio_attr.i2s_attr.mono_channel =
		KD_I2S_IN_MONO_RIGHT_CHANNEL;
	attr->kd_audio_attr.i2s_attr.i2s_mode = K_STANDARD_MODE;
	attr->kd_audio_attr.i2s_attr.frame_num = 5;
	attr->kd_audio_attr.i2s_attr.point_num_per_frame = input ?
		audio->config.frame_samples : audio->config.output_frame_samples;
	attr->kd_audio_attr.i2s_attr.i2s_type =
		audio->config.internal_codec ? K_AIO_I2STYPE_INNERCODEC :
		K_AIO_I2STYPE_EXTERN;
}

static void close_codec(struct xiaozhi_audio *audio)
{
	if (audio->codec_fd >= 0) {
		close(audio->codec_fd);
		audio->codec_fd = -1;
	}
}

static int open_codec(struct xiaozhi_audio *audio)
{
	if (!audio->config.internal_codec)
		return -1;
	if (audio->codec_fd >= 0)
		return 0;
	audio->codec_fd = open("/dev/acodec_device", O_RDWR);
	return audio->codec_fd >= 0 ? 0 : -1;
}

static float volume_to_db(int volume)
{
	/* Match the CanMV AO volume API: 0..100 maps to -43..+7 dB. */
	if (volume <= 0)
		return -43.0f;
	if (volume >= 100)
		return 7.0f;
	return -43.0f + 0.5f * (float)volume;
}

static int db_to_volume(float db)
{
	int volume;

	if (db <= -43.0f)
		return 0;
	volume = (int)((db + 43.0f) * 2.0f + 0.5f);
	if (volume < 0)
		return 0;
	if (volume > 100)
		return 100;
	return volume;
}

static int set_codec_volume(struct xiaozhi_audio *audio, int volume)
{
	float db;
	int ret;

	if (open_codec(audio))
		return -1;
	db = volume_to_db(volume);
	ret = ioctl(audio->codec_fd, k_acodec_set_dacl_volume, &db);
	if (!ret)
		ret = ioctl(audio->codec_fd, k_acodec_set_dacr_volume, &db);
	return ret;
}

static void release_stream_buffer(struct xiaozhi_audio *audio)
{
	if (audio->stream_virt) {
		kd_mpi_sys_munmap(audio->stream_virt, XIAOZHI_MAX_OPUS_PACKET);
		audio->stream_virt = NULL;
	}
	if (audio->stream_handle != VB_INVALID_HANDLE) {
		kd_mpi_vb_release_block(audio->stream_handle);
		audio->stream_handle = VB_INVALID_HANDLE;
	}
	if ((k_u32)audio->stream_pool_id != VB_INVALID_POOLID) {
		kd_mpi_vb_destory_pool((k_u32)audio->stream_pool_id);
		audio->stream_pool_id = VB_INVALID_POOLID;
	}
	audio->stream_phys = 0;
}

static int create_stream_buffer(struct xiaozhi_audio *audio)
{
	k_vb_pool_config pool_config;

	memset(&pool_config, 0, sizeof(pool_config));
	pool_config.blk_cnt = 1;
	pool_config.blk_size = XIAOZHI_MAX_OPUS_PACKET;
	pool_config.mode = VB_REMAP_MODE_NOCACHE;
	audio->stream_pool_id = kd_mpi_vb_create_pool(&pool_config);
	if ((k_u32)audio->stream_pool_id == VB_INVALID_POOLID) {
		printf("xiaozhi: kd_mpi_vb_create_pool failed\n");
		return -1;
	}

	audio->stream_handle = kd_mpi_vb_get_block(
		(k_u32)audio->stream_pool_id, XIAOZHI_MAX_OPUS_PACKET, NULL);
	if (audio->stream_handle == VB_INVALID_HANDLE)
		goto failure;
	audio->stream_phys = kd_mpi_vb_handle_to_phyaddr(audio->stream_handle);
	audio->stream_virt = kd_mpi_sys_mmap(audio->stream_phys,
						 XIAOZHI_MAX_OPUS_PACKET);
	if (!audio->stream_virt)
		goto failure;
	return 0;

failure:
	printf("xiaozhi: failed to allocate Opus stream VB buffer\n");
	release_stream_buffer(audio);
	return -1;
}

static void release_playback_resources(struct xiaozhi_audio *audio)
{
	if (audio->playback_resampler) {
		speex_resampler_destroy(audio->playback_resampler);
		audio->playback_resampler = NULL;
	}
	if ((k_u32)audio->playback_pool_id != VB_INVALID_POOLID) {
		kd_mpi_vb_destory_pool((k_u32)audio->playback_pool_id);
		audio->playback_pool_id = VB_INVALID_POOLID;
	}
	audio->playback_buffer_size = 0;
	audio->playback_capacity_samples = 0;
}

static int create_playback_resources(struct xiaozhi_audio *audio)
{
	k_vb_pool_config pool_config;
	uint64_t scaled_samples;
	uint64_t capacity_samples;
	uint64_t buffer_size;
	int err;

	if (audio->config.decode_sample_rate ==
		audio->config.output_sample_rate)
		return 0;

	audio->playback_resampler = speex_resampler_init(
		audio->config.output_channels,
		audio->config.decode_sample_rate,
		audio->config.output_sample_rate, 5, &err);
	if (!audio->playback_resampler) {
		printf("xiaozhi: Speex resampler init failed ret=%d\n", err);
		return -1;
	}

	/* Leave room for the resampler filter delay on every decoded frame. */
	scaled_samples = ((uint64_t)audio->config.decode_frame_samples *
			  (uint64_t)audio->config.output_sample_rate +
			  (uint64_t)audio->config.decode_sample_rate - 1) /
			 (uint64_t)audio->config.decode_sample_rate;
	capacity_samples = scaled_samples + 256;
	if (capacity_samples < (uint64_t)audio->config.output_frame_samples + 256)
		capacity_samples = (uint64_t)audio->config.output_frame_samples + 256;
	if (capacity_samples > UINT32_MAX / sizeof(int16_t))
		goto failure;

	buffer_size = capacity_samples * sizeof(int16_t);
	buffer_size = (buffer_size + 4095) & ~((uint64_t)4095);
	if (buffer_size > UINT32_MAX)
		goto failure;

	memset(&pool_config, 0, sizeof(pool_config));
	pool_config.blk_cnt = XIAOZHI_AUDIO_QUEUE_DEPTH;
	pool_config.blk_size = buffer_size;
	pool_config.mode = VB_REMAP_MODE_NOCACHE;
	audio->playback_pool_id = kd_mpi_vb_create_pool(&pool_config);
	if ((k_u32)audio->playback_pool_id == VB_INVALID_POOLID) {
		printf("xiaozhi: playback resampler VB pool creation failed\n");
		goto failure;
	}
	audio->playback_buffer_size = (k_u32)buffer_size;
	audio->playback_capacity_samples =
		(k_u32)(buffer_size / sizeof(int16_t));
	return 0;

failure:
	release_playback_resources(audio);
	printf("xiaozhi: playback resampler buffer allocation failed\n");
	return -1;
}

static k_s32 create_aenc_channel(struct xiaozhi_audio *audio,
					 const k_aenc_chn_attr *attr)
{
	k_s32 ret = -1;
	int channel;

	for (channel = 0; channel < AENC_MAX_CHN_NUMS; channel++) {
		ret = kd_mpi_aenc_create_chn((k_aenc_chn)channel, attr);
		if (!ret) {
			audio->aenc_chn = (k_aenc_chn)channel;
			audio->aenc_created = 1;
			return 0;
		}
	}
	return ret;
}

static k_s32 create_adec_channel(struct xiaozhi_audio *audio,
					 const k_adec_chn_attr *attr)
{
	k_s32 ret = -1;
	int channel;

	for (channel = 0; channel < ADEC_MAX_CHN_NUMS; channel++) {
		ret = kd_mpi_adec_create_chn((k_adec_chn)channel, attr);
		if (!ret) {
			audio->adec_chn = (k_adec_chn)channel;
			audio->adec_created = 1;
			return 0;
		}
	}
	return ret;
}

static void disable_ai(struct xiaozhi_audio *audio)
{
	if (audio->ai_chn_enabled) {
		kd_mpi_ai_disable_chn((k_audio_dev)audio->config.input_device,
				       (k_ai_chn)audio->config.input_channel);
		audio->ai_chn_enabled = 0;
	}
	if (audio->ai_enabled)
		kd_mpi_ai_disable((k_audio_dev)audio->config.input_device);
	audio->ai_enabled = 0;
}

static void disable_ao(struct xiaozhi_audio *audio)
{
	if (audio->ao_chn_enabled) {
		kd_mpi_ao_disable_chn((k_audio_dev)audio->config.output_device,
				       (k_ao_chn)audio->config.output_channel);
		audio->ao_chn_enabled = 0;
	}
	if (audio->ao_enabled)
		kd_mpi_ao_disable((k_audio_dev)audio->config.output_device);
	audio->ao_enabled = 0;
}

static int configure_audio3a(struct xiaozhi_audio *audio)
{
	k_ai_vqe_enable vqe;
	k_s32 ret;

	if (!audio->config.audio3a_mask)
		return 0;

	memset(&vqe, 0, sizeof(vqe));
	vqe.ans_enable = (audio->config.audio3a_mask & XIAOZHI_AUDIO_3A_ANS) ?
		K_TRUE : K_FALSE;
	vqe.agc_enable = (audio->config.audio3a_mask & XIAOZHI_AUDIO_3A_AGC) ?
		K_TRUE : K_FALSE;
	vqe.aec_enable = (audio->config.audio3a_mask & XIAOZHI_AUDIO_3A_AEC) ?
		K_TRUE : K_FALSE;
	if (vqe.aec_enable)
		vqe.aec_echo_delay_ms = XIAOZHI_AUDIO_AEC_ECHO_DELAY_MS;
	ret = kd_mpi_ai_set_vqe_attr(
		(k_audio_dev)audio->config.input_device,
		(k_ai_chn)audio->config.input_channel, vqe);
	if (ret) {
		printf("xiaozhi: kd_mpi_ai_set_vqe_attr failed ret=%d\n", ret);
		return -1;
	}
	printf("xiaozhi: K230 audio 3A enabled: %s%s%s\n",
		vqe.aec_enable ? "AEC " : "",
		vqe.agc_enable ? "AGC " : "",
		vqe.ans_enable ? "ANS" : "");
	return 0;
}

static void *capture_thread_main(void *opaque)
{
	struct xiaozhi_audio *audio = (struct xiaozhi_audio *)opaque;
	k_audio_frame frame;
	k_audio_stream stream;
	void *capture_data;
	void *stream_data;
	int capture_mapped;

	while (audio->capture_running) {
		memset(&frame, 0, sizeof(frame));
		if (kd_mpi_ai_get_frame((k_audio_dev)audio->config.input_device,
					(k_ai_chn)audio->config.input_channel,
					&frame, 1000))
			continue;

		capture_data = NULL;
		capture_mapped = 0;
		if (audio->capture_callback && frame.phys_addr && frame.len) {
			capture_data = kd_mpi_sys_mmap(frame.phys_addr, frame.len);
			if (capture_data)
				capture_mapped = 1;
			if (capture_data)
				audio->capture_callback(audio->capture_opaque,
					(const int16_t *)capture_data,
					frame.len / sizeof(int16_t));
		}

		if (audio->capture_running &&
		    !kd_mpi_aenc_send_frame(audio->aenc_chn, &frame)) {
			memset(&stream, 0, sizeof(stream));
			if (!kd_mpi_aenc_get_stream(audio->aenc_chn, &stream, 1000)) {
				stream_data = kd_mpi_sys_mmap(stream.phys_addr, stream.len);
				if (stream_data) {
					if (audio->send)
						audio->send(audio->send_opaque, stream_data,
							   stream.len);
					kd_mpi_sys_munmap(stream_data, stream.len);
				}
				kd_mpi_aenc_release_stream(audio->aenc_chn, &stream);
			}
		}
		if (capture_mapped)
			kd_mpi_sys_munmap(capture_data, frame.len);
		kd_mpi_ai_release_frame((k_audio_dev)audio->config.input_device,
					(k_ai_chn)audio->config.input_channel, &frame);
	}

	return NULL;
}

static int get_playback_packet(struct xiaozhi_audio *audio,
				       struct xiaozhi_audio_packet *packet)
{
	pthread_mutex_lock(&audio->lock);
	while (!audio->playback_count && audio->playback_running)
		pthread_cond_wait(&audio->playback_cond, &audio->lock);
	if (!audio->playback_running) {
		pthread_mutex_unlock(&audio->lock);
		return -1;
	}
	*packet = audio->playback_queue[audio->playback_head];
	audio->playback_head =
		(audio->playback_head + 1) % XIAOZHI_AUDIO_QUEUE_DEPTH;
	audio->playback_count--;
	pthread_mutex_unlock(&audio->lock);
	return 0;
}

static int send_decoded_frame(struct xiaozhi_audio *audio,
				      const k_audio_frame *decoded)
{
	k_audio_frame output;
	k_vb_blk_handle output_handle;
	k_u64 output_phys;
	void *input_virt = NULL;
	void *output_virt = NULL;
	unsigned int input_samples;
	unsigned int output_samples;
	k_s32 ret;

	if (!audio->playback_resampler) {
		ret = kd_mpi_ao_send_frame(
			(k_audio_dev)audio->config.output_device,
			(k_ao_chn)audio->config.output_channel, decoded, 100);
		if (ret)
			printf("xiaozhi: AO send frame failed ret=%d\n", ret);
		return ret;
	}

	if (!decoded->len || decoded->len % sizeof(int16_t))
		return -1;
	input_samples = decoded->len / sizeof(int16_t);
	if (input_samples > (unsigned int)audio->config.decode_frame_samples)
		return -1;

	output_handle = kd_mpi_vb_get_block(
		(k_u32)audio->playback_pool_id, audio->playback_buffer_size, NULL);
	if (output_handle == VB_INVALID_HANDLE)
		return -1;
	output_phys = kd_mpi_vb_handle_to_phyaddr(output_handle);
	output_virt = kd_mpi_sys_mmap(output_phys, audio->playback_buffer_size);
	if (!output_virt)
		goto failure;

	input_virt = kd_mpi_sys_mmap(decoded->phys_addr, decoded->len);
	if (!input_virt)
		goto failure;

	output_samples = audio->playback_capacity_samples;
	pthread_mutex_lock(&audio->playback_resampler_lock);
	ret = speex_resampler_process_int(
		audio->playback_resampler, 0, (const short *)input_virt,
		&input_samples, (short *)output_virt, &output_samples);
	pthread_mutex_unlock(&audio->playback_resampler_lock);
	kd_mpi_sys_munmap(input_virt, decoded->len);
	input_virt = NULL;
	if (ret || input_samples != decoded->len / sizeof(int16_t) ||
		!output_samples)
		goto failure;

	memset(&output, 0, sizeof(output));
	output.bit_width = KD_AUDIO_BIT_WIDTH_16;
	output.snd_mode = KD_AUDIO_SOUND_MODE_MONO;
	output.virt_addr = output_virt;
	output.phys_addr = output_phys;
	output.time_stamp = decoded->time_stamp;
	output.seq = decoded->seq;
	output.len = output_samples * sizeof(int16_t);
	output.pool_id = (k_u32)kd_mpi_vb_handle_to_pool_id(output_handle);
	ret = kd_mpi_ao_send_frame(
		(k_audio_dev)audio->config.output_device,
		(k_ao_chn)audio->config.output_channel, &output, 100);
	if (ret)
		printf("xiaozhi: AO send resampled frame failed ret=%d\n", ret);
	kd_mpi_sys_munmap(output_virt, audio->playback_buffer_size);
	kd_mpi_vb_release_block(output_handle);
	return ret;

failure:
	if (input_virt)
		kd_mpi_sys_munmap(input_virt, decoded->len);
	if (output_virt)
		kd_mpi_sys_munmap(output_virt, audio->playback_buffer_size);
	kd_mpi_vb_release_block(output_handle);
	return -1;
}

static void *playback_thread_main(void *opaque)
{
	struct xiaozhi_audio *audio = (struct xiaozhi_audio *)opaque;
	struct xiaozhi_audio_packet packet;
	k_audio_stream stream;
	k_audio_frame frame;

	while (audio->playback_running) {
		if (get_playback_packet(audio, &packet))
			break;

		memcpy(audio->stream_virt, packet.data, packet.len);
		memset(&stream, 0, sizeof(stream));
		stream.stream = audio->stream_virt;
		stream.phys_addr = audio->stream_phys;
		stream.len = (k_u32)packet.len;
		if (kd_mpi_adec_send_stream(audio->adec_chn, &stream, K_TRUE))
			continue;

		memset(&frame, 0, sizeof(frame));
		if (!kd_mpi_adec_get_frame(audio->adec_chn, &frame, 1000)) {
			if (audio->playback_running)
				send_decoded_frame(audio, &frame);
			kd_mpi_adec_release_frame(audio->adec_chn, &frame);
		}
	}

	return NULL;
}

struct xiaozhi_audio *xiaozhi_audio_create(
	const struct xiaozhi_audio_config *config,
	xiaozhi_audio_send_fn send, void *send_opaque)
{
	struct xiaozhi_audio *audio;

	audio = (struct xiaozhi_audio *)calloc(1, sizeof(*audio));
	if (!audio)
		return NULL;
	if (config)
		audio->config = *config;
	if (audio->config.output_sample_rate <= 0)
		audio->config.output_sample_rate = XIAOZHI_AUDIO_SAMPLE_RATE;
	if (audio->config.output_channels <= 0)
		audio->config.output_channels = XIAOZHI_AUDIO_CHANNELS;
	if (audio->config.output_frame_samples <= 0)
		audio->config.output_frame_samples =
			XIAOZHI_AUDIO_FRAME_SAMPLES;
	if (audio->config.decode_sample_rate <= 0)
		audio->config.decode_sample_rate = audio->config.output_sample_rate;
	if (audio->config.decode_channels <= 0)
		audio->config.decode_channels = audio->config.output_channels;
	if (audio->config.decode_frame_samples <= 0)
		audio->config.decode_frame_samples =
			audio->config.output_frame_samples;
	audio->send = send;
	audio->send_opaque = send_opaque;
	audio->stream_pool_id = VB_INVALID_POOLID;
	audio->stream_handle = VB_INVALID_HANDLE;
	audio->playback_pool_id = VB_INVALID_POOLID;
	audio->codec_fd = -1;
	audio->volume = 70;
	if (pthread_mutex_init(&audio->lock, NULL)) {
		free(audio);
		return NULL;
	}
	if (pthread_cond_init(&audio->playback_cond, NULL)) {
		pthread_mutex_destroy(&audio->lock);
		free(audio);
		return NULL;
	}
	if (pthread_mutex_init(&audio->playback_resampler_lock, NULL)) {
		pthread_cond_destroy(&audio->playback_cond);
		pthread_mutex_destroy(&audio->lock);
		free(audio);
		return NULL;
	}
	return audio;
}

void xiaozhi_audio_destroy(struct xiaozhi_audio *audio)
{
	if (!audio)
		return;
	xiaozhi_audio_deinitialize(audio);
	pthread_mutex_destroy(&audio->playback_resampler_lock);
	pthread_cond_destroy(&audio->playback_cond);
	pthread_mutex_destroy(&audio->lock);
	free(audio);
}

int xiaozhi_audio_initialize(struct xiaozhi_audio *audio)
{
	k_aio_dev_attr ai_attr;
	k_aio_dev_attr ao_attr;
	k_aenc_chn_attr aenc_attr;
	k_adec_chn_attr adec_attr;
	k_s32 ret;

	if (!audio || !audio->config.enabled)
		return -1;
	if (audio->initialized)
		return 0;
	if (audio->config.sample_rate <= 0 || audio->config.channels <= 0 ||
	    audio->config.frame_samples <= 0 ||
	    audio->config.decode_sample_rate <= 0 ||
	    audio->config.decode_channels != 1 ||
	    audio->config.decode_frame_samples <= 0 ||
	    audio->config.output_sample_rate <= 0 ||
	    audio->config.output_channels != 1 ||
	    audio->config.output_frame_samples <= 0)
		return -1;

	memset(&aenc_attr, 0, sizeof(aenc_attr));
	aenc_attr.type = K_PT_OPUS;
	aenc_attr.buf_size = XIAOZHI_AUDIO_BUFFER_COUNT;
	aenc_attr.point_num_per_frame = audio->config.frame_samples;
	aenc_attr.sample_rate = audio->config.sample_rate;
	aenc_attr.channels = audio->config.channels;
	aenc_attr.bitrate = audio->config.bitrate;
	ret = create_aenc_channel(audio, &aenc_attr);
	if (ret) {
		printf("xiaozhi: no free Opus AENC channel ret=%d\n", ret);
		return -1;
	}

	memset(&adec_attr, 0, sizeof(adec_attr));
	adec_attr.type = K_PT_OPUS;
	adec_attr.mode = K_ADEC_MODE_PACK;
	adec_attr.sample_rate = audio->config.decode_sample_rate;
	adec_attr.channels = audio->config.decode_channels;
	adec_attr.point_num_per_frame = audio->config.decode_frame_samples;
	adec_attr.buf_size = XIAOZHI_AUDIO_BUFFER_COUNT;
	ret = create_adec_channel(audio, &adec_attr);
	if (ret) {
		printf("xiaozhi: no free Opus ADEC channel ret=%d\n", ret);
		goto failure;
	}

	init_audio_attr(audio, &ai_attr, 1);
	ret = kd_mpi_ai_set_pub_attr((k_audio_dev)audio->config.input_device,
				     &ai_attr);
	if (ret) {
		printf("xiaozhi: kd_mpi_ai_set_pub_attr failed ret=%d\n", ret);
		goto failure;
	}

	init_audio_attr(audio, &ao_attr, 0);
	ret = kd_mpi_ao_set_pub_attr((k_audio_dev)audio->config.output_device,
				     &ao_attr);
	if (ret) {
		printf("xiaozhi: kd_mpi_ao_set_pub_attr failed ret=%d\n", ret);
		goto failure;
	}
	ret = kd_mpi_ao_enable((k_audio_dev)audio->config.output_device);
	if (ret) {
		printf("xiaozhi: kd_mpi_ao_enable failed ret=%d\n", ret);
		goto failure;
	}
	audio->ao_enabled = 1;
	ret = kd_mpi_ao_enable_chn((k_audio_dev)audio->config.output_device,
				       (k_ao_chn)audio->config.output_channel);
	if (ret) {
		printf("xiaozhi: kd_mpi_ao_enable_chn failed ret=%d\n", ret);
		goto failure;
	}
	audio->ao_chn_enabled = 1;
	if (audio->config.internal_codec &&
	    set_codec_volume(audio, audio->volume))
		printf("xiaozhi: failed to initialize speaker volume\n");

	if (create_stream_buffer(audio))
		goto failure;
	if (create_playback_resources(audio))
		goto failure;
	audio->initialized = 1;
	return 0;

failure:
	disable_ao(audio);
	release_playback_resources(audio);
	release_stream_buffer(audio);
	if (audio->adec_created) {
		kd_mpi_adec_destroy_chn(audio->adec_chn);
		audio->adec_created = 0;
	}
	if (audio->aenc_created) {
		kd_mpi_aenc_destroy_chn(audio->aenc_chn);
		audio->aenc_created = 0;
	}
	return -1;
}

void xiaozhi_audio_deinitialize(struct xiaozhi_audio *audio)
{
	if (!audio)
		return;
	xiaozhi_audio_stop_capture(audio);
	pthread_mutex_lock(&audio->lock);
	audio->playback_running = 0;
	audio->playback_count = 0;
	audio->playback_head = 0;
	audio->playback_tail = 0;
	pthread_cond_broadcast(&audio->playback_cond);
	pthread_mutex_unlock(&audio->lock);
	if (audio->playback_thread_started) {
		pthread_join(audio->playback_thread, NULL);
		audio->playback_thread_started = 0;
	}

	disable_ai(audio);
	disable_ao(audio);
	release_playback_resources(audio);
	release_stream_buffer(audio);
	if (audio->adec_created) {
		kd_mpi_adec_destroy_chn(audio->adec_chn);
		audio->adec_created = 0;
	}
	if (audio->aenc_created) {
		kd_mpi_aenc_destroy_chn(audio->aenc_chn);
		audio->aenc_created = 0;
	}
	close_codec(audio);
	audio->initialized = 0;
}

int xiaozhi_audio_start_capture(struct xiaozhi_audio *audio)
{
	if (!audio || !audio->initialized || audio->capture_thread_started)
		return audio && audio->capture_thread_started ? 0 : -1;
	if (kd_mpi_ai_enable((k_audio_dev)audio->config.input_device))
		return -1;
	audio->ai_enabled = 1;
	if (kd_mpi_ai_enable_chn((k_audio_dev)audio->config.input_device,
				 (k_ai_chn)audio->config.input_channel)) {
		disable_ai(audio);
		return -1;
	}
	audio->ai_chn_enabled = 1;
	if (configure_audio3a(audio)) {
		disable_ai(audio);
		return -1;
	}
	audio->capture_running = 1;
	if (pthread_create(&audio->capture_thread, NULL, capture_thread_main,
			   audio)) {
		audio->capture_running = 0;
		disable_ai(audio);
		return -1;
	}
	audio->capture_thread_started = 1;
	return 0;
}

int xiaozhi_audio_set_capture_callback(struct xiaozhi_audio *audio,
					       xiaozhi_audio_capture_fn callback,
					       void *opaque)
{
	if (!audio || audio->capture_thread_started)
		return -1;
	audio->capture_callback = callback;
	audio->capture_opaque = opaque;
	return 0;
}

int xiaozhi_audio_set_output_format(struct xiaozhi_audio *audio,
					    int sample_rate, int channels,
					    int frame_duration)
{
	long long frame_samples;

	if (!audio || audio->initialized || sample_rate <= 0 || channels != 1 ||
	    frame_duration <= 0)
		return -1;
	frame_samples = (long long)sample_rate * frame_duration;
	if (frame_samples % 1000 || frame_samples / 1000 > 0x7fffffff)
		return -1;
	audio->config.output_sample_rate = sample_rate;
	audio->config.output_channels = channels;
	audio->config.output_frame_samples = (int)(frame_samples / 1000);
	return 0;
}

int xiaozhi_audio_set_decode_format(struct xiaozhi_audio *audio,
					    int sample_rate, int channels,
					    int frame_duration)
{
	long long frame_samples;

	if (!audio || audio->initialized || sample_rate <= 0 || channels != 1 ||
	    frame_duration <= 0)
		return -1;
	frame_samples = (long long)sample_rate * frame_duration;
	if (frame_samples % 1000 || frame_samples / 1000 > 0x7fffffff)
		return -1;
	audio->config.decode_sample_rate = sample_rate;
	audio->config.decode_channels = channels;
	audio->config.decode_frame_samples = (int)(frame_samples / 1000);
	return 0;
}

void xiaozhi_audio_stop_capture(struct xiaozhi_audio *audio)
{
	if (!audio || !audio->capture_thread_started)
		return;
	audio->capture_running = 0;
	pthread_join(audio->capture_thread, NULL);
	audio->capture_thread_started = 0;
	disable_ai(audio);
}

int xiaozhi_audio_queue_opus(struct xiaozhi_audio *audio,
				     const void *data, size_t len)
{
	int index;

	if (!audio || !audio->initialized || !data || !len ||
	    len > XIAOZHI_MAX_OPUS_PACKET)
		return -1;

	pthread_mutex_lock(&audio->lock);
	if (audio->playback_count >= XIAOZHI_AUDIO_QUEUE_DEPTH) {
		audio->playback_head =
			(audio->playback_head + 1) % XIAOZHI_AUDIO_QUEUE_DEPTH;
		audio->playback_count--;
	}
	index = audio->playback_tail;
	audio->playback_queue[index].len = len;
	memcpy(audio->playback_queue[index].data, data, len);
	audio->playback_tail =
		(audio->playback_tail + 1) % XIAOZHI_AUDIO_QUEUE_DEPTH;
	audio->playback_count++;
	if (!audio->playback_thread_started) {
		audio->playback_running = 1;
		if (pthread_create(&audio->playback_thread, NULL,
				   playback_thread_main, audio)) {
			audio->playback_running = 0;
			audio->playback_count--;
			audio->playback_tail = index;
			pthread_mutex_unlock(&audio->lock);
			return -1;
		}
		audio->playback_thread_started = 1;
	}
	pthread_cond_signal(&audio->playback_cond);
	pthread_mutex_unlock(&audio->lock);
	return 0;
}

void xiaozhi_audio_clear_playback(struct xiaozhi_audio *audio)
{
	if (!audio)
		return;
	pthread_mutex_lock(&audio->lock);
	audio->playback_count = 0;
	audio->playback_head = 0;
	audio->playback_tail = 0;
	pthread_mutex_unlock(&audio->lock);
	if (audio->playback_resampler) {
		pthread_mutex_lock(&audio->playback_resampler_lock);
		speex_resampler_reset_mem(audio->playback_resampler);
		pthread_mutex_unlock(&audio->playback_resampler_lock);
	}
	if (audio->adec_created)
		kd_mpi_adec_clr_chn_buf(audio->adec_chn);
}

int xiaozhi_audio_is_ready(const struct xiaozhi_audio *audio)
{
	return audio && audio->initialized;
}

int xiaozhi_audio_set_volume(struct xiaozhi_audio *audio, int volume)
{
	int ret = 0;

	if (!audio || volume < 0 || volume > 100 ||
	    !audio->config.internal_codec)
		return -1;
	pthread_mutex_lock(&audio->lock);
	ret = set_codec_volume(audio, volume);
	if (!ret)
		audio->volume = volume;
	pthread_mutex_unlock(&audio->lock);
	return ret;
}

int xiaozhi_audio_get_volume(struct xiaozhi_audio *audio, int *volume)
{
	float db;

	if (!audio || !volume)
		return -1;
	pthread_mutex_lock(&audio->lock);
	if (audio->config.internal_codec && !open_codec(audio) &&
	    !ioctl(audio->codec_fd, k_acodec_get_dacl_volume, &db))
		audio->volume = db_to_volume(db);
	*volume = audio->volume;
	pthread_mutex_unlock(&audio->lock);
	return 0;
}
