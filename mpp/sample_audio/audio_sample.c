/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "audio_sample.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "audio_file.h"
#include "audio_io.h"
#include "audio_wav.h"
#include "k_acodec_comm.h"
#include "k_sys_comm.h"
#include "k_vb_comm.h"
#include "mpi_adec_api.h"
#include "mpi_aenc_api.h"
#include "mpi_ai_api.h"
#include "mpi_ao_api.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"

#define AUDIO_FRAMES_PER_SECOND 25
#define AUDIO_QUEUE_FRAMES 25
#define AUDIO_FRAME_TIMEOUT_MS 100
#define AUDIO_RECORD_SECONDS 15
#define AUDIO_BUFFER_COUNT 4
#define AUDIO_MAX_SAMPLE_RATE 192000
#define AUDIO_OUTPUT_PIPELINE_FRAMES 2
#define AUDIO_DRAIN_POLL_US 10000
#define AUDIO_FILE_RING_BLOCKS 32
#define SAMPLE_OPERATION_STOPPED 1

typedef struct
{
    k_vb_blk_handle handle;
    k_u64 phys_addr;
    void *virt_addr;
    k_u32 size;
} sample_audio_buffer;

typedef struct
{
    FILE *file;
    long size;
    long index;
} sample_file_reader;

typedef struct
{
    k_aenc_chn aenc_chn;
    k_adec_chn adec_chn;
    audio_io_writer *file_writer;
    audio_io_reader *file_reader;
    sample_audio_buffer *play_buffer;
    k_u32 frame_size;
    k_bool failed;
} overall_thread_context;

static k_bool g_exit_requested;
static k_bool g_enable_audio_codec = K_FALSE;
static k_bool g_vb_initialized = K_FALSE;
static k_u32 g_audio_pool_id = VB_INVALID_POOLID;

static void merge_result(k_s32 *result, k_s32 operation_result,
                         const char *operation)
{
    if (operation_result != K_SUCCESS)
    {
        printf("%s failed\n", operation);
        *result = K_FAILED;
    }
}

static k_bool overall_failed(const overall_thread_context *context)
{
    return __atomic_load_n(&context->failed, __ATOMIC_ACQUIRE);
}

static void set_overall_failed(overall_thread_context *context)
{
    __atomic_store_n(&context->failed, K_TRUE, __ATOMIC_RELEASE);
}

static k_bool exit_requested(void)
{
    return __atomic_load_n(&g_exit_requested, __ATOMIC_ACQUIRE);
}

static k_bool file_io_stop_requested(void *context)
{
    (void)context;
    return exit_requested();
}

static k_bool overall_io_stop_requested(void *context)
{
    overall_thread_context *overall = context;

    return exit_requested() || overall_failed(overall);
}

static k_s32 write_wav_data(void *context, const void *data, k_u32 size)
{
    return audio_wav_writer_write(context, data, size);
}

static k_s32 read_wav_data(void *context, void *data, k_u32 capacity,
                           k_u32 *bytes_read)
{
    return audio_wav_reader_read(context, data, capacity, bytes_read);
}

static int sample_bytes(k_audio_bit_width bit_width)
{
    switch (bit_width)
    {
    case KD_AUDIO_BIT_WIDTH_16:
        return 2;
    case KD_AUDIO_BIT_WIDTH_24:
        return 3;
    case KD_AUDIO_BIT_WIDTH_32:
        return 4;
    default:
        return 0;
    }
}

static k_bool sample_rate_is_supported(k_u32 sample_rate)
{
    static const k_u32 rates[] = {
        8000, 12000, 16000, 24000, 32000,
        44100, 48000, 96000, 192000};

    for (size_t index = 0; index < sizeof(rates) / sizeof(rates[0]); ++index)
    {
        if (sample_rate == rates[index])
        {
            return K_TRUE;
        }
    }
    return K_FALSE;
}

static k_audio_bit_width bit_width_from_bits(int bits_per_sample)
{
    switch (bits_per_sample)
    {
    case 16:
        return KD_AUDIO_BIT_WIDTH_16;
    case 24:
        return KD_AUDIO_BIT_WIDTH_24;
    case 32:
        return KD_AUDIO_BIT_WIDTH_32;
    default:
        return (k_audio_bit_width)-1;
    }
}

static void wait_for_ao_drain(k_u32 frames_sent)
{
    k_u32 queued_frames = frames_sent < AUDIO_QUEUE_FRAMES
                              ? frames_sent
                              : AUDIO_QUEUE_FRAMES;
    k_u32 remaining_us =
        (queued_frames + AUDIO_OUTPUT_PIPELINE_FRAMES) *
        1000000U / AUDIO_FRAMES_PER_SECOND;

    while (remaining_us > 0 && !exit_requested())
    {
        k_u32 delay_us = remaining_us < AUDIO_DRAIN_POLL_US
                             ? remaining_us
                             : AUDIO_DRAIN_POLL_US;

        usleep(delay_us);
        remaining_us -= delay_us;
    }
}

static void init_i2s_attr(k_aio_dev_attr *attr, k_bool input, k_u32 sample_rate,
                          k_audio_bit_width bit_width, k_u32 channel_count,
                          k_i2s_in_mono_channel mono_channel,
                          k_i2s_work_mode i2s_mode)
{
    memset(attr, 0, sizeof(*attr));
    attr->audio_type = input ? KD_AUDIO_INPUT_TYPE_I2S : KD_AUDIO_OUTPUT_TYPE_I2S;
    attr->kd_audio_attr.i2s_attr.sample_rate = sample_rate;
    attr->kd_audio_attr.i2s_attr.bit_width = bit_width;
    attr->kd_audio_attr.i2s_attr.chn_cnt = 2;
    attr->kd_audio_attr.i2s_attr.snd_mode = channel_count == 1
                                               ? KD_AUDIO_SOUND_MODE_MONO
                                               : KD_AUDIO_SOUND_MODE_STEREO;
    attr->kd_audio_attr.i2s_attr.mono_channel = mono_channel;
    attr->kd_audio_attr.i2s_attr.i2s_mode = i2s_mode;
    attr->kd_audio_attr.i2s_attr.frame_num = AUDIO_QUEUE_FRAMES;
    attr->kd_audio_attr.i2s_attr.point_num_per_frame =
        sample_rate / AUDIO_FRAMES_PER_SECOND;
    attr->kd_audio_attr.i2s_attr.i2s_type = g_enable_audio_codec
                                               ? K_AIO_I2STYPE_INNERCODEC
                                               : K_AIO_I2STYPE_EXTERN;
}

static void init_pdm_attr(k_aio_dev_attr *attr, k_u32 sample_rate,
                         k_audio_bit_width bit_width, k_u32 channel_count)
{
    memset(attr, 0, sizeof(*attr));
    attr->audio_type = KD_AUDIO_INPUT_TYPE_PDM;
    attr->kd_audio_attr.pdm_attr.sample_rate = sample_rate;
    attr->kd_audio_attr.pdm_attr.bit_width = bit_width;
    attr->kd_audio_attr.pdm_attr.chn_cnt = 4;
    attr->kd_audio_attr.pdm_attr.snd_mode = channel_count == 1
                                              ? KD_AUDIO_SOUND_MODE_MONO
                                              : KD_AUDIO_SOUND_MODE_STEREO;
    attr->kd_audio_attr.pdm_attr.frame_num = AUDIO_QUEUE_FRAMES;
    attr->kd_audio_attr.pdm_attr.pdm_oversample = KD_AUDIO_PDM_INPUT_OVERSAMPLE_64;
    attr->kd_audio_attr.pdm_attr.point_num_per_frame =
        sample_rate / AUDIO_FRAMES_PER_SECOND;
}

static void init_codec_input_attr(k_aio_dev_attr *attr, k_audio_dev ai_dev,
                                  k_u32 sample_rate,
                                  k_audio_bit_width bit_width)
{
    if (ai_dev == 0)
    {
        init_i2s_attr(attr, K_TRUE, sample_rate, bit_width, 1,
                      KD_I2S_IN_MONO_RIGHT_CHANNEL, K_STANDARD_MODE);
    }
    else
    {
        init_pdm_attr(attr, sample_rate, bit_width, 1);
    }
}

static k_s32 enable_audio3a(k_audio_dev dev, k_ai_chn chn,
                           k_audio_bit_width bit_width, k_u32 mask)
{
    k_ai_vqe_enable vqe;

    if (mask == 0)
    {
        return K_SUCCESS;
    }
    if (bit_width != KD_AUDIO_BIT_WIDTH_16)
    {
        printf("audio3a only supports 16-bit audio\n");
        return K_FAILED;
    }

    memset(&vqe, 0, sizeof(vqe));
    vqe.ans_enable = (mask & 0x1) != 0;
    vqe.agc_enable = (mask & 0x2) != 0;
    vqe.aec_enable = (mask & 0x4) != 0;
    if (kd_mpi_ai_set_vqe_attr(dev, chn, vqe) != K_SUCCESS)
    {
        printf("kd_mpi_ai_set_vqe_attr failed\n");
        return K_FAILED;
    }

    return K_SUCCESS;
}

static k_s32 start_ai(k_audio_dev dev, k_ai_chn chn,
                      const k_aio_dev_attr *attr, k_u32 audio3a)
{
    if (kd_mpi_ai_set_pub_attr(dev, attr) != K_SUCCESS)
    {
        printf("kd_mpi_ai_set_pub_attr failed\n");
        return K_FAILED;
    }
    if (kd_mpi_ai_enable(dev) != K_SUCCESS)
    {
        printf("kd_mpi_ai_enable failed\n");
        return K_FAILED;
    }
    if (enable_audio3a(dev, chn, attr->audio_type == KD_AUDIO_INPUT_TYPE_I2S
                                    ? attr->kd_audio_attr.i2s_attr.bit_width
                                    : attr->kd_audio_attr.pdm_attr.bit_width,
                      audio3a) != K_SUCCESS ||
        kd_mpi_ai_enable_chn(dev, chn) != K_SUCCESS)
    {
        printf("kd_mpi_ai_enable_chn failed\n");
        kd_mpi_ai_disable(dev);
        return K_FAILED;
    }

    return K_SUCCESS;
}

static k_s32 stop_ai(k_audio_dev dev, k_ai_chn chn)
{
    k_s32 ret = K_SUCCESS;

    if (kd_mpi_ai_disable_chn(dev, chn) != K_SUCCESS)
    {
        printf("kd_mpi_ai_disable_chn failed\n");
        ret = K_FAILED;
    }
    if (kd_mpi_ai_disable(dev) != K_SUCCESS)
    {
        printf("kd_mpi_ai_disable failed\n");
        ret = K_FAILED;
    }
    return ret;
}

static k_s32 start_ao(k_audio_dev dev, k_ao_chn chn,
                      const k_aio_dev_attr *attr)
{
    if (kd_mpi_ao_set_pub_attr(dev, attr) != K_SUCCESS)
    {
        printf("kd_mpi_ao_set_pub_attr failed\n");
        return K_FAILED;
    }
    if (kd_mpi_ao_enable(dev) != K_SUCCESS)
    {
        printf("kd_mpi_ao_enable failed\n");
        return K_FAILED;
    }
    if (kd_mpi_ao_enable_chn(dev, chn) != K_SUCCESS)
    {
        printf("kd_mpi_ao_enable_chn failed\n");
        kd_mpi_ao_disable(dev);
        return K_FAILED;
    }

    return K_SUCCESS;
}

static k_s32 stop_ao(k_audio_dev dev, k_ao_chn chn)
{
    k_s32 ret = K_SUCCESS;

    if (kd_mpi_ao_disable_chn(dev, chn) != K_SUCCESS)
    {
        printf("kd_mpi_ao_disable_chn failed\n");
        ret = K_FAILED;
    }
    if (kd_mpi_ao_disable(dev) != K_SUCCESS)
    {
        printf("kd_mpi_ao_disable failed\n");
        ret = K_FAILED;
    }
    return ret;
}

static k_mpp_chn make_mpp_channel(k_mod_id module, k_s32 dev, k_s32 chn)
{
    k_mpp_chn channel;

    channel.mod_id = module;
    channel.dev_id = dev;
    channel.chn_id = chn;
    return channel;
}

static k_s32 allocate_audio_buffer(sample_audio_buffer *buffer, k_u32 size)
{
    memset(buffer, 0, sizeof(*buffer));
    buffer->handle = VB_INVALID_HANDLE;
    buffer->size = size;
    buffer->handle = kd_mpi_vb_get_block(g_audio_pool_id, size, NULL);
    if (buffer->handle == VB_INVALID_HANDLE)
    {
        printf("get audio VB block (%u bytes) failed\n", size);
        return K_FAILED;
    }

    buffer->phys_addr = kd_mpi_vb_handle_to_phyaddr(buffer->handle);
    buffer->virt_addr = kd_mpi_sys_mmap(buffer->phys_addr, size);
    if (buffer->virt_addr == NULL)
    {
        kd_mpi_vb_release_block(buffer->handle);
        buffer->handle = VB_INVALID_HANDLE;
        return K_FAILED;
    }

    return K_SUCCESS;
}

static k_s32 release_audio_buffer(sample_audio_buffer *buffer)
{
    k_s32 ret = K_SUCCESS;

    if (buffer->virt_addr != NULL)
    {
        if (kd_mpi_sys_munmap(buffer->virt_addr, buffer->size) != K_SUCCESS)
        {
            printf("audio buffer munmap failed\n");
            ret = K_FAILED;
        }
    }
    if (buffer->handle != VB_INVALID_HANDLE)
    {
        if (kd_mpi_vb_release_block(buffer->handle) != K_SUCCESS)
        {
            printf("audio VB block release failed\n");
            ret = K_FAILED;
        }
    }
    memset(buffer, 0, sizeof(*buffer));
    buffer->handle = VB_INVALID_HANDLE;
    return ret;
}

static void init_audio_frame(k_audio_frame *frame,
                             const sample_audio_buffer *buffer)
{
    memset(frame, 0, sizeof(*frame));
    frame->len = buffer->size;
    frame->pool_id = kd_mpi_vb_handle_to_pool_id(buffer->handle);
    frame->phys_addr = buffer->phys_addr;
    frame->virt_addr = buffer->virt_addr;
}

static void init_audio_stream(k_audio_stream *stream,
                              const sample_audio_buffer *buffer)
{
    memset(stream, 0, sizeof(*stream));
    stream->len = buffer->size;
    stream->phys_addr = buffer->phys_addr;
    stream->stream = buffer->virt_addr;
}

static k_s32 open_looping_file(sample_file_reader *reader, const char *filename)
{
    if (reader == NULL || filename == NULL)
    {
        return K_FAILED;
    }
    memset(reader, 0, sizeof(*reader));
    reader->file = fopen(filename, "rb");
    if (reader->file == NULL || fseek(reader->file, 0, SEEK_END) != 0)
    {
        printf("open input file %s failed\n", filename);
        if (reader->file != NULL)
        {
            fclose(reader->file);
            reader->file = NULL;
        }
        return K_FAILED;
    }

    reader->size = ftell(reader->file);
    if (reader->size <= 0 || fseek(reader->file, 0, SEEK_SET) != 0)
    {
        printf("input file %s is empty or unreadable\n", filename);
        fclose(reader->file);
        reader->file = NULL;
        return K_FAILED;
    }

    return K_SUCCESS;
}

static k_s32 close_looping_file(sample_file_reader *reader)
{
    k_s32 ret = K_SUCCESS;

    if (reader->file != NULL)
    {
        if (fclose(reader->file) != 0)
        {
            ret = K_FAILED;
        }
    }
    memset(reader, 0, sizeof(*reader));
    return ret;
}

static k_s32 read_looping_file(sample_file_reader *reader, void *data, k_u32 size)
{
    k_u8 *output = data;
    size_t left = size;

    if (reader->file == NULL || data == NULL || size == 0)
    {
        return K_FAILED;
    }

    while (left > 0)
    {
        size_t available = reader->size - reader->index;
        size_t chunk_size = left < available ? left : available;

        if (fread(output, 1, chunk_size, reader->file) != chunk_size)
        {
            return K_FAILED;
        }

        output += chunk_size;
        left -= chunk_size;
        reader->index += chunk_size;
        if (reader->index == reader->size)
        {
            if (fseek(reader->file, 0, SEEK_SET) != 0)
            {
                return K_FAILED;
            }
            reader->index = 0;
        }
    }

    return K_SUCCESS;
}

static k_s32 read_looping_data(void *context, void *data, k_u32 capacity,
                               k_u32 *bytes_read)
{
    if (read_looping_file(context, data, capacity) != K_SUCCESS)
    {
        return K_FAILED;
    }
    *bytes_read = capacity;
    return K_SUCCESS;
}

static k_s32 write_file_data(void *context, const void *data, k_u32 size)
{
    FILE *file = context;

    return fwrite(data, 1, size, file) == size ? K_SUCCESS : K_FAILED;
}

static k_s32 queue_audio_stream(audio_io_writer *writer,
                                const k_audio_stream *stream)
{
    k_u8 *data = kd_mpi_sys_mmap(stream->phys_addr, stream->len);
    k_s32 ret;

    if (data == NULL)
    {
        return K_FAILED;
    }
    ret = audio_io_writer_push(writer, data, stream->len);
    if (kd_mpi_sys_munmap(data, stream->len) != K_SUCCESS)
    {
        ret = K_FAILED;
    }
    return ret;
}

static k_s32 send_adec_stream(k_adec_chn chn, k_audio_stream *stream,
                              k_bool block, const k_bool *local_stop)
{
    int retries = 0;

    while (!exit_requested() &&
           (local_stop == NULL ||
            !__atomic_load_n(local_stop, __ATOMIC_ACQUIRE)))
    {
        if (kd_mpi_adec_send_stream(chn, stream, block) == K_SUCCESS)
        {
            return K_SUCCESS;
        }
        if (++retries >= 100)
        {
            return K_FAILED;
        }
        usleep(10000);
    }

    return SAMPLE_OPERATION_STOPPED;
}

static void init_aenc_attr(k_aenc_chn_attr *attr, k_payload_type type,
                           k_u32 sample_rate, k_u32 channels)
{
    memset(attr, 0, sizeof(*attr));
    attr->type = type;
    attr->buf_size = AUDIO_FRAMES_PER_SECOND;
    attr->point_num_per_frame = sample_rate / AUDIO_FRAMES_PER_SECOND;
    attr->sample_rate = sample_rate;
    attr->channels = channels;
    if (type == K_PT_OPUS)
    {
        attr->bitrate = 16000;
    }
}

static void init_adec_attr(k_adec_chn_attr *attr, k_payload_type type,
                           k_u32 sample_rate, k_u32 channels)
{
    memset(attr, 0, sizeof(*attr));
    attr->type = type;
    attr->buf_size = AUDIO_FRAMES_PER_SECOND;
    attr->point_num_per_frame = sample_rate / AUDIO_FRAMES_PER_SECOND;
    attr->sample_rate = sample_rate;
    attr->channels = channels;
}

static k_s32 record_ai_frames(const char *filename, k_audio_dev dev, k_ai_chn chn,
                              const k_aio_dev_attr *attr,
                              k_audio_bit_width bit_width, k_u32 sample_rate,
                              k_u32 channel_count, k_u32 audio3a)
{
    audio_wav_format format;
    audio_wav_writer *writer = NULL;
    audio_io_writer *file_writer = NULL;
    k_u32 target_data_size;
    k_u32 byte_rate;
    k_u32 expected_frame_size;
    k_u32 captured_size = 0;
    k_u32 expected_sequence = 0;
    k_u64 target_size;
    k_u64 written_size = 0;
    int bytes_per_sample;
    k_s32 ret = K_FAILED;
    k_bool ai_started = K_FALSE;
    k_bool sequence_started = K_FALSE;

    bytes_per_sample = sample_bytes(bit_width);
    if (bytes_per_sample == 0)
    {
        goto cleanup;
    }
    format.channel_count = channel_count;
    format.sample_rate = sample_rate;
    format.bits_per_sample = bytes_per_sample * 8;
    if (audio_wav_format_byte_rate(&format, &byte_rate) != K_SUCCESS ||
        audio_wav_format_frame_size(&format, AUDIO_FRAMES_PER_SECOND,
                                    &expected_frame_size) != K_SUCCESS)
    {
        goto cleanup;
    }
    target_size = (k_u64)byte_rate * AUDIO_RECORD_SECONDS;
    if (target_size > AUDIO_WAV_MAX_DATA_SIZE)
    {
        goto cleanup;
    }
    target_data_size = (k_u32)target_size;
    if (audio_wav_writer_open(&writer, filename, &format) != K_SUCCESS)
    {
        printf("open output WAV %s failed\n", filename);
        goto cleanup;
    }
    printf("recording to %s\n", audio_wav_writer_path(writer));
    if (audio_io_writer_create(&file_writer, expected_frame_size,
                               AUDIO_FILE_RING_BLOCKS, write_wav_data, writer,
                               file_io_stop_requested, NULL) != K_SUCCESS)
    {
        printf("start WAV writer failed\n");
        goto cleanup;
    }
    if (start_ai(dev, chn, attr, audio3a) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_started = K_TRUE;
    ret = K_SUCCESS;

    while (!exit_requested())
    {
        k_audio_frame frame;
        k_u8 *data;
        k_u32 remaining;
        k_u32 write_size;

        if (kd_mpi_ai_get_frame(dev, chn, &frame, AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS)
        {
            continue;
        }
        if (frame.len != expected_frame_size)
        {
            printf("unexpected AI frame size: %u (expected %u)\n",
                   frame.len, expected_frame_size);
            merge_result(&ret, kd_mpi_ai_release_frame(dev, chn, &frame),
                         "kd_mpi_ai_release_frame");
            ret = K_FAILED;
            break;
        }
        if (sequence_started && frame.seq != expected_sequence)
        {
            printf("AI frame discontinuity: got %u (expected %u)\n",
                   frame.seq, expected_sequence);
            merge_result(&ret, kd_mpi_ai_release_frame(dev, chn, &frame),
                         "kd_mpi_ai_release_frame");
            ret = K_FAILED;
            break;
        }
        sequence_started = K_TRUE;
        expected_sequence = frame.seq + 1U;
        data = kd_mpi_sys_mmap(frame.phys_addr, frame.len);
        if (data == NULL)
        {
            merge_result(&ret, kd_mpi_ai_release_frame(dev, chn, &frame),
                         "kd_mpi_ai_release_frame");
            ret = K_FAILED;
            break;
        }

        remaining = target_data_size - captured_size;
        write_size = frame.len < remaining ? frame.len : remaining;
        k_s32 push_result = audio_io_writer_push(file_writer, data, write_size);
        if (push_result == K_SUCCESS)
        {
            captured_size += write_size;
        }
        else if (push_result != AUDIO_IO_STOPPED)
        {
            ret = K_FAILED;
        }
        merge_result(&ret, kd_mpi_sys_munmap(data, frame.len),
                     "kd_mpi_sys_munmap");
        merge_result(&ret, kd_mpi_ai_release_frame(dev, chn, &frame),
                     "kd_mpi_ai_release_frame");
        if (captured_size == target_data_size ||
            push_result == AUDIO_IO_STOPPED)
        {
            break;
        }
        if (ret != K_SUCCESS)
        {
            break;
        }
    }

cleanup:
    if (ai_started)
    {
        merge_result(&ret, stop_ai(dev, chn), "stop AI");
    }
    if (file_writer != NULL)
    {
        merge_result(&ret,
                     audio_io_writer_finish(&file_writer, &written_size),
                     "finish WAV writer");
        if (written_size != captured_size)
        {
            printf("WAV writer byte mismatch: captured %u, wrote %llu\n",
                   captured_size, (unsigned long long)written_size);
            ret = K_FAILED;
        }
    }
    if (writer != NULL && audio_wav_writer_data_size(writer) == 0)
    {
        merge_result(&ret, audio_wav_writer_discard(&writer),
                     "discard empty WAV output");
    }
    else
    {
        merge_result(&ret, audio_wav_writer_close(&writer),
                     "close WAV output");
    }
    return ret;
}

void audio_sample_reset(void)
{
    __atomic_store_n(&g_exit_requested, K_FALSE, __ATOMIC_RELEASE);
}

k_s32 audio_sample_exit(void)
{
    __atomic_store_n(&g_exit_requested, K_TRUE, __ATOMIC_RELEASE);
    return K_SUCCESS;
}

k_s32 audio_sample_vb_init(void)
{
    k_vb_config config;
    k_vb_pool_config pool_config;
    k_u32 block_size;

    if (g_vb_initialized)
    {
        return K_SUCCESS;
    }
    memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 64;
    if (kd_mpi_vb_set_config(&config) != K_SUCCESS ||
        kd_mpi_vb_init() != K_SUCCESS)
    {
        printf("VB initialization failed\n");
        return K_FAILED;
    }
    g_vb_initialized = K_TRUE;

    block_size = (AUDIO_MAX_SAMPLE_RATE / AUDIO_FRAMES_PER_SECOND) * 4 * 2;
    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.blk_cnt = AUDIO_BUFFER_COUNT;
    pool_config.blk_size = VB_ALIGN_UP(block_size, 4096);
    pool_config.mode = VB_REMAP_MODE_NOCACHE;
    g_audio_pool_id = kd_mpi_vb_create_pool(&pool_config);
    if (g_audio_pool_id == VB_INVALID_POOLID)
    {
        printf("audio VB pool creation failed\n");
        if (kd_mpi_vb_exit() != K_SUCCESS)
        {
            printf("VB exit failed after pool creation failure\n");
        }
        g_vb_initialized = K_FALSE;
        return K_FAILED;
    }

    return K_SUCCESS;
}

k_s32 audio_sample_vb_destroy(void)
{
    k_s32 ret = K_SUCCESS;

    if (!g_vb_initialized)
    {
        return K_SUCCESS;
    }
    if (g_audio_pool_id != VB_INVALID_POOLID)
    {
        if (kd_mpi_vb_destory_pool(g_audio_pool_id) != K_SUCCESS)
        {
            printf("audio VB pool destruction failed\n");
            ret = K_FAILED;
        }
        g_audio_pool_id = VB_INVALID_POOLID;
    }
    if (kd_mpi_vb_exit() != K_SUCCESS)
    {
        printf("VB exit failed\n");
        ret = K_FAILED;
    }
    g_vb_initialized = K_FALSE;
    return ret;
}

k_s32 audio_sample_enable_audio_codec(k_bool enable_audio_codec)
{
    g_enable_audio_codec = enable_audio_codec;
    return K_SUCCESS;
}

k_s32 audio_sample_get_ai_i2s_data(const char *filename,
                                   k_audio_bit_width bit_width,
                                   k_u32 sample_rate, k_u32 channel_count,
                                   k_i2s_in_mono_channel mono_channel,
                                   k_i2s_work_mode i2s_work_mode,
                                   k_u32 enable_audio3a)
{
    k_aio_dev_attr attr;

    init_i2s_attr(&attr, K_TRUE, sample_rate, bit_width, channel_count,
                  mono_channel, i2s_work_mode);
    return record_ai_frames(filename, 0, 0, &attr, bit_width, sample_rate,
                            channel_count, enable_audio3a);
}

k_s32 audio_sample_get_ai_pdm_data(const char *filename,
                                   k_audio_bit_width bit_width,
                                   k_u32 sample_rate, k_u32 channel_count,
                                   k_u32 enable_audio3a)
{
    k_aio_dev_attr attr;

    init_pdm_attr(&attr, sample_rate, bit_width, channel_count);
    return record_ai_frames(filename, 1, 0, &attr, bit_width, sample_rate,
                            channel_count, enable_audio3a);
}

k_s32 audio_sample_send_ao_data(const char *filename, int dev, int chn,
                                k_i2s_work_mode i2s_work_mode)
{
    audio_wav_reader *reader = NULL;
    audio_io_reader *file_reader = NULL;
    audio_wav_format format;
    int bytes_per_sample;
    k_audio_bit_width bit_width;
    k_aio_dev_attr attr;
    sample_audio_buffer buffer;
    k_audio_frame frame;
    k_bool buffer_allocated = K_FALSE;
    k_bool ao_started = K_FALSE;
    k_s32 ret = K_FAILED;
    k_u32 byte_rate;
    k_u32 frame_size;
    k_u32 data_size;
    k_u32 data_read = 0;
    k_u32 frames_sent = 0;
    k_bool reached_end = K_FALSE;
    k_u64 duration_ms;

    if (audio_wav_reader_open(&reader, filename) != K_SUCCESS ||
        audio_wav_reader_get_format(reader, &format) != K_SUCCESS ||
        audio_wav_reader_data_size(reader) == 0)
    {
        printf("open input WAV failed%s%s\n",
               filename == NULL ? "" : ": ",
               filename == NULL ? "" : filename);
        goto cleanup;
    }
    data_size = audio_wav_reader_data_size(reader);
    if (!sample_rate_is_supported(format.sample_rate))
    {
        printf("WAV sample rate %u is not supported by AO\n", format.sample_rate);
        goto cleanup;
    }
    bit_width = bit_width_from_bits(format.bits_per_sample);
    bytes_per_sample = sample_bytes(bit_width);
    if (audio_wav_format_byte_rate(&format, &byte_rate) != K_SUCCESS ||
        audio_wav_format_frame_size(&format, AUDIO_FRAMES_PER_SECOND,
                                    &frame_size) != K_SUCCESS)
    {
        goto cleanup;
    }
    duration_ms = (k_u64)data_size * 1000U / byte_rate;
    printf("WAV: rate=%u bits=%u channels=%u byte-rate=%u frame-bytes=%u "
           "duration=%llu.%03llu s\n",
           format.sample_rate, format.bits_per_sample, format.channel_count,
           byte_rate, frame_size, (unsigned long long)(duration_ms / 1000U),
           (unsigned long long)(duration_ms % 1000U));
    if (g_enable_audio_codec && bit_width == KD_AUDIO_BIT_WIDTH_32)
    {
        printf("the internal codec cannot play 32-bit WAV data\n");
        goto cleanup;
    }
    if (bytes_per_sample == 0 ||
        allocate_audio_buffer(&buffer, frame_size) != K_SUCCESS)
    {
        goto cleanup;
    }
    buffer_allocated = K_TRUE;
    init_audio_frame(&frame, &buffer);
    if (audio_io_reader_create(&file_reader, frame_size,
                               AUDIO_FILE_RING_BLOCKS, read_wav_data, reader,
                               file_io_stop_requested, NULL) != K_SUCCESS)
    {
        printf("start WAV reader failed\n");
        goto cleanup;
    }

    if (!g_enable_audio_codec)
    {
        i2s_work_mode = K_RIGHT_JUSTIFYING_MODE;
    }
    init_i2s_attr(&attr, K_FALSE, format.sample_rate, bit_width,
                  format.channel_count,
                  KD_I2S_IN_MONO_RIGHT_CHANNEL, i2s_work_mode);
    if (start_ao(dev, chn, &attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_started = K_TRUE;
    ret = K_SUCCESS;

    while (!exit_requested())
    {
        k_u32 bytes_read = 0;
        k_bool final_frame;
        k_s32 read_result;

        read_result = audio_io_reader_pop(file_reader, frame.virt_addr,
                                          frame.len, &bytes_read);
        if (read_result == AUDIO_IO_END)
        {
            reached_end = K_TRUE;
            break;
        }
        if (read_result == AUDIO_IO_STOPPED)
        {
            break;
        }
        if (read_result != K_SUCCESS)
        {
            if (!exit_requested())
            {
                ret = K_FAILED;
            }
            break;
        }
        final_frame = data_read + bytes_read == data_size;
        if (bytes_read < frame.len)
        {
            memset((k_u8 *)frame.virt_addr + bytes_read, 0,
                   frame.len - bytes_read);
        }
        if (kd_mpi_ao_send_frame(dev, chn, &frame,
                                 AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS)
        {
            if (!exit_requested())
            {
                ret = K_FAILED;
            }
            break;
        }
        data_read += bytes_read;
        ++frames_sent;
        if (final_frame)
        {
            reached_end = K_TRUE;
            break;
        }
    }
    if (reached_end && ret == K_SUCCESS && !exit_requested())
    {
        wait_for_ao_drain(frames_sent);
    }

cleanup:
    if (file_reader != NULL)
    {
        merge_result(&ret, audio_io_reader_destroy(&file_reader),
                     "stop WAV reader");
    }
    if (ao_started)
    {
        merge_result(&ret, stop_ao(dev, chn), "stop AO");
    }
    if (buffer_allocated)
    {
        merge_result(&ret, release_audio_buffer(&buffer),
                     "release playback buffer");
    }
    if (audio_wav_reader_close(&reader) != K_SUCCESS)
    {
        ret = K_FAILED;
    }
    return ret;
}

static k_s32 validate_duplex_configuration(k_ai_chn ai_chn,
                                           k_ao_chn ao_chn,
                                           k_audio_bit_width bit_width)
{
    if (g_enable_audio_codec && (ai_chn != 0 || ao_chn != 0))
    {
        printf("the built-in codec requires AI/AO channel 0\n");
        return K_FAILED;
    }
    if (!g_enable_audio_codec && bit_width != KD_AUDIO_BIT_WIDTH_32)
    {
        printf("external I2S duplex mode requires 32-bit samples\n");
        return K_FAILED;
    }
    return K_SUCCESS;
}

static void init_duplex_attrs(k_aio_dev_attr *ai_attr, k_aio_dev_attr *ao_attr,
                              k_audio_dev ai_dev, k_u32 sample_rate,
                              k_audio_bit_width bit_width,
                              k_i2s_work_mode i2s_work_mode)
{
    if (ai_dev == 0)
    {
        init_i2s_attr(ai_attr, K_TRUE, sample_rate, bit_width, 2,
                      KD_I2S_IN_MONO_RIGHT_CHANNEL, i2s_work_mode);
    }
    else
    {
        init_pdm_attr(ai_attr, sample_rate, bit_width, 2);
    }
    init_i2s_attr(ao_attr, K_FALSE, sample_rate, bit_width, 2,
                  KD_I2S_IN_MONO_RIGHT_CHANNEL, i2s_work_mode);
}

k_s32 audio_sample_api_ai_to_ao(int ai_dev, int ai_chn, int ao_dev, int ao_chn,
                                k_u32 sample_rate,
                                k_audio_bit_width bit_width,
                                k_i2s_work_mode i2s_work_mode,
                                k_u32 enable_audio3a)
{
    k_aio_dev_attr ai_attr;
    k_aio_dev_attr ao_attr;
    k_bool ai_started = K_FALSE;
    k_bool ao_started = K_FALSE;
    k_s32 ret = K_FAILED;

    if (validate_duplex_configuration(ai_chn, ao_chn, bit_width) != K_SUCCESS)
    {
        return K_FAILED;
    }
    init_duplex_attrs(&ai_attr, &ao_attr, ai_dev, sample_rate, bit_width,
                      i2s_work_mode);
    if (start_ai(ai_dev, ai_chn, &ai_attr, enable_audio3a) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_started = K_TRUE;
    if (start_ao(ao_dev, ao_chn, &ao_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_started = K_TRUE;
    ret = K_SUCCESS;

    while (!exit_requested())
    {
        k_audio_frame frame;

        if (kd_mpi_ai_get_frame(ai_dev, ai_chn, &frame,
                                AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS)
        {
            continue;
        }
        if (kd_mpi_ao_send_frame(ao_dev, ao_chn, &frame,
                                 AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS &&
            !exit_requested())
        {
            ret = K_FAILED;
        }
        merge_result(&ret,
                     kd_mpi_ai_release_frame(ai_dev, ai_chn, &frame),
                     "kd_mpi_ai_release_frame");
        if (ret != K_SUCCESS)
        {
            break;
        }
    }

cleanup:
    if (ao_started)
    {
        merge_result(&ret, stop_ao(ao_dev, ao_chn), "stop AO");
    }
    if (ai_started)
    {
        merge_result(&ret, stop_ai(ai_dev, ai_chn), "stop AI");
    }
    return ret;
}

k_s32 audio_sample_bind_ai_to_ao(int ai_dev, int ai_chn, int ao_dev, int ao_chn,
                                 k_u32 sample_rate,
                                 k_audio_bit_width bit_width,
                                 k_i2s_work_mode i2s_work_mode,
                                 k_u32 enable_audio3a)
{
    k_aio_dev_attr ai_attr;
    k_aio_dev_attr ao_attr;
    k_mpp_chn ai_channel = make_mpp_channel(K_ID_AI, ai_dev, ai_chn);
    k_mpp_chn ao_channel = make_mpp_channel(K_ID_AO, ao_dev, ao_chn);
    k_bool ai_started = K_FALSE;
    k_bool ao_started = K_FALSE;
    k_bool bound = K_FALSE;
    k_s32 ret = K_FAILED;

    if (validate_duplex_configuration(ai_chn, ao_chn, bit_width) != K_SUCCESS)
    {
        return K_FAILED;
    }
    init_duplex_attrs(&ai_attr, &ao_attr, ai_dev, sample_rate, bit_width,
                      i2s_work_mode);
    if (start_ai(ai_dev, ai_chn, &ai_attr, enable_audio3a) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_started = K_TRUE;
    if (start_ao(ao_dev, ao_chn, &ao_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_started = K_TRUE;
    if (kd_mpi_sys_bind(&ai_channel, &ao_channel) != K_SUCCESS)
    {
        printf("AI to AO bind failed\n");
        goto cleanup;
    }
    bound = K_TRUE;
    ret = K_SUCCESS;

    while (!exit_requested())
    {
        usleep(100000);
    }

cleanup:
    if (bound)
    {
        merge_result(&ret, kd_mpi_sys_unbind(&ai_channel, &ao_channel),
                     "AI to AO unbind");
    }
    if (ao_started)
    {
        merge_result(&ret, stop_ao(ao_dev, ao_chn), "stop AO");
    }
    if (ai_started)
    {
        merge_result(&ret, stop_ai(ai_dev, ai_chn), "stop AI");
    }
    return ret;
}

k_s32 audio_sample_ai_encode(k_audio_dev ai_dev, k_bool use_sysbind,
                             k_u32 sample_rate, k_audio_bit_width bit_width,
                             int enc_chn, k_payload_type type,
                             const char *filename, k_u32 enable_audio3a)
{
    k_aio_dev_attr ai_attr;
    k_aenc_chn_attr aenc_attr;
    k_mpp_chn ai_channel = make_mpp_channel(K_ID_AI, ai_dev, 0);
    k_mpp_chn aenc_channel = make_mpp_channel(K_ID_AENC, 0, enc_chn);
    FILE *output = NULL;
    audio_io_writer *file_writer = NULL;
    char output_path[AUDIO_FILE_PATH_SIZE];
    k_bool encoder_created = K_FALSE;
    k_bool ai_started = K_FALSE;
    k_bool bound = K_FALSE;
    k_s32 ret = K_FAILED;
    k_u64 output_size = 0;
    k_u32 encoded_block_size = sample_rate / AUDIO_FRAMES_PER_SECOND * 2U;

    if (bit_width != KD_AUDIO_BIT_WIDTH_16)
    {
        bit_width = KD_AUDIO_BIT_WIDTH_16;
        printf("encoder input forced to 16-bit\n");
    }

    if (audio_file_open_unique(filename, "wb", &output, output_path,
                               sizeof(output_path)) != K_SUCCESS)
    {
        printf("open output file %s failed\n", filename);
        goto cleanup;
    }
    printf("recording to %s\n", output_path);
    if (audio_io_writer_create(&file_writer, encoded_block_size,
                               AUDIO_FILE_RING_BLOCKS, write_file_data, output,
                               file_io_stop_requested, NULL) != K_SUCCESS)
    {
        printf("start encoded writer failed\n");
        goto cleanup;
    }
    init_aenc_attr(&aenc_attr, type, sample_rate, 1);
    if (kd_mpi_aenc_create_chn(enc_chn, &aenc_attr) != K_SUCCESS)
    {
        printf("kd_mpi_aenc_create_chn failed\n");
        goto cleanup;
    }
    encoder_created = K_TRUE;

    init_codec_input_attr(&ai_attr, ai_dev, sample_rate, bit_width);
    if (start_ai(ai_dev, 0, &ai_attr, enable_audio3a) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_started = K_TRUE;

    if (use_sysbind)
    {
        if (kd_mpi_sys_bind(&ai_channel, &aenc_channel) != K_SUCCESS)
        {
            printf("AI to AENC bind failed\n");
            goto cleanup;
        }
        bound = K_TRUE;
    }
    ret = K_SUCCESS;

    while (!exit_requested())
    {
        k_audio_frame frame;
        k_audio_stream stream;
        k_bool frame_acquired = K_FALSE;

        if (!use_sysbind)
        {
            if (kd_mpi_ai_get_frame(ai_dev, 0, &frame,
                                    AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS)
            {
                continue;
            }
            frame_acquired = K_TRUE;
            if (kd_mpi_aenc_send_frame(enc_chn, &frame) != K_SUCCESS)
            {
                merge_result(&ret,
                             kd_mpi_ai_release_frame(ai_dev, 0, &frame),
                             "kd_mpi_ai_release_frame");
                ret = K_FAILED;
                break;
            }
        }

        if (kd_mpi_aenc_get_stream(enc_chn, &stream,
                                   AUDIO_FRAME_TIMEOUT_MS) == K_SUCCESS)
        {
            k_s32 queue_result = queue_audio_stream(file_writer, &stream);

            if (queue_result != K_SUCCESS &&
                queue_result != AUDIO_IO_STOPPED)
            {
                ret = K_FAILED;
            }
            merge_result(&ret, kd_mpi_aenc_release_stream(enc_chn, &stream),
                         "kd_mpi_aenc_release_stream");
            if (queue_result == AUDIO_IO_STOPPED)
            {
                break;
            }
        }
        if (frame_acquired)
        {
            merge_result(&ret, kd_mpi_ai_release_frame(ai_dev, 0, &frame),
                         "kd_mpi_ai_release_frame");
        }
        if (ret != K_SUCCESS)
        {
            break;
        }
    }

cleanup:
    if (bound)
    {
        merge_result(&ret, kd_mpi_sys_unbind(&ai_channel, &aenc_channel),
                     "AI to AENC unbind");
    }
    if (ai_started)
    {
        merge_result(&ret, stop_ai(ai_dev, 0), "stop AI");
    }
    if (encoder_created)
    {
        merge_result(&ret, kd_mpi_aenc_destroy_chn(enc_chn),
                     "destroy AENC channel");
    }
    if (file_writer != NULL)
    {
        merge_result(&ret,
                     audio_io_writer_finish(&file_writer, &output_size),
                     "finish encoded writer");
    }
    if (output != NULL)
    {
        if (output_size == 0)
        {
            merge_result(&ret, audio_file_discard(&output, output_path),
                         "discard empty encoded output");
        }
        else if (fclose(output) != 0)
        {
            printf("close encoded output failed\n");
            ret = K_FAILED;
        }
    }
    return ret;
}

k_s32 audio_sample_decode_ao(k_bool use_sysbind, k_u32 sample_rate,
                             k_audio_bit_width bit_width, int dec_chn,
                             k_payload_type type, const char *filename)
{
    k_aio_dev_attr ao_attr;
    k_adec_chn_attr adec_attr;
    k_mpp_chn adec_channel = make_mpp_channel(K_ID_ADEC, 0, dec_chn);
    k_mpp_chn ao_channel = make_mpp_channel(K_ID_AO, 0, 0);
    sample_file_reader reader;
    audio_io_reader *file_reader = NULL;
    sample_audio_buffer buffer;
    k_audio_stream stream;
    k_bool reader_open = K_FALSE;
    k_bool decoder_created = K_FALSE;
    k_bool ao_started = K_FALSE;
    k_bool buffer_allocated = K_FALSE;
    k_bool bound = K_FALSE;
    k_s32 ret = K_FAILED;
    k_u32 frame_size;

    if (bit_width != KD_AUDIO_BIT_WIDTH_16)
    {
        bit_width = KD_AUDIO_BIT_WIDTH_16;
        printf("decoder output forced to 16-bit\n");
    }
    if (open_looping_file(&reader, filename) != K_SUCCESS)
    {
        goto cleanup;
    }
    reader_open = K_TRUE;
    frame_size = sample_rate / AUDIO_FRAMES_PER_SECOND;
    if (audio_io_reader_create(&file_reader, frame_size,
                               AUDIO_FILE_RING_BLOCKS, read_looping_data,
                               &reader, file_io_stop_requested, NULL) !=
        K_SUCCESS)
    {
        printf("start encoded reader failed\n");
        goto cleanup;
    }

    init_adec_attr(&adec_attr, type, sample_rate, 1);
    if (kd_mpi_adec_create_chn(dec_chn, &adec_attr) != K_SUCCESS)
    {
        printf("kd_mpi_adec_create_chn failed\n");
        goto cleanup;
    }
    decoder_created = K_TRUE;

    init_i2s_attr(&ao_attr, K_FALSE, sample_rate, bit_width, 1,
                  KD_I2S_IN_MONO_RIGHT_CHANNEL,
                  g_enable_audio_codec ? K_STANDARD_MODE : K_RIGHT_JUSTIFYING_MODE);
    if (start_ao(0, 0, &ao_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_started = K_TRUE;
    kd_mpi_adec_clr_chn_buf(dec_chn);

    if (use_sysbind)
    {
        if (kd_mpi_sys_bind(&adec_channel, &ao_channel) != K_SUCCESS)
        {
            printf("ADEC to AO bind failed\n");
            goto cleanup;
        }
        bound = K_TRUE;
    }

    if (allocate_audio_buffer(&buffer, frame_size) != K_SUCCESS)
    {
        goto cleanup;
    }
    buffer_allocated = K_TRUE;
    init_audio_stream(&stream, &buffer);
    ret = K_SUCCESS;

    while (!exit_requested())
    {
        k_audio_frame frame;
        k_u32 bytes_read = 0;
        k_s32 read_result = audio_io_reader_pop(
            file_reader, stream.stream, frame_size, &bytes_read);

        if (read_result == AUDIO_IO_STOPPED)
        {
            break;
        }
        if (read_result != K_SUCCESS || bytes_read != frame_size)
        {
            ret = K_FAILED;
            break;
        }
        stream.len = frame_size;
        stream.seq++;
        k_s32 send_ret = send_adec_stream(dec_chn, &stream, use_sysbind, NULL);
        if (send_ret == SAMPLE_OPERATION_STOPPED)
        {
            break;
        }
        if (send_ret != K_SUCCESS)
        {
            ret = K_FAILED;
            break;
        }
        if (!use_sysbind &&
            kd_mpi_adec_get_frame(dec_chn, &frame, AUDIO_FRAME_TIMEOUT_MS) == K_SUCCESS)
        {
            if (kd_mpi_ao_send_frame(0, 0, &frame,
                                     AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS &&
                !exit_requested())
            {
                ret = K_FAILED;
            }
            merge_result(&ret, kd_mpi_adec_release_frame(dec_chn, &frame),
                         "kd_mpi_adec_release_frame");
        }
        if (ret != K_SUCCESS)
        {
            break;
        }
    }

cleanup:
    if (file_reader != NULL)
    {
        merge_result(&ret, audio_io_reader_destroy(&file_reader),
                     "stop encoded reader");
    }
    if (bound)
    {
        merge_result(&ret, kd_mpi_sys_unbind(&adec_channel, &ao_channel),
                     "ADEC to AO unbind");
    }
    if (ao_started)
    {
        merge_result(&ret, stop_ao(0, 0), "stop AO");
    }
    if (decoder_created)
    {
        merge_result(&ret, kd_mpi_adec_destroy_chn(dec_chn),
                     "destroy ADEC channel");
    }
    if (buffer_allocated)
    {
        merge_result(&ret, release_audio_buffer(&buffer),
                     "release decoder buffer");
    }
    if (reader_open)
    {
        merge_result(&ret, close_looping_file(&reader), "close encoded input");
    }
    return ret;
}

static void *overall_record_thread(void *arg)
{
    overall_thread_context *context = arg;

    while (!exit_requested() && !overall_failed(context))
    {
        k_audio_stream stream;

        if (kd_mpi_aenc_get_stream(context->aenc_chn, &stream,
                                   AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS)
        {
            continue;
        }
        k_s32 queue_result = queue_audio_stream(context->file_writer, &stream);

        if (queue_result != K_SUCCESS && queue_result != AUDIO_IO_STOPPED)
        {
            set_overall_failed(context);
        }
        if (kd_mpi_aenc_release_stream(context->aenc_chn, &stream) != K_SUCCESS)
        {
            set_overall_failed(context);
        }
        if (queue_result == AUDIO_IO_STOPPED)
        {
            break;
        }
    }
    return NULL;
}

static void *overall_play_thread(void *arg)
{
    overall_thread_context *context = arg;
    k_audio_stream stream;

    init_audio_stream(&stream, context->play_buffer);
    while (!exit_requested() && !overall_failed(context))
    {
        k_u32 bytes_read = 0;
        k_s32 read_result = audio_io_reader_pop(
            context->file_reader, stream.stream, context->frame_size,
            &bytes_read);

        if (read_result == AUDIO_IO_STOPPED)
        {
            break;
        }
        if (read_result != K_SUCCESS || bytes_read != context->frame_size)
        {
            set_overall_failed(context);
            break;
        }
        stream.len = context->frame_size;
        stream.seq++;
        k_s32 send_ret = send_adec_stream(context->adec_chn, &stream, K_TRUE,
                                          &context->failed);
        if (send_ret == SAMPLE_OPERATION_STOPPED)
        {
            break;
        }
        if (send_ret != K_SUCCESS)
        {
            set_overall_failed(context);
            break;
        }
    }
    return NULL;
}

k_s32 audio_sample_ai_aenc_adec_ao(k_audio_dev ai_dev, k_ai_chn ai_chn,
                                   k_audio_dev ao_dev, k_ao_chn ao_chn,
                                   k_aenc_chn aenc_chn, k_adec_chn adec_chn,
                                   k_u32 sample_rate,
                                   k_audio_bit_width bit_width,
                                   k_payload_type type,
                                   const char *load_filename,
                                   const char *record_filename,
                                   k_u32 enable_audio3a)
{
    k_aio_dev_attr ai_attr;
    k_aio_dev_attr ao_attr;
    k_aenc_chn_attr aenc_attr;
    k_adec_chn_attr adec_attr;
    k_mpp_chn ai_channel = make_mpp_channel(K_ID_AI, ai_dev, ai_chn);
    k_mpp_chn aenc_channel = make_mpp_channel(K_ID_AENC, 0, aenc_chn);
    k_mpp_chn adec_channel = make_mpp_channel(K_ID_ADEC, 0, adec_chn);
    k_mpp_chn ao_channel = make_mpp_channel(K_ID_AO, ao_dev, ao_chn);
    sample_file_reader reader;
    sample_audio_buffer play_buffer;
    overall_thread_context context;
    pthread_t record_thread;
    pthread_t play_thread;
    audio_io_writer *file_writer = NULL;
    audio_io_reader *file_reader = NULL;
    FILE *record_file = NULL;
    char record_path[AUDIO_FILE_PATH_SIZE];
    k_bool reader_open = K_FALSE;
    k_bool buffer_allocated = K_FALSE;
    k_bool encoder_created = K_FALSE;
    k_bool decoder_created = K_FALSE;
    k_bool ai_started = K_FALSE;
    k_bool ao_started = K_FALSE;
    k_bool ai_bound = K_FALSE;
    k_bool ao_bound = K_FALSE;
    k_bool record_thread_created = K_FALSE;
    k_bool play_thread_created = K_FALSE;
    k_s32 ret = K_FAILED;
    k_u32 frame_size;
    k_u64 output_size = 0;

    (void)bit_width;
    g_enable_audio_codec = K_TRUE;
    bit_width = KD_AUDIO_BIT_WIDTH_16;
    memset(&context, 0, sizeof(context));

    if (open_looping_file(&reader, load_filename) != K_SUCCESS)
    {
        goto cleanup;
    }
    reader_open = K_TRUE;
    if (audio_file_open_unique(record_filename, "wb", &record_file,
                               record_path, sizeof(record_path)) != K_SUCCESS)
    {
        printf("open output file %s failed\n", record_filename);
        goto cleanup;
    }
    printf("recording to %s\n", record_path);

    frame_size = sample_rate / AUDIO_FRAMES_PER_SECOND;
    if (allocate_audio_buffer(&play_buffer, frame_size) != K_SUCCESS)
    {
        goto cleanup;
    }
    buffer_allocated = K_TRUE;

    context.aenc_chn = aenc_chn;
    context.adec_chn = adec_chn;
    context.play_buffer = &play_buffer;
    context.frame_size = frame_size;
    if (audio_io_reader_create(&file_reader, frame_size,
                               AUDIO_FILE_RING_BLOCKS, read_looping_data,
                               &reader, overall_io_stop_requested,
                               &context) != K_SUCCESS)
    {
        printf("start duplex encoded reader failed\n");
        goto cleanup;
    }
    context.file_reader = file_reader;
    if (audio_io_writer_create(&file_writer, frame_size * 2U,
                               AUDIO_FILE_RING_BLOCKS, write_file_data,
                               record_file, overall_io_stop_requested,
                               &context) != K_SUCCESS)
    {
        printf("start duplex encoded writer failed\n");
        goto cleanup;
    }
    context.file_writer = file_writer;

    init_aenc_attr(&aenc_attr, type, sample_rate, 1);
    init_adec_attr(&adec_attr, type, sample_rate, 1);
    if (kd_mpi_aenc_create_chn(aenc_chn, &aenc_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    encoder_created = K_TRUE;
    if (kd_mpi_adec_create_chn(adec_chn, &adec_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    decoder_created = K_TRUE;

    init_codec_input_attr(&ai_attr, ai_dev, sample_rate, bit_width);
    init_i2s_attr(&ao_attr, K_FALSE, sample_rate, bit_width, 1,
                  KD_I2S_IN_MONO_RIGHT_CHANNEL, K_STANDARD_MODE);
    if (start_ai(ai_dev, ai_chn, &ai_attr, enable_audio3a) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_started = K_TRUE;
    if (start_ao(ao_dev, ao_chn, &ao_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_started = K_TRUE;
    if (kd_mpi_sys_bind(&ai_channel, &aenc_channel) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_bound = K_TRUE;
    if (kd_mpi_sys_bind(&adec_channel, &ao_channel) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_bound = K_TRUE;

    if (pthread_create(&play_thread, NULL, overall_play_thread, &context) != 0)
    {
        set_overall_failed(&context);
        goto join_threads;
    }
    play_thread_created = K_TRUE;
    if (pthread_create(&record_thread, NULL, overall_record_thread, &context) != 0)
    {
        set_overall_failed(&context);
        goto join_threads;
    }
    record_thread_created = K_TRUE;

join_threads:
    if (record_thread_created && pthread_join(record_thread, NULL) != 0)
    {
        set_overall_failed(&context);
    }
    if (play_thread_created && pthread_join(play_thread, NULL) != 0)
    {
        set_overall_failed(&context);
    }
    ret = overall_failed(&context) ? K_FAILED : K_SUCCESS;

cleanup:
    if (file_reader != NULL)
    {
        merge_result(&ret, audio_io_reader_destroy(&file_reader),
                     "stop duplex encoded reader");
        context.file_reader = NULL;
    }
    if (file_writer != NULL)
    {
        merge_result(&ret,
                     audio_io_writer_finish(&file_writer, &output_size),
                     "finish duplex encoded writer");
        context.file_writer = NULL;
    }
    if (ao_bound)
    {
        merge_result(&ret, kd_mpi_sys_unbind(&adec_channel, &ao_channel),
                     "ADEC to AO unbind");
    }
    if (ai_bound)
    {
        merge_result(&ret, kd_mpi_sys_unbind(&ai_channel, &aenc_channel),
                     "AI to AENC unbind");
    }
    if (ao_started)
    {
        merge_result(&ret, stop_ao(ao_dev, ao_chn), "stop AO");
    }
    if (ai_started)
    {
        merge_result(&ret, stop_ai(ai_dev, ai_chn), "stop AI");
    }
    if (decoder_created)
    {
        merge_result(&ret, kd_mpi_adec_destroy_chn(adec_chn),
                     "destroy ADEC channel");
    }
    if (encoder_created)
    {
        merge_result(&ret, kd_mpi_aenc_destroy_chn(aenc_chn),
                     "destroy AENC channel");
    }
    if (buffer_allocated)
    {
        merge_result(&ret, release_audio_buffer(&play_buffer),
                     "release decoder buffer");
    }
    if (record_file != NULL)
    {
        if (output_size == 0)
        {
            merge_result(&ret, audio_file_discard(&record_file, record_path),
                         "discard empty encoded output");
        }
        else if (fclose(record_file) != 0)
        {
            printf("close encoded output failed\n");
            ret = K_FAILED;
        }
    }
    if (reader_open)
    {
        merge_result(&ret, close_looping_file(&reader), "close encoded input");
    }
    return ret;
}

static k_s32 run_codec_loopback(k_audio_dev ai_dev, k_ai_chn ai_chn,
                                k_audio_dev ao_dev, k_ao_chn ao_chn,
                                k_aenc_chn aenc_chn, k_adec_chn adec_chn,
                                k_u32 sample_rate, k_payload_type type,
                                k_u32 enable_audio3a)
{
    k_aio_dev_attr ai_attr;
    k_aio_dev_attr ao_attr;
    k_aenc_chn_attr aenc_attr;
    k_adec_chn_attr adec_attr;
    k_mpp_chn ai_channel = make_mpp_channel(K_ID_AI, ai_dev, ai_chn);
    k_mpp_chn aenc_channel = make_mpp_channel(K_ID_AENC, 0, aenc_chn);
    k_mpp_chn adec_channel = make_mpp_channel(K_ID_ADEC, 0, adec_chn);
    k_mpp_chn ao_channel = make_mpp_channel(K_ID_AO, ao_dev, ao_chn);
    k_bool encoder_created = K_FALSE;
    k_bool decoder_created = K_FALSE;
    k_bool ai_started = K_FALSE;
    k_bool ao_started = K_FALSE;
    k_bool ai_bound = K_FALSE;
    k_bool ao_bound = K_FALSE;
    k_s32 ret = K_FAILED;

    g_enable_audio_codec = K_TRUE;
    init_aenc_attr(&aenc_attr, type, sample_rate, 1);
    init_adec_attr(&adec_attr, type, sample_rate, 1);
    if (kd_mpi_aenc_create_chn(aenc_chn, &aenc_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    encoder_created = K_TRUE;
    if (kd_mpi_adec_create_chn(adec_chn, &adec_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    decoder_created = K_TRUE;

    init_codec_input_attr(&ai_attr, ai_dev, sample_rate,
                          KD_AUDIO_BIT_WIDTH_16);
    init_i2s_attr(&ao_attr, K_FALSE, sample_rate, KD_AUDIO_BIT_WIDTH_16, 1,
                  KD_I2S_IN_MONO_RIGHT_CHANNEL, K_STANDARD_MODE);
    if (start_ai(ai_dev, ai_chn, &ai_attr, enable_audio3a) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_started = K_TRUE;
    if (start_ao(ao_dev, ao_chn, &ao_attr) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_started = K_TRUE;
    if (kd_mpi_sys_bind(&ai_channel, &aenc_channel) != K_SUCCESS)
    {
        goto cleanup;
    }
    ai_bound = K_TRUE;
    if (kd_mpi_sys_bind(&adec_channel, &ao_channel) != K_SUCCESS)
    {
        goto cleanup;
    }
    ao_bound = K_TRUE;
    ret = K_SUCCESS;

    while (!exit_requested())
    {
        k_audio_stream stream;

        if (kd_mpi_aenc_get_stream(aenc_chn, &stream,
                                   AUDIO_FRAME_TIMEOUT_MS) != K_SUCCESS)
        {
            continue;
        }
        k_s32 send_ret = send_adec_stream(adec_chn, &stream, K_FALSE, NULL);
        if (send_ret == SAMPLE_OPERATION_STOPPED)
        {
            merge_result(&ret,
                         kd_mpi_aenc_release_stream(aenc_chn, &stream),
                         "kd_mpi_aenc_release_stream");
            break;
        }
        if (send_ret != K_SUCCESS)
        {
            ret = K_FAILED;
        }
        merge_result(&ret, kd_mpi_aenc_release_stream(aenc_chn, &stream),
                     "kd_mpi_aenc_release_stream");
        if (ret != K_SUCCESS)
        {
            break;
        }
    }

cleanup:
    if (ao_bound)
    {
        merge_result(&ret, kd_mpi_sys_unbind(&adec_channel, &ao_channel),
                     "ADEC to AO unbind");
    }
    if (ai_bound)
    {
        merge_result(&ret, kd_mpi_sys_unbind(&ai_channel, &aenc_channel),
                     "AI to AENC unbind");
    }
    if (ao_started)
    {
        merge_result(&ret, stop_ao(ao_dev, ao_chn), "stop AO");
    }
    if (ai_started)
    {
        merge_result(&ret, stop_ai(ai_dev, ai_chn), "stop AI");
    }
    if (decoder_created)
    {
        merge_result(&ret, kd_mpi_adec_destroy_chn(adec_chn),
                     "destroy ADEC channel");
    }
    if (encoder_created)
    {
        merge_result(&ret, kd_mpi_aenc_destroy_chn(aenc_chn),
                     "destroy AENC channel");
    }
    return ret;
}

k_s32 audio_sample_ai_aenc_adec_ao_2(k_audio_dev ai_dev, k_ai_chn ai_chn,
                                     k_audio_dev ao_dev, k_ao_chn ao_chn,
                                     k_aenc_chn aenc_chn, k_adec_chn adec_chn,
                                     k_u32 sample_rate,
                                     k_audio_bit_width bit_width,
                                     k_payload_type type,
                                     k_u32 enable_audio3a)
{
    (void)bit_width;
    return run_codec_loopback(ai_dev, ai_chn, ao_dev, ao_chn, aenc_chn,
                              adec_chn, sample_rate, type, enable_audio3a);
}

k_s32 audio_sample_ai_aenc_adec_ao_opus(k_audio_dev ai_dev, k_ai_chn ai_chn,
                                        k_audio_dev ao_dev, k_ao_chn ao_chn,
                                        k_aenc_chn aenc_chn,
                                        k_adec_chn adec_chn,
                                        k_u32 sample_rate,
                                        k_audio_bit_width bit_width,
                                        k_payload_type type,
                                        k_u32 enable_audio3a)
{
    (void)sample_rate;
    (void)bit_width;
    return run_codec_loopback(ai_dev, ai_chn, ao_dev, ao_chn, aenc_chn,
                              adec_chn, 8000, type, enable_audio3a);
}

static k_s32 g_acodec_fd = -1;
static pthread_mutex_t g_acodec_mutex = PTHREAD_MUTEX_INITIALIZER;

static k_s32 acodec_open(void)
{
    pthread_mutex_lock(&g_acodec_mutex);
    if (g_acodec_fd < 0)
    {
        g_acodec_fd = open("/dev/acodec_device", O_RDWR);
    }
    pthread_mutex_unlock(&g_acodec_mutex);

    if (g_acodec_fd < 0)
    {
        perror("open /dev/acodec_device");
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_bool wait_for_continue(void)
{
    int key;

    printf("input c to continue or q to return\n");
    while (!exit_requested() && (key = getchar()) != EOF)
    {
        if (key == 'c')
        {
            return K_TRUE;
        }
        if (key == 'q')
        {
            return K_FALSE;
        }
    }
    return K_FALSE;
}

static void test_adc_mic_gain(void)
{
    static const k_u32 gains[] = {0, 6, 20, 30};
    size_t index = 0;

    do
    {
        k_u32 gain = gains[index++ % (sizeof(gains) / sizeof(gains[0]))];
        ioctl(g_acodec_fd, k_acodec_set_gain_micl, &gain);
        ioctl(g_acodec_fd, k_acodec_set_gain_micr, &gain);
        printf("ADC microphone gain: %u dB\n", gain);
    } while (wait_for_continue());
}

static void test_adc_volume(void)
{
    float volume = -97.0f;

    do
    {
        volume += 0.5f;
        if (volume > 30.0f)
        {
            volume = -96.5f;
        }
        ioctl(g_acodec_fd, k_acodec_set_adcl_volume, &volume);
        ioctl(g_acodec_fd, k_acodec_set_adcr_volume, &volume);
        printf("ADC volume: %.2f dB\n", volume);
    } while (wait_for_continue());
}

static void test_alc_gain(void)
{
    float gain = -18.0f;

    do
    {
        gain += 1.5f;
        if (gain > 28.5f)
        {
            gain = -16.5f;
        }
        ioctl(g_acodec_fd, k_acodec_set_alc_gain_micl, &gain);
        ioctl(g_acodec_fd, k_acodec_set_alc_gain_micr, &gain);
        printf("ALC microphone gain: %.2f dB\n", gain);
    } while (wait_for_continue());
}

static void test_dac_gain(void)
{
    float gain = -39.0f;

    do
    {
        gain += 1.5f;
        if (gain > 6.0f)
        {
            gain = -37.5f;
        }
        ioctl(g_acodec_fd, k_acodec_set_gain_hpoutl, &gain);
        ioctl(g_acodec_fd, k_acodec_set_gain_hpoutr, &gain);
        printf("DAC headphone gain: %.2f dB\n", gain);
    } while (wait_for_continue());
}

static void test_dac_volume(void)
{
    float volume = -120.0f;

    do
    {
        volume += 0.5f;
        if (volume > 7.0f)
        {
            volume = -119.5f;
        }
        ioctl(g_acodec_fd, k_acodec_set_dacl_volume, &volume);
        ioctl(g_acodec_fd, k_acodec_set_dacr_volume, &volume);
        printf("DAC volume: %.2f dB\n", volume);
    } while (wait_for_continue());
}

static void test_adc_mute(void)
{
    k_bool mute = K_TRUE;

    do
    {
        ioctl(g_acodec_fd, k_acodec_set_micl_mute, &mute);
        ioctl(g_acodec_fd, k_acodec_set_micr_mute, &mute);
        printf("ADC mute: %d\n", mute);
        mute = !mute;
    } while (wait_for_continue());
}

static void test_dac_mute(void)
{
    k_bool mute = K_TRUE;

    do
    {
        ioctl(g_acodec_fd, k_acodec_set_dacl_mute, &mute);
        ioctl(g_acodec_fd, k_acodec_set_dacr_mute, &mute);
        printf("DAC mute: %d\n", mute);
        mute = !mute;
    } while (wait_for_continue());
}

static void print_acodec_values(void)
{
    k_u32 mic_left;
    k_u32 mic_right;
    float adc_left;
    float adc_right;
    float alc_left;
    float alc_right;
    float dac_left;
    float dac_right;
    float hp_left;
    float hp_right;

    ioctl(g_acodec_fd, k_acodec_get_gain_micl, &mic_left);
    ioctl(g_acodec_fd, k_acodec_get_gain_micr, &mic_right);
    ioctl(g_acodec_fd, k_acodec_get_adcl_volume, &adc_left);
    ioctl(g_acodec_fd, k_acodec_get_adcr_volume, &adc_right);
    ioctl(g_acodec_fd, k_acodec_get_alc_gain_micl, &alc_left);
    ioctl(g_acodec_fd, k_acodec_get_alc_gain_micr, &alc_right);
    ioctl(g_acodec_fd, k_acodec_get_dacl_volume, &dac_left);
    ioctl(g_acodec_fd, k_acodec_get_dacr_volume, &dac_right);
    ioctl(g_acodec_fd, k_acodec_get_gain_hpoutl, &hp_left);
    ioctl(g_acodec_fd, k_acodec_get_gain_hpoutr, &hp_right);

    printf("mic=(%u,%u) adc=(%.1f,%.1f) alc=(%.1f,%.1f) "
           "dac=(%.1f,%.1f) headphone=(%.1f,%.1f)\n",
           mic_left, mic_right, adc_left, adc_right, alc_left, alc_right,
           dac_left, dac_right, hp_left, hp_right);
}

static void show_acodec_menu(void)
{
    printf("\n0: ADC microphone gain\n"
           "1: ADC volume\n"
           "2: ALC microphone gain\n"
           "3: DAC headphone gain\n"
           "4: DAC volume\n"
           "5: ADC mute\n"
           "6: DAC mute\n"
           "7: show current values\n"
           "8: reset codec\n"
           "q: exit\n");
}

k_s32 audio_sample_acodec(void)
{
    int key;

    if (acodec_open() != K_SUCCESS)
    {
        return K_FAILED;
    }

    while (!exit_requested())
    {
        show_acodec_menu();
        key = getchar();
        if (key == EOF || key == 'q')
        {
            break;
        }
        switch (key)
        {
        case '0':
            test_adc_mic_gain();
            break;
        case '1':
            test_adc_volume();
            break;
        case '2':
            test_alc_gain();
            break;
        case '3':
            test_dac_gain();
            break;
        case '4':
            test_dac_volume();
            break;
        case '5':
            test_adc_mute();
            break;
        case '6':
            test_dac_mute();
            break;
        case '7':
            print_acodec_values();
            break;
        case '8':
            ioctl(g_acodec_fd, k_acodec_reset, NULL);
            break;
        default:
            break;
        }
    }

    if (close(g_acodec_fd) != 0)
    {
        perror("close /dev/acodec_device");
        g_acodec_fd = -1;
        return K_FAILED;
    }
    g_acodec_fd = -1;
    return K_SUCCESS;
}
