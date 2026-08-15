#ifndef SAMPLE_AUDIO_AUDIO_WAV_H
#define SAMPLE_AUDIO_AUDIO_WAV_H

#ifdef __cplusplus
extern "C" {
#endif

#include "k_type.h"

#define AUDIO_WAV_HEADER_SIZE 44U
#define AUDIO_WAV_MAX_DATA_SIZE ((k_u32)0xffffffdaU)

typedef struct
{
    k_u16 channel_count;
    k_u32 sample_rate;
    k_u16 bits_per_sample;
} audio_wav_format;

typedef struct audio_wav_reader audio_wav_reader;
typedef struct audio_wav_writer audio_wav_writer;

k_s32 audio_wav_format_byte_rate(const audio_wav_format *format,
                                 k_u32 *byte_rate);
k_s32 audio_wav_format_frame_size(const audio_wav_format *format,
                                  k_u32 frames_per_second,
                                  k_u32 *frame_size);

/* Reader and writer handles own their FILE and are independent of each other. */
k_s32 audio_wav_reader_open(audio_wav_reader **reader, const char *filename);
k_s32 audio_wav_reader_get_format(const audio_wav_reader *reader,
                                  audio_wav_format *format);
k_u32 audio_wav_reader_data_size(const audio_wav_reader *reader);
k_s32 audio_wav_reader_read(audio_wav_reader *reader, void *data,
                            k_u32 capacity, k_u32 *bytes_read);
k_s32 audio_wav_reader_close(audio_wav_reader **reader);

k_s32 audio_wav_writer_open(audio_wav_writer **writer, const char *filename,
                            const audio_wav_format *format);
k_s32 audio_wav_writer_write(audio_wav_writer *writer, const void *data,
                             k_u32 size);
k_u32 audio_wav_writer_data_size(const audio_wav_writer *writer);
const char *audio_wav_writer_path(const audio_wav_writer *writer);
/* Finalizes the RIFF sizes, closes the file, and clears the handle. */
k_s32 audio_wav_writer_close(audio_wav_writer **writer);
/* Closes the writer and removes its incomplete output file. */
k_s32 audio_wav_writer_discard(audio_wav_writer **writer);

#ifdef __cplusplus
}
#endif

#endif
