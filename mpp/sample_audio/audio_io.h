#ifndef SAMPLE_AUDIO_AUDIO_IO_H
#define SAMPLE_AUDIO_AUDIO_IO_H

#ifdef __cplusplus
extern "C" {
#endif

#include "k_type.h"

#define AUDIO_IO_END 1
#define AUDIO_IO_STOPPED 2

typedef struct audio_io_reader audio_io_reader;
typedef struct audio_io_writer audio_io_writer;

typedef k_bool (*audio_io_stop_fn)(void *context);
typedef k_s32 (*audio_io_read_fn)(void *context, void *data, k_u32 capacity,
                                  k_u32 *bytes_read);
typedef k_s32 (*audio_io_write_fn)(void *context, const void *data,
                                   k_u32 size);

k_s32 audio_io_reader_create(audio_io_reader **reader, k_u32 block_size,
                             k_u32 block_count, audio_io_read_fn read_fn,
                             void *read_context, audio_io_stop_fn stop_fn,
                             void *stop_context);
k_s32 audio_io_reader_pop(audio_io_reader *reader, void *data,
                          k_u32 capacity, k_u32 *bytes_read);
k_s32 audio_io_reader_destroy(audio_io_reader **reader);

k_s32 audio_io_writer_create(audio_io_writer **writer, k_u32 block_size,
                             k_u32 block_count, audio_io_write_fn write_fn,
                             void *write_context, audio_io_stop_fn stop_fn,
                             void *stop_context);
k_s32 audio_io_writer_push(audio_io_writer *writer, const void *data,
                           k_u32 size);
k_s32 audio_io_writer_finish(audio_io_writer **writer,
                             k_u64 *bytes_written);

#ifdef __cplusplus
}
#endif

#endif
