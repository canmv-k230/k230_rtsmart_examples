#include "audio_wav.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_file.h"

struct audio_wav_reader
{
    FILE *file;
    audio_wav_format format;
    k_u32 data_size;
    k_u32 position;
};

struct audio_wav_writer
{
    FILE *file;
    audio_wav_format format;
    k_u32 data_size;
    char path[AUDIO_FILE_PATH_SIZE];
};

static k_u16 read_le16(const k_u8 *data)
{
    return (k_u16)data[0] | ((k_u16)data[1] << 8);
}

static k_u32 read_le32(const k_u8 *data)
{
    return (k_u32)data[0] | ((k_u32)data[1] << 8) |
           ((k_u32)data[2] << 16) | ((k_u32)data[3] << 24);
}

static void write_le16(k_u8 *data, k_u16 value)
{
    data[0] = (k_u8)value;
    data[1] = (k_u8)(value >> 8);
}

static void write_le32(k_u8 *data, k_u32 value)
{
    data[0] = (k_u8)value;
    data[1] = (k_u8)(value >> 8);
    data[2] = (k_u8)(value >> 16);
    data[3] = (k_u8)(value >> 24);
}

static k_bool format_is_valid(const audio_wav_format *format)
{
    k_u64 block_align;
    k_u64 byte_rate;

    if (format == NULL ||
        (format->channel_count != 1 && format->channel_count != 2) ||
        format->sample_rate == 0 ||
        (format->bits_per_sample != 16 &&
         format->bits_per_sample != 24 &&
         format->bits_per_sample != 32))
    {
        return K_FALSE;
    }

    block_align = (k_u64)format->channel_count * format->bits_per_sample / 8;
    byte_rate = (k_u64)format->sample_rate * block_align;
    return block_align <= UINT16_MAX && byte_rate <= UINT32_MAX;
}

k_s32 audio_wav_format_byte_rate(const audio_wav_format *format,
                                 k_u32 *byte_rate)
{
    if (!format_is_valid(format) || byte_rate == NULL)
    {
        return K_FAILED;
    }
    *byte_rate = format->sample_rate *
                 (format->channel_count * format->bits_per_sample / 8U);
    return K_SUCCESS;
}

k_s32 audio_wav_format_frame_size(const audio_wav_format *format,
                                  k_u32 frames_per_second,
                                  k_u32 *frame_size)
{
    k_u32 block_align;

    if (!format_is_valid(format) || frames_per_second == 0 ||
        frame_size == NULL || format->sample_rate % frames_per_second != 0)
    {
        return K_FAILED;
    }
    block_align = format->channel_count * format->bits_per_sample / 8U;
    *frame_size = format->sample_rate / frames_per_second * block_align;
    return K_SUCCESS;
}

static k_s32 get_file_size(FILE *file, long *file_size)
{
    long current = ftell(file);

    if (current < 0 || fseek(file, 0, SEEK_END) != 0)
    {
        return K_FAILED;
    }
    *file_size = ftell(file);
    if (*file_size < 0 || fseek(file, current, SEEK_SET) != 0)
    {
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 validate_pcm_format(const k_u8 *fmt, audio_wav_format *format)
{
    k_u16 block_align;
    k_u32 byte_rate;
    k_u64 expected_byte_rate;

    if (read_le16(fmt) != 1)
    {
        return K_FAILED;
    }
    format->channel_count = read_le16(fmt + 2);
    format->sample_rate = read_le32(fmt + 4);
    byte_rate = read_le32(fmt + 8);
    block_align = read_le16(fmt + 12);
    format->bits_per_sample = read_le16(fmt + 14);
    if (!format_is_valid(format) ||
        block_align != format->channel_count * format->bits_per_sample / 8)
    {
        return K_FAILED;
    }
    expected_byte_rate = (k_u64)format->sample_rate * block_align;
    return byte_rate == expected_byte_rate ? K_SUCCESS : K_FAILED;
}

static k_s32 parse_wav(audio_wav_reader *reader)
{
    k_u8 riff_header[12];
    audio_wav_format format;
    long file_size;
    long riff_end;
    long data_offset = 0;
    k_u32 data_size = 0;
    k_bool format_found = K_FALSE;
    k_bool data_found = K_FALSE;

    if (get_file_size(reader->file, &file_size) != K_SUCCESS ||
        file_size < (long)sizeof(riff_header) ||
        fread(riff_header, 1, sizeof(riff_header), reader->file) !=
            sizeof(riff_header) ||
        memcmp(riff_header, "RIFF", 4) != 0 ||
        memcmp(riff_header + 8, "WAVE", 4) != 0 ||
        read_le32(riff_header + 4) < 4 ||
        (k_u64)read_le32(riff_header + 4) + 8 > (k_u64)file_size)
    {
        return K_FAILED;
    }
    riff_end = (long)read_le32(riff_header + 4) + 8;

    while (ftell(reader->file) >= 0 && ftell(reader->file) + 8 <= riff_end)
    {
        k_u8 chunk_header[8];
        k_u32 chunk_size;
        long chunk_offset;
        long next_chunk;

        if (fread(chunk_header, 1, sizeof(chunk_header), reader->file) !=
            sizeof(chunk_header))
        {
            return K_FAILED;
        }
        chunk_size = read_le32(chunk_header + 4);
        chunk_offset = ftell(reader->file);
        if (chunk_offset < 0 ||
            (k_u64)chunk_size > (k_u64)(riff_end - chunk_offset))
        {
            return K_FAILED;
        }

        if (memcmp(chunk_header, "fmt ", 4) == 0)
        {
            k_u8 fmt[16];

            if (chunk_size < sizeof(fmt) ||
                fread(fmt, 1, sizeof(fmt), reader->file) != sizeof(fmt) ||
                validate_pcm_format(fmt, &format) != K_SUCCESS)
            {
                return K_FAILED;
            }
            format_found = K_TRUE;
        }
        else if (memcmp(chunk_header, "data", 4) == 0)
        {
            data_offset = chunk_offset;
            data_size = chunk_size;
            data_found = K_TRUE;
        }

        if (format_found && data_found)
        {
            k_u16 block_align =
                format.channel_count * format.bits_per_sample / 8;

            if (data_size % block_align != 0)
            {
                return K_FAILED;
            }
            reader->format = format;
            reader->data_size = data_size;
            reader->position = 0;
            return fseek(reader->file, data_offset, SEEK_SET) == 0
                       ? K_SUCCESS
                       : K_FAILED;
        }

        next_chunk = chunk_offset + chunk_size + (chunk_size & 1U);
        if (next_chunk > riff_end ||
            fseek(reader->file, next_chunk, SEEK_SET) != 0)
        {
            return K_FAILED;
        }
    }

    return K_FAILED;
}

static void build_wav_header(const audio_wav_format *format, k_u32 data_size,
                             k_u8 header[AUDIO_WAV_HEADER_SIZE])
{
    k_u16 block_align = format->channel_count * format->bits_per_sample / 8;
    k_u32 byte_rate = 0;
    k_u32 padding = data_size & 1U;

    audio_wav_format_byte_rate(format, &byte_rate);

    memset(header, 0, AUDIO_WAV_HEADER_SIZE);
    memcpy(header, "RIFF", 4);
    write_le32(header + 4, 36U + data_size + padding);
    memcpy(header + 8, "WAVE", 4);
    memcpy(header + 12, "fmt ", 4);
    write_le32(header + 16, 16);
    write_le16(header + 20, 1);
    write_le16(header + 22, format->channel_count);
    write_le32(header + 24, format->sample_rate);
    write_le32(header + 28, byte_rate);
    write_le16(header + 32, block_align);
    write_le16(header + 34, format->bits_per_sample);
    memcpy(header + 36, "data", 4);
    write_le32(header + 40, data_size);
}

k_s32 audio_wav_reader_open(audio_wav_reader **reader, const char *filename)
{
    audio_wav_reader *context;

    if (reader == NULL || filename == NULL)
    {
        return K_FAILED;
    }
    *reader = NULL;
    context = calloc(1, sizeof(*context));
    if (context == NULL)
    {
        return K_FAILED;
    }
    context->file = fopen(filename, "rb");
    if (context->file == NULL || parse_wav(context) != K_SUCCESS)
    {
        if (context->file != NULL)
        {
            fclose(context->file);
        }
        free(context);
        return K_FAILED;
    }

    *reader = context;
    return K_SUCCESS;
}

k_s32 audio_wav_reader_get_format(const audio_wav_reader *reader,
                                  audio_wav_format *format)
{
    if (reader == NULL || format == NULL)
    {
        return K_FAILED;
    }
    *format = reader->format;
    return K_SUCCESS;
}

k_u32 audio_wav_reader_data_size(const audio_wav_reader *reader)
{
    return reader == NULL ? 0 : reader->data_size;
}

k_s32 audio_wav_reader_read(audio_wav_reader *reader, void *data,
                            k_u32 capacity, k_u32 *bytes_read)
{
    k_u32 remaining;
    k_u32 read_size;
    size_t result;

    if (reader == NULL || bytes_read == NULL ||
        (data == NULL && capacity != 0))
    {
        return K_FAILED;
    }
    *bytes_read = 0;
    remaining = reader->data_size - reader->position;
    read_size = capacity < remaining ? capacity : remaining;
    if (read_size == 0)
    {
        return K_SUCCESS;
    }

    result = fread(data, 1, read_size, reader->file);
    reader->position += result;
    *bytes_read = result;
    return result == read_size ? K_SUCCESS : K_FAILED;
}

k_s32 audio_wav_reader_close(audio_wav_reader **reader)
{
    k_s32 ret = K_SUCCESS;

    if (reader == NULL || *reader == NULL)
    {
        return K_SUCCESS;
    }
    if (fclose((*reader)->file) != 0)
    {
        ret = K_FAILED;
    }
    free(*reader);
    *reader = NULL;
    return ret;
}

k_s32 audio_wav_writer_open(audio_wav_writer **writer, const char *filename,
                            const audio_wav_format *format)
{
    audio_wav_writer *context;
    k_u8 header[AUDIO_WAV_HEADER_SIZE];

    if (writer == NULL || filename == NULL || !format_is_valid(format))
    {
        return K_FAILED;
    }
    *writer = NULL;
    context = calloc(1, sizeof(*context));
    if (context == NULL)
    {
        return K_FAILED;
    }
    context->format = *format;
    build_wav_header(format, 0, header);
    if (audio_file_open_unique(filename, "wb+", &context->file,
                               context->path, sizeof(context->path)) !=
            K_SUCCESS ||
        fwrite(header, 1, sizeof(header), context->file) != sizeof(header))
    {
        if (context->file != NULL)
        {
            fclose(context->file);
            remove(context->path);
        }
        free(context);
        return K_FAILED;
    }

    *writer = context;
    return K_SUCCESS;
}

k_s32 audio_wav_writer_write(audio_wav_writer *writer, const void *data,
                             k_u32 size)
{
    size_t written;

    if (writer == NULL || (data == NULL && size != 0) ||
        size > AUDIO_WAV_MAX_DATA_SIZE - writer->data_size)
    {
        return K_FAILED;
    }
    if (size == 0)
    {
        return K_SUCCESS;
    }

    written = fwrite(data, 1, size, writer->file);
    writer->data_size += written;
    return written == size ? K_SUCCESS : K_FAILED;
}

k_u32 audio_wav_writer_data_size(const audio_wav_writer *writer)
{
    return writer == NULL ? 0 : writer->data_size;
}

const char *audio_wav_writer_path(const audio_wav_writer *writer)
{
    return writer == NULL ? NULL : writer->path;
}

k_s32 audio_wav_writer_close(audio_wav_writer **writer)
{
    audio_wav_writer *context;
    k_u8 header[AUDIO_WAV_HEADER_SIZE];
    k_s32 ret = K_SUCCESS;

    if (writer == NULL || *writer == NULL)
    {
        return K_SUCCESS;
    }
    context = *writer;
    if (context->data_size %
            (context->format.channel_count *
             context->format.bits_per_sample / 8) != 0)
    {
        ret = K_FAILED;
    }
    if ((context->data_size & 1U) != 0)
    {
        const k_u8 padding = 0;
        if (fwrite(&padding, 1, 1, context->file) != 1)
        {
            ret = K_FAILED;
        }
    }
    build_wav_header(&context->format, context->data_size, header);
    if (fseek(context->file, 0, SEEK_SET) != 0 ||
        fwrite(header, 1, sizeof(header), context->file) != sizeof(header) ||
        fflush(context->file) != 0)
    {
        ret = K_FAILED;
    }
    if (fclose(context->file) != 0)
    {
        ret = K_FAILED;
    }
    free(context);
    *writer = NULL;
    return ret;
}

k_s32 audio_wav_writer_discard(audio_wav_writer **writer)
{
    audio_wav_writer *context;
    k_s32 ret;

    if (writer == NULL || *writer == NULL)
    {
        return K_SUCCESS;
    }
    context = *writer;
    ret = audio_file_discard(&context->file, context->path);
    free(context);
    *writer = NULL;
    return ret;
}
