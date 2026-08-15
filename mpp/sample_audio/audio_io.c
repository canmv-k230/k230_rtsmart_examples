#include "audio_io.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define AUDIO_IO_WAIT_MS 100

typedef struct
{
    k_u8 *blocks;
    k_u32 *sizes;
    k_u8 *worker_block;
    k_u32 block_size;
    k_u32 block_count;
    k_u32 read_index;
    k_u32 write_index;
    k_u32 used;
    k_bool closed;
    k_bool cancelled;
    k_bool failed;
    pthread_mutex_t mutex;
    pthread_cond_t can_read;
    pthread_cond_t can_write;
} audio_io_ring;

struct audio_io_reader
{
    audio_io_ring ring;
    audio_io_read_fn read_fn;
    void *read_context;
    audio_io_stop_fn stop_fn;
    void *stop_context;
    pthread_t thread;
};

struct audio_io_writer
{
    audio_io_ring ring;
    audio_io_write_fn write_fn;
    void *write_context;
    audio_io_stop_fn stop_fn;
    void *stop_context;
    pthread_t thread;
    k_u64 bytes_written;
};

static k_bool stop_requested(audio_io_stop_fn stop_fn, void *context)
{
    return stop_fn != NULL && stop_fn(context);
}

static int wait_for_condition(pthread_cond_t *condition,
                              pthread_mutex_t *mutex)
{
    struct timespec deadline;

    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
    {
        return errno;
    }
    deadline.tv_nsec += AUDIO_IO_WAIT_MS * 1000000L;
    if (deadline.tv_nsec >= 1000000000L)
    {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    return pthread_cond_timedwait(condition, mutex, &deadline);
}

static k_s32 ring_init(audio_io_ring *ring, k_u32 block_size,
                       k_u32 block_count)
{
    k_bool mutex_ready = K_FALSE;
    k_bool can_read_ready = K_FALSE;

    if (block_size == 0 || block_count < 2 ||
        block_size > SIZE_MAX / block_count)
    {
        return K_FAILED;
    }
    memset(ring, 0, sizeof(*ring));
    ring->blocks = malloc((size_t)block_size * block_count);
    ring->sizes = calloc(block_count, sizeof(*ring->sizes));
    ring->worker_block = malloc(block_size);
    if (ring->blocks == NULL || ring->sizes == NULL ||
        ring->worker_block == NULL)
    {
        goto fail;
    }
    ring->block_size = block_size;
    ring->block_count = block_count;
    if (pthread_mutex_init(&ring->mutex, NULL) != 0)
    {
        goto fail;
    }
    mutex_ready = K_TRUE;
    if (pthread_cond_init(&ring->can_read, NULL) != 0)
    {
        goto fail;
    }
    can_read_ready = K_TRUE;
    if (pthread_cond_init(&ring->can_write, NULL) != 0)
    {
        goto fail;
    }
    return K_SUCCESS;

fail:
    if (can_read_ready)
    {
        pthread_cond_destroy(&ring->can_read);
    }
    if (mutex_ready)
    {
        pthread_mutex_destroy(&ring->mutex);
    }
    free(ring->worker_block);
    free(ring->sizes);
    free(ring->blocks);
    memset(ring, 0, sizeof(*ring));
    return K_FAILED;
}

static k_s32 ring_destroy(audio_io_ring *ring)
{
    k_s32 ret = K_SUCCESS;

    if (pthread_cond_destroy(&ring->can_write) != 0)
    {
        ret = K_FAILED;
    }
    if (pthread_cond_destroy(&ring->can_read) != 0)
    {
        ret = K_FAILED;
    }
    if (pthread_mutex_destroy(&ring->mutex) != 0)
    {
        ret = K_FAILED;
    }
    free(ring->worker_block);
    free(ring->sizes);
    free(ring->blocks);
    memset(ring, 0, sizeof(*ring));
    return ret;
}

static void ring_fail(audio_io_ring *ring)
{
    pthread_mutex_lock(&ring->mutex);
    ring->failed = K_TRUE;
    ring->closed = K_TRUE;
    pthread_cond_broadcast(&ring->can_read);
    pthread_cond_broadcast(&ring->can_write);
    pthread_mutex_unlock(&ring->mutex);
}

static void *reader_thread(void *argument)
{
    audio_io_reader *reader = argument;
    audio_io_ring *ring = &reader->ring;

    while (!stop_requested(reader->stop_fn, reader->stop_context))
    {
        k_u32 bytes_read = 0;

        if (reader->read_fn(reader->read_context, ring->worker_block,
                            ring->block_size, &bytes_read) != K_SUCCESS ||
            bytes_read > ring->block_size)
        {
            ring_fail(ring);
            return NULL;
        }
        if (bytes_read == 0)
        {
            break;
        }

        pthread_mutex_lock(&ring->mutex);
        while (ring->used == ring->block_count && !ring->cancelled &&
               !stop_requested(reader->stop_fn, reader->stop_context))
        {
            int wait_result = wait_for_condition(&ring->can_write,
                                                 &ring->mutex);
            if (wait_result != 0 && wait_result != ETIMEDOUT)
            {
                pthread_mutex_unlock(&ring->mutex);
                ring_fail(ring);
                return NULL;
            }
        }
        if (ring->cancelled ||
            stop_requested(reader->stop_fn, reader->stop_context))
        {
            pthread_mutex_unlock(&ring->mutex);
            break;
        }
        memcpy(ring->blocks + (size_t)ring->write_index * ring->block_size,
               ring->worker_block, bytes_read);
        ring->sizes[ring->write_index] = bytes_read;
        ring->write_index = (ring->write_index + 1U) % ring->block_count;
        ring->used++;
        pthread_cond_signal(&ring->can_read);
        pthread_mutex_unlock(&ring->mutex);
    }

    pthread_mutex_lock(&ring->mutex);
    ring->closed = K_TRUE;
    pthread_cond_broadcast(&ring->can_read);
    pthread_cond_broadcast(&ring->can_write);
    pthread_mutex_unlock(&ring->mutex);
    return NULL;
}

k_s32 audio_io_reader_create(audio_io_reader **reader, k_u32 block_size,
                             k_u32 block_count, audio_io_read_fn read_fn,
                             void *read_context, audio_io_stop_fn stop_fn,
                             void *stop_context)
{
    audio_io_reader *context;

    if (reader == NULL || read_fn == NULL)
    {
        return K_FAILED;
    }
    *reader = NULL;
    context = calloc(1, sizeof(*context));
    if (context == NULL ||
        ring_init(&context->ring, block_size, block_count) != K_SUCCESS)
    {
        free(context);
        return K_FAILED;
    }
    context->read_fn = read_fn;
    context->read_context = read_context;
    context->stop_fn = stop_fn;
    context->stop_context = stop_context;
    if (pthread_create(&context->thread, NULL, reader_thread, context) != 0)
    {
        ring_destroy(&context->ring);
        free(context);
        return K_FAILED;
    }
    *reader = context;
    return K_SUCCESS;
}

k_s32 audio_io_reader_pop(audio_io_reader *reader, void *data,
                          k_u32 capacity, k_u32 *bytes_read)
{
    audio_io_ring *ring;

    if (reader == NULL || data == NULL || bytes_read == NULL)
    {
        return K_FAILED;
    }
    *bytes_read = 0;
    ring = &reader->ring;
    pthread_mutex_lock(&ring->mutex);
    while (ring->used == 0 && !ring->closed && !ring->failed &&
           !stop_requested(reader->stop_fn, reader->stop_context))
    {
        int wait_result = wait_for_condition(&ring->can_read, &ring->mutex);
        if (wait_result != 0 && wait_result != ETIMEDOUT)
        {
            pthread_mutex_unlock(&ring->mutex);
            return K_FAILED;
        }
    }
    if (stop_requested(reader->stop_fn, reader->stop_context))
    {
        pthread_mutex_unlock(&ring->mutex);
        return AUDIO_IO_STOPPED;
    }
    if (ring->failed)
    {
        pthread_mutex_unlock(&ring->mutex);
        return K_FAILED;
    }
    if (ring->used == 0 && ring->closed)
    {
        pthread_mutex_unlock(&ring->mutex);
        return AUDIO_IO_END;
    }
    if (ring->sizes[ring->read_index] > capacity)
    {
        pthread_mutex_unlock(&ring->mutex);
        return K_FAILED;
    }
    *bytes_read = ring->sizes[ring->read_index];
    memcpy(data, ring->blocks + (size_t)ring->read_index * ring->block_size,
           *bytes_read);
    ring->read_index = (ring->read_index + 1U) % ring->block_count;
    ring->used--;
    pthread_cond_signal(&ring->can_write);
    pthread_mutex_unlock(&ring->mutex);
    return K_SUCCESS;
}

k_s32 audio_io_reader_destroy(audio_io_reader **reader)
{
    audio_io_reader *context;
    k_s32 ret = K_SUCCESS;

    if (reader == NULL || *reader == NULL)
    {
        return K_SUCCESS;
    }
    context = *reader;
    pthread_mutex_lock(&context->ring.mutex);
    context->ring.cancelled = K_TRUE;
    pthread_cond_broadcast(&context->ring.can_read);
    pthread_cond_broadcast(&context->ring.can_write);
    pthread_mutex_unlock(&context->ring.mutex);
    if (pthread_join(context->thread, NULL) != 0 || context->ring.failed)
    {
        ret = K_FAILED;
    }
    if (ring_destroy(&context->ring) != K_SUCCESS)
    {
        ret = K_FAILED;
    }
    free(context);
    *reader = NULL;
    return ret;
}

static void *writer_thread(void *argument)
{
    audio_io_writer *writer = argument;
    audio_io_ring *ring = &writer->ring;

    while (K_TRUE)
    {
        k_u32 size;

        pthread_mutex_lock(&ring->mutex);
        while (ring->used == 0 && !ring->closed && !ring->cancelled)
        {
            if (pthread_cond_wait(&ring->can_read, &ring->mutex) != 0)
            {
                pthread_mutex_unlock(&ring->mutex);
                ring_fail(ring);
                return NULL;
            }
        }
        if (ring->cancelled || (ring->used == 0 && ring->closed))
        {
            pthread_mutex_unlock(&ring->mutex);
            break;
        }
        size = ring->sizes[ring->read_index];
        memcpy(ring->worker_block,
               ring->blocks + (size_t)ring->read_index * ring->block_size,
               size);
        ring->read_index = (ring->read_index + 1U) % ring->block_count;
        ring->used--;
        pthread_cond_signal(&ring->can_write);
        pthread_mutex_unlock(&ring->mutex);

        if (writer->write_fn(writer->write_context, ring->worker_block,
                             size) != K_SUCCESS)
        {
            ring_fail(ring);
            return NULL;
        }
        writer->bytes_written += size;
    }
    return NULL;
}

k_s32 audio_io_writer_create(audio_io_writer **writer, k_u32 block_size,
                             k_u32 block_count, audio_io_write_fn write_fn,
                             void *write_context, audio_io_stop_fn stop_fn,
                             void *stop_context)
{
    audio_io_writer *context;

    if (writer == NULL || write_fn == NULL)
    {
        return K_FAILED;
    }
    *writer = NULL;
    context = calloc(1, sizeof(*context));
    if (context == NULL ||
        ring_init(&context->ring, block_size, block_count) != K_SUCCESS)
    {
        free(context);
        return K_FAILED;
    }
    context->write_fn = write_fn;
    context->write_context = write_context;
    context->stop_fn = stop_fn;
    context->stop_context = stop_context;
    if (pthread_create(&context->thread, NULL, writer_thread, context) != 0)
    {
        ring_destroy(&context->ring);
        free(context);
        return K_FAILED;
    }
    *writer = context;
    return K_SUCCESS;
}

k_s32 audio_io_writer_push(audio_io_writer *writer, const void *data,
                           k_u32 size)
{
    audio_io_ring *ring;

    if (writer == NULL || data == NULL || size == 0 ||
        size > writer->ring.block_size)
    {
        return K_FAILED;
    }
    ring = &writer->ring;
    pthread_mutex_lock(&ring->mutex);
    while (ring->used == ring->block_count && !ring->failed &&
           !stop_requested(writer->stop_fn, writer->stop_context))
    {
        int wait_result = wait_for_condition(&ring->can_write, &ring->mutex);
        if (wait_result != 0 && wait_result != ETIMEDOUT)
        {
            pthread_mutex_unlock(&ring->mutex);
            return K_FAILED;
        }
    }
    if (stop_requested(writer->stop_fn, writer->stop_context))
    {
        pthread_mutex_unlock(&ring->mutex);
        return AUDIO_IO_STOPPED;
    }
    if (ring->failed || ring->closed)
    {
        pthread_mutex_unlock(&ring->mutex);
        return K_FAILED;
    }
    memcpy(ring->blocks + (size_t)ring->write_index * ring->block_size,
           data, size);
    ring->sizes[ring->write_index] = size;
    ring->write_index = (ring->write_index + 1U) % ring->block_count;
    ring->used++;
    pthread_cond_signal(&ring->can_read);
    pthread_mutex_unlock(&ring->mutex);
    return K_SUCCESS;
}

k_s32 audio_io_writer_finish(audio_io_writer **writer,
                             k_u64 *bytes_written)
{
    audio_io_writer *context;
    k_s32 ret = K_SUCCESS;

    if (writer == NULL || *writer == NULL)
    {
        if (bytes_written != NULL)
        {
            *bytes_written = 0;
        }
        return K_SUCCESS;
    }
    context = *writer;
    pthread_mutex_lock(&context->ring.mutex);
    context->ring.closed = K_TRUE;
    pthread_cond_broadcast(&context->ring.can_read);
    pthread_cond_broadcast(&context->ring.can_write);
    pthread_mutex_unlock(&context->ring.mutex);
    if (pthread_join(context->thread, NULL) != 0 || context->ring.failed)
    {
        ret = K_FAILED;
    }
    if (bytes_written != NULL)
    {
        *bytes_written = context->bytes_written;
    }
    if (ring_destroy(&context->ring) != K_SUCCESS)
    {
        ret = K_FAILED;
    }
    free(context);
    *writer = NULL;
    return ret;
}
