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

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <unistd.h>

#include "audio_sample.h"
#include "audio_file.h"
#include "mpi_sys_api.h"
#include "sample_audio_config.h"
#include "sample_audio_modes.h"

typedef struct
{
    const sample_audio_config *config;
    k_s32 result;
    k_bool finished;
    pthread_mutex_t mutex;
} sample_thread_context;

static volatile sig_atomic_t g_signal_received;

static void set_module_log(k_mod_id module, k_s32 level)
{
    k_log_level_conf config;

    config.mod_id = module;
    config.level = level;
    kd_mpi_log_set_level_conf(&config);
}

static k_bool mode_uses_audio_output(sample_audio_mode mode)
{
    return mode == SAMPLE_AUDIO_MODE_PLAY_WAV ||
           mode == SAMPLE_AUDIO_MODE_LOOP_PCM ||
           mode == SAMPLE_AUDIO_MODE_BIND_PCM ||
           mode == SAMPLE_AUDIO_MODE_PLAY_G711_BIND ||
           mode == SAMPLE_AUDIO_MODE_PLAY_G711 ||
           mode == SAMPLE_AUDIO_MODE_DUPLEX_G711 ||
           mode == SAMPLE_AUDIO_MODE_LOOP_G711 ||
           mode == SAMPLE_AUDIO_MODE_LOOP_OPUS;
}

static void show_configuration(const sample_audio_config *config)
{
    printf("mode: %d (%s)\n", config->mode,
           sample_audio_mode_name(config->mode));
    if (sample_audio_mode_uses_capture(config->mode))
    {
        printf("source: %s\n", sample_audio_source_name(config->source));
    }
    if (sample_audio_mode_uses_input(config->mode))
    {
        printf("input: %s\n", config->input_file);
    }
    if (sample_audio_mode_uses_output(config->mode))
    {
        printf("output: %s\n", config->output_file);
    }
    if (config->mode == SAMPLE_AUDIO_MODE_PLAY_WAV)
    {
        printf("audio: format-from-wav codec=%s\n",
               config->enable_codec ? "internal" : "external");
        return;
    }
    printf("audio: rate=%u bits=%d", config->sample_rate,
           config->bit_width_bits);
    if (config->source == SAMPLE_AUDIO_SOURCE_I2S ||
        mode_uses_audio_output(config->mode))
    {
        printf(" codec=%s", config->enable_codec ? "internal" : "external");
    }
    if (config->mode == SAMPLE_AUDIO_MODE_RECORD_WAV)
    {
        printf(" channels=%u", config->channels);
    }
    printf("\n");
}

static void resolve_default_input(sample_audio_config *config)
{
    char latest[SAMPLE_AUDIO_FILENAME_SIZE];

    if (!config->input_is_default ||
        !sample_audio_mode_uses_input(config->mode))
    {
        return;
    }
    if (audio_file_find_latest(config->input_file, latest, sizeof(latest)) ==
        K_SUCCESS)
    {
        snprintf(config->input_file, sizeof(config->input_file), "%s", latest);
    }
}

static void *sample_thread(void *arg)
{
    sample_thread_context *context = arg;

    context->result = sample_audio_run_mode(context->config);
    pthread_mutex_lock(&context->mutex);
    context->finished = K_TRUE;
    pthread_mutex_unlock(&context->mutex);
    return NULL;
}

static k_bool sample_thread_finished(sample_thread_context *context)
{
    k_bool finished;

    pthread_mutex_lock(&context->mutex);
    finished = context->finished;
    pthread_mutex_unlock(&context->mutex);
    return finished;
}

static k_bool join_sample_thread(pthread_t thread)
{
    int result = pthread_join(thread, NULL);

    if (result != 0)
    {
        printf("pthread_join failed: %s\n", strerror(result));
        return K_FALSE;
    }
    return K_TRUE;
}

static void signal_handler(int signal_number)
{
    (void)signal_number;
    g_signal_received = 1;
    audio_sample_exit();
}

static k_s32 install_signal_handler(void)
{
    struct sigaction action;

    memset(&action, 0, sizeof(action));
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) != 0 ||
        sigaction(SIGTERM, &action, NULL) != 0)
    {
        perror("sigaction");
        return K_FAILED;
    }
    return K_SUCCESS;
}

static void wait_for_sample(sample_thread_context *context,
                            sample_audio_mode mode)
{
    k_bool interactive = sample_audio_mode_is_interactive(mode);

    if (interactive)
    {
        printf(mode == SAMPLE_AUDIO_MODE_PLAY_WAV
                   ? "press q or Ctrl-C to stop early\n"
                   : "press q or Ctrl-C to stop\n");
    }

    while (!g_signal_received && !sample_thread_finished(context))
    {
        if (!interactive)
        {
            usleep(100000);
            continue;
        }

        fd_set read_fds;
        struct timeval timeout;
        int select_ret;

        FD_ZERO(&read_fds);
        FD_SET(STDIN_FILENO, &read_fds);
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000;
        select_ret = select(STDIN_FILENO + 1, &read_fds, NULL, NULL, &timeout);
        if (select_ret > 0 && FD_ISSET(STDIN_FILENO, &read_fds))
        {
            int key = getchar();
            if (key == 'q' || key == EOF)
            {
                break;
            }
        }
        else if (select_ret < 0 && errno != EINTR)
        {
            perror("select");
            break;
        }
    }

    audio_sample_exit();
}

int main(int argc, char **argv)
{
    sample_audio_config config;
    sample_thread_context thread_context;
    pthread_t thread;
    k_bool vb_initialized = K_FALSE;
    k_bool thread_created = K_FALSE;
    k_bool mutex_initialized = K_FALSE;
    sample_audio_parse_result parse_ret;
    int exit_code = 1;

    parse_ret = sample_audio_parse_arguments(argc, argv, &config);
    if (parse_ret == SAMPLE_AUDIO_PARSE_HELP)
    {
        sample_audio_show_help(argv[0]);
        return 0;
    }
    if (parse_ret != SAMPLE_AUDIO_PARSE_OK)
    {
        sample_audio_show_help(argv[0]);
        return 1;
    }

    resolve_default_input(&config);

    audio_sample_reset();
    g_signal_received = 0;
    if (install_signal_handler() != K_SUCCESS)
    {
        return 1;
    }

    if (!sample_audio_mode_uses_vb(config.mode))
    {
        return sample_audio_run_mode(&config) == K_SUCCESS ? 0 : 1;
    }

    set_module_log(K_ID_AO, config.log_level);
    set_module_log(K_ID_AI, config.log_level);
    set_module_log(K_ID_AENC, config.log_level);
    set_module_log(K_ID_ADEC, config.log_level);
    show_configuration(&config);

    if (audio_sample_vb_init() != K_SUCCESS)
    {
        goto cleanup;
    }
    vb_initialized = K_TRUE;
    if (g_signal_received)
    {
        exit_code = 0;
        goto cleanup;
    }

    memset(&thread_context, 0, sizeof(thread_context));
    thread_context.config = &config;
    thread_context.result = K_FAILED;
    if (pthread_mutex_init(&thread_context.mutex, NULL) != 0)
    {
        printf("pthread_mutex_init failed\n");
        goto cleanup;
    }
    mutex_initialized = K_TRUE;
    if (pthread_create(&thread, NULL, sample_thread, &thread_context) != 0)
    {
        perror("pthread_create");
        goto cleanup;
    }
    thread_created = K_TRUE;

    wait_for_sample(&thread_context, config.mode);
    if (!join_sample_thread(thread))
    {
        goto cleanup;
    }
    thread_created = K_FALSE;
    exit_code = thread_context.result == K_SUCCESS ? 0 : 1;

cleanup:
    audio_sample_exit();
    if (thread_created)
    {
        if (join_sample_thread(thread))
        {
            thread_created = K_FALSE;
        }
        else
        {
            exit_code = 1;
        }
    }
    if (!thread_created && mutex_initialized)
    {
        if (pthread_mutex_destroy(&thread_context.mutex) != 0)
        {
            printf("pthread_mutex_destroy failed\n");
            exit_code = 1;
        }
    }
    if (!thread_created && vb_initialized &&
        audio_sample_vb_destroy() != K_SUCCESS)
    {
        exit_code = 1;
    }
    printf("sample %s\n", exit_code == 0 ? "done" : "failed");
    return exit_code;
}
