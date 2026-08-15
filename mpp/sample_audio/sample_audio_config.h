#ifndef SAMPLE_AUDIO_CONFIG_H
#define SAMPLE_AUDIO_CONFIG_H

#include "k_audio_comm.h"
#include "k_type.h"

#define SAMPLE_AUDIO_FILENAME_SIZE 256

typedef enum
{
    SAMPLE_AUDIO_MODE_NONE = -1,

    SAMPLE_AUDIO_MODE_RECORD_WAV = 0,
    SAMPLE_AUDIO_MODE_PLAY_WAV,

    SAMPLE_AUDIO_MODE_LOOP_PCM,
    SAMPLE_AUDIO_MODE_BIND_PCM,

    SAMPLE_AUDIO_MODE_RECORD_G711_BIND,
    SAMPLE_AUDIO_MODE_PLAY_G711_BIND,
    SAMPLE_AUDIO_MODE_RECORD_G711,
    SAMPLE_AUDIO_MODE_PLAY_G711,
    SAMPLE_AUDIO_MODE_DUPLEX_G711,

    SAMPLE_AUDIO_MODE_LOOP_G711,
    SAMPLE_AUDIO_MODE_LOOP_OPUS,

    SAMPLE_AUDIO_MODE_CODEC_MENU,
    SAMPLE_AUDIO_MODE_COUNT,
} sample_audio_mode;

typedef enum
{
    SAMPLE_AUDIO_GROUP_WAV,
    SAMPLE_AUDIO_GROUP_PCM_LOOPBACK,
    SAMPLE_AUDIO_GROUP_G711,
    SAMPLE_AUDIO_GROUP_CODEC_LOOPBACK,
    SAMPLE_AUDIO_GROUP_CODEC_CONTROL,
    SAMPLE_AUDIO_GROUP_COUNT,
} sample_audio_mode_group;

#define SAMPLE_AUDIO_MODE_MIN SAMPLE_AUDIO_MODE_RECORD_WAV
#define SAMPLE_AUDIO_MODE_MAX SAMPLE_AUDIO_MODE_CODEC_MENU

typedef enum
{
    SAMPLE_AUDIO_SOURCE_I2S = 0,
    SAMPLE_AUDIO_SOURCE_PDM = 1,
} sample_audio_source;

typedef struct
{
    sample_audio_mode mode;
    sample_audio_source source;
    k_u32 sample_rate;
    k_audio_bit_width bit_width;
    int bit_width_bits;
    k_bool enable_codec;
    k_i2s_work_mode i2s_mode;
    k_u32 audio3a;
    k_u32 channels;
    k_i2s_in_mono_channel mono_channel;
    int log_level;
    char input_file[SAMPLE_AUDIO_FILENAME_SIZE];
    char output_file[SAMPLE_AUDIO_FILENAME_SIZE];
    k_bool input_is_default;
} sample_audio_config;

typedef enum
{
    SAMPLE_AUDIO_PARSE_ERROR = -1,
    SAMPLE_AUDIO_PARSE_OK = 0,
    SAMPLE_AUDIO_PARSE_HELP = 1,
} sample_audio_parse_result;

void sample_audio_config_init(sample_audio_config *config);
const char *sample_audio_mode_name(sample_audio_mode mode);
const char *sample_audio_mode_group_name(sample_audio_mode mode);
const char *sample_audio_source_name(sample_audio_source source);
k_bool sample_audio_mode_uses_capture(sample_audio_mode mode);
k_bool sample_audio_mode_uses_input(sample_audio_mode mode);
k_bool sample_audio_mode_uses_output(sample_audio_mode mode);
sample_audio_parse_result sample_audio_parse_arguments(
    int argc, char **argv, sample_audio_config *config);
void sample_audio_show_help(const char *program);

#endif
