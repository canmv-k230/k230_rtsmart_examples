#include "sample_audio_config.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_WAV_FILE "/sdcard/test.wav"
#define DEFAULT_G711_FILE "/sdcard/test.g711a"

#define OPTION_RATE (1U << 0)
#define OPTION_BITS (1U << 1)
#define OPTION_CHANNELS (1U << 2)
#define OPTION_MONO_CHANNEL (1U << 3)
#define OPTION_CODEC (1U << 4)
#define OPTION_I2S_FORMAT (1U << 5)
#define OPTION_AUDIO3A (1U << 6)
#define OPTION_LOG_LEVEL (1U << 7)
#define OPTION_INPUT (1U << 8)
#define OPTION_OUTPUT (1U << 9)
#define OPTION_SOURCE (1U << 10)

typedef struct
{
    sample_audio_mode mode;
    const char *name;
    const char *description;
    sample_audio_mode_group group;
} sample_audio_mode_info;

static const char *const g_mode_groups[] = {
    [SAMPLE_AUDIO_GROUP_WAV] = "WAV file I/O",
    [SAMPLE_AUDIO_GROUP_PCM_LOOPBACK] = "PCM loopback",
    [SAMPLE_AUDIO_GROUP_G711] = "G711 file I/O",
    [SAMPLE_AUDIO_GROUP_CODEC_LOOPBACK] = "Codec loopback",
    [SAMPLE_AUDIO_GROUP_CODEC_CONTROL] = "Codec control",
};

static const sample_audio_mode_info g_modes[] = {
    {SAMPLE_AUDIO_MODE_RECORD_WAV, "record-wav",
     "Record the selected input to WAV for 15 seconds",
     SAMPLE_AUDIO_GROUP_WAV},
    {SAMPLE_AUDIO_MODE_PLAY_WAV, "play-wav",
     "Play a WAV file once through I2S output", SAMPLE_AUDIO_GROUP_WAV},

    {SAMPLE_AUDIO_MODE_LOOP_PCM, "loop-pcm",
     "Selected input to I2S output using frame APIs",
     SAMPLE_AUDIO_GROUP_PCM_LOOPBACK},
    {SAMPLE_AUDIO_MODE_BIND_PCM, "bind-pcm",
     "Bind the selected input directly to I2S output",
     SAMPLE_AUDIO_GROUP_PCM_LOOPBACK},

    {SAMPLE_AUDIO_MODE_RECORD_G711_BIND, "record-g711-bind",
     "Record selected input using bound G711A encoder",
     SAMPLE_AUDIO_GROUP_G711},
    {SAMPLE_AUDIO_MODE_PLAY_G711_BIND, "play-g711-bind",
     "Play G711A using bound decoder and AO", SAMPLE_AUDIO_GROUP_G711},
    {SAMPLE_AUDIO_MODE_RECORD_G711, "record-g711",
     "Record selected input as G711A using frame APIs",
     SAMPLE_AUDIO_GROUP_G711},
    {SAMPLE_AUDIO_MODE_PLAY_G711, "play-g711",
     "Play G711A using frame APIs", SAMPLE_AUDIO_GROUP_G711},
    {SAMPLE_AUDIO_MODE_DUPLEX_G711, "duplex-g711",
     "Play and record G711A concurrently", SAMPLE_AUDIO_GROUP_G711},

    {SAMPLE_AUDIO_MODE_LOOP_G711, "loop-g711",
     "Selected input through G711A encode/decode loopback",
     SAMPLE_AUDIO_GROUP_CODEC_LOOPBACK},
    {SAMPLE_AUDIO_MODE_LOOP_OPUS, "loop-opus",
     "Selected input through Opus loopback at 8000 Hz",
     SAMPLE_AUDIO_GROUP_CODEC_LOOPBACK},

    {SAMPLE_AUDIO_MODE_CODEC_MENU, "codec-menu",
     "Open the audio codec control menu", SAMPLE_AUDIO_GROUP_CODEC_CONTROL},
};

typedef struct
{
    const char *name;
    sample_audio_mode mode;
    sample_audio_source source;
} sample_audio_mode_alias;

static const sample_audio_mode_alias g_mode_aliases[] = {
    {"record-i2s", SAMPLE_AUDIO_MODE_RECORD_WAV, SAMPLE_AUDIO_SOURCE_I2S},
    {"record-pdm", SAMPLE_AUDIO_MODE_RECORD_WAV, SAMPLE_AUDIO_SOURCE_PDM},
    {"loop-i2s", SAMPLE_AUDIO_MODE_LOOP_PCM, SAMPLE_AUDIO_SOURCE_I2S},
    {"loop-pdm", SAMPLE_AUDIO_MODE_LOOP_PCM, SAMPLE_AUDIO_SOURCE_PDM},
    {"bind-i2s", SAMPLE_AUDIO_MODE_BIND_PCM, SAMPLE_AUDIO_SOURCE_I2S},
    {"bind-pdm", SAMPLE_AUDIO_MODE_BIND_PCM, SAMPLE_AUDIO_SOURCE_PDM},
};

static const sample_audio_mode_info *find_mode(sample_audio_mode mode)
{
    for (size_t index = 0; index < sizeof(g_modes) / sizeof(g_modes[0]); ++index)
    {
        if (g_modes[index].mode == mode)
        {
            return &g_modes[index];
        }
    }
    return NULL;
}

const char *sample_audio_mode_name(sample_audio_mode mode)
{
    const sample_audio_mode_info *info = find_mode(mode);

    return info == NULL ? "unknown" : info->name;
}

const char *sample_audio_mode_group_name(sample_audio_mode mode)
{
    const sample_audio_mode_info *info = find_mode(mode);

    if (info == NULL || (unsigned int)info->group >= SAMPLE_AUDIO_GROUP_COUNT)
    {
        return "Unknown";
    }
    return g_mode_groups[info->group];
}

const char *sample_audio_source_name(sample_audio_source source)
{
    switch (source)
    {
    case SAMPLE_AUDIO_SOURCE_I2S:
        return "i2s";
    case SAMPLE_AUDIO_SOURCE_PDM:
        return "pdm";
    default:
        return "unknown";
    }
}

void sample_audio_show_help(const char *program)
{
    sample_audio_mode_group group = SAMPLE_AUDIO_GROUP_COUNT;

    printf("Usage:\n"
           "  %s <mode> [options]\n\n"
           "Modes (number or name):\n",
           program);
    for (size_t index = 0; index < sizeof(g_modes) / sizeof(g_modes[0]); ++index)
    {
        if (group != g_modes[index].group)
        {
            group = g_modes[index].group;
            printf("\n  %s:\n", sample_audio_mode_group_name(
                                      g_modes[index].mode));
        }
        printf("    %2d  %-18s %s\n", g_modes[index].mode,
               g_modes[index].name, g_modes[index].description);
    }
    printf("\n  Compatibility names:\n"
           "    record-i2s / record-pdm  record-wav with source 0 / 1\n"
           "    loop-i2s / loop-pdm      loop-pcm with source 0 / 1\n"
           "    bind-i2s / bind-pdm      bind-pcm with source 0 / 1\n"
           "\nFile options:\n"
           "  -i, --input PATH       Input for playback modes and duplex-g711\n"
           "  -o, --output PATH      Output for recording modes and duplex-g711\n"
           "      Default WAV path: /sdcard/test.wav\n"
           "      Default G711A path: /sdcard/test.g711a\n"
           "      Default playback selects the newest matching suffixed file\n"
           "\nAudio options:\n"
           "  -r, --rate HZ          8000, 12000, 16000, 24000, 32000, 44100,\n"
           "                         48000, 96000, or 192000 (default 44100)\n"
           "  -b, --bits BITS        16, 24, or 32 (codec modes use 16)\n"
           "  -c, --channels COUNT   1=mono, 2=stereo for WAV recording (default 1)\n"
           "  -s, --source 0|1       0=I2S input, 1=PDM input (default 0)\n"
           "  -C, --codec 0|1        0=external, 1=internal (default 1)\n"
           "  -m, --i2s-format N     1=standard, 2=right, 4=left\n"
           "  -M, --mono-channel N   0=right, 1=left (I2S record-wav only)\n"
           "  -a, --3a MASK          ANS=1, AGC=2, AEC=4; combine as 0-7\n"
           "                         PDM supports ANS only\n"
           "                         AGC: 8/16/32/48 kHz; AEC: modes 8-10\n"
           "                         at 8/12 kHz\n"
           "  -l, --log-level N      MPP log level from 0 to 7\n"
           "  -h, --help             Show this help\n"
           "\nWAV playback runs once and reads rate, bit width, and channels\n"
           "from the file header. Press q or Ctrl-C to stop it early.\n"
           "\nExamples:\n"
           "  %s 0 -o /sdcard/mic.wav\n"
           "  %s 0 -s 1 -o /sdcard/pdm.wav\n"
           "  %s 1 -i /sdcard/mic.wav\n"
           "  %s 2\n"
           "  %s 2 -C 0\n",
           program, program, program, program, program);
}

static k_s32 parse_int(const char *text, int *value)
{
    char *end;
    long parsed;

    errno = 0;
    parsed = strtol(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < INT_MIN || parsed > INT_MAX)
    {
        return K_FAILED;
    }
    *value = (int)parsed;
    return K_SUCCESS;
}

static k_s32 parse_mode(const char *text, sample_audio_mode *mode,
                        int *implied_source)
{
    int value;

    if (parse_int(text, &value) == K_SUCCESS)
    {
        *mode = (sample_audio_mode)value;
        return K_SUCCESS;
    }
    for (size_t index = 0; index < sizeof(g_modes) / sizeof(g_modes[0]); ++index)
    {
        if (strcmp(text, g_modes[index].name) == 0)
        {
            *mode = g_modes[index].mode;
            return K_SUCCESS;
        }
    }
    for (size_t index = 0;
         index < sizeof(g_mode_aliases) / sizeof(g_mode_aliases[0]); ++index)
    {
        if (strcmp(text, g_mode_aliases[index].name) == 0)
        {
            *mode = g_mode_aliases[index].mode;
            *implied_source = g_mode_aliases[index].source;
            return K_SUCCESS;
        }
    }
    printf("unknown mode: %s\n", text);
    return K_FAILED;
}

static k_s32 set_mode(sample_audio_config *config, const char *text,
                      int *implied_source)
{
    if (config->mode != SAMPLE_AUDIO_MODE_NONE)
    {
        printf("mode was specified more than once\n");
        return K_FAILED;
    }
    return parse_mode(text, &config->mode, implied_source);
}

static k_s32 set_bit_width(sample_audio_config *config, int bits)
{
    config->bit_width_bits = bits;
    switch (bits)
    {
    case 16:
        config->bit_width = KD_AUDIO_BIT_WIDTH_16;
        return K_SUCCESS;
    case 24:
        config->bit_width = KD_AUDIO_BIT_WIDTH_24;
        return K_SUCCESS;
    case 32:
        config->bit_width = KD_AUDIO_BIT_WIDTH_32;
        return K_SUCCESS;
    default:
        printf("bits must be 16, 24, or 32\n");
        return K_FAILED;
    }
}

static k_s32 copy_path(char *destination, const char *source)
{
    if (snprintf(destination, SAMPLE_AUDIO_FILENAME_SIZE, "%s", source) >=
        SAMPLE_AUDIO_FILENAME_SIZE)
    {
        printf("file path is too long\n");
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 require_value(int argc, char **argv, int *index,
                           const char **value)
{
    if (*index + 1 >= argc)
    {
        printf("missing value for %s\n", argv[*index]);
        return K_FAILED;
    }
    *value = argv[++(*index)];
    return K_SUCCESS;
}

static k_bool option_is(const char *option, const char *short_name,
                        const char *long_name, const char *legacy_name)
{
    return (short_name != NULL && strcmp(option, short_name) == 0) ||
           (long_name != NULL && strcmp(option, long_name) == 0) ||
           (legacy_name != NULL && strcmp(option, legacy_name) == 0);
}

static k_s32 parse_codec(const char *text, k_bool *enable_codec)
{
    if (strcmp(text, "internal") == 0 || strcmp(text, "1") == 0)
    {
        *enable_codec = K_TRUE;
        return K_SUCCESS;
    }
    if (strcmp(text, "external") == 0 || strcmp(text, "0") == 0)
    {
        *enable_codec = K_FALSE;
        return K_SUCCESS;
    }
    printf("codec must be 0 (external) or 1 (internal)\n");
    return K_FAILED;
}

static k_s32 parse_source(const char *text, sample_audio_source *source)
{
    if (strcmp(text, "i2s") == 0 || strcmp(text, "0") == 0)
    {
        *source = SAMPLE_AUDIO_SOURCE_I2S;
        return K_SUCCESS;
    }
    if (strcmp(text, "pdm") == 0 || strcmp(text, "1") == 0)
    {
        *source = SAMPLE_AUDIO_SOURCE_PDM;
        return K_SUCCESS;
    }
    printf("source must be 0 (I2S) or 1 (PDM)\n");
    return K_FAILED;
}

static k_s32 parse_i2s_format(const char *text, k_i2s_work_mode *mode)
{
    if (strcmp(text, "standard") == 0 || strcmp(text, "1") == 0)
    {
        *mode = K_STANDARD_MODE;
        return K_SUCCESS;
    }
    if (strcmp(text, "right") == 0 || strcmp(text, "2") == 0)
    {
        *mode = K_RIGHT_JUSTIFYING_MODE;
        return K_SUCCESS;
    }
    if (strcmp(text, "left") == 0 || strcmp(text, "4") == 0)
    {
        *mode = K_LEFT_JUSTIFYING_MODE;
        return K_SUCCESS;
    }
    printf("i2s format must be 1 (standard), 2 (right), or 4 (left)\n");
    return K_FAILED;
}

static k_s32 parse_mono_channel(const char *text,
                                k_i2s_in_mono_channel *channel)
{
    if (strcmp(text, "right") == 0 || strcmp(text, "0") == 0)
    {
        *channel = KD_I2S_IN_MONO_RIGHT_CHANNEL;
        return K_SUCCESS;
    }
    if (strcmp(text, "left") == 0 || strcmp(text, "1") == 0)
    {
        *channel = KD_I2S_IN_MONO_LEFT_CHANNEL;
        return K_SUCCESS;
    }
    printf("mono channel must be 0 (right) or 1 (left)\n");
    return K_FAILED;
}

static k_s32 parse_audio3a(const char *text, k_u32 *audio3a)
{
    int value;

    if (strcmp(text, "off") == 0)
    {
        *audio3a = 0;
        return K_SUCCESS;
    }
    if (strcmp(text, "ans") == 0)
    {
        *audio3a = 1;
        return K_SUCCESS;
    }
    if (strcmp(text, "agc") == 0)
    {
        *audio3a = 2;
        return K_SUCCESS;
    }
    if (strcmp(text, "aec") == 0)
    {
        *audio3a = 4;
        return K_SUCCESS;
    }
    if (strcmp(text, "all") == 0)
    {
        *audio3a = 7;
        return K_SUCCESS;
    }
    if (parse_int(text, &value) == K_SUCCESS)
    {
        *audio3a = value;
        return K_SUCCESS;
    }
    printf("3a mode must be off, ans, agc, aec, all, or a 0-7 mask\n");
    return K_FAILED;
}

static k_s32 parse_log_level(const char *text, int *level)
{
    static const char *const names[] = {
        "emerg", "alert", "crit", "error", "warn", "notice", "info", "debug"};

    for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); ++index)
    {
        if (strcmp(text, names[index]) == 0)
        {
            *level = (int)index;
            return K_SUCCESS;
        }
    }
    if (parse_int(text, level) == K_SUCCESS)
    {
        return K_SUCCESS;
    }
    printf("invalid log level\n");
    return K_FAILED;
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

k_bool sample_audio_mode_uses_input(sample_audio_mode mode)
{
    return mode == SAMPLE_AUDIO_MODE_PLAY_WAV ||
           mode == SAMPLE_AUDIO_MODE_PLAY_G711_BIND ||
           mode == SAMPLE_AUDIO_MODE_PLAY_G711 ||
           mode == SAMPLE_AUDIO_MODE_DUPLEX_G711;
}

k_bool sample_audio_mode_uses_output(sample_audio_mode mode)
{
    return mode == SAMPLE_AUDIO_MODE_RECORD_WAV ||
           mode == SAMPLE_AUDIO_MODE_RECORD_G711_BIND ||
           mode == SAMPLE_AUDIO_MODE_RECORD_G711 ||
           mode == SAMPLE_AUDIO_MODE_DUPLEX_G711;
}

k_bool sample_audio_mode_uses_capture(sample_audio_mode mode)
{
    return mode == SAMPLE_AUDIO_MODE_RECORD_WAV ||
           mode == SAMPLE_AUDIO_MODE_LOOP_PCM ||
           mode == SAMPLE_AUDIO_MODE_BIND_PCM ||
           mode == SAMPLE_AUDIO_MODE_RECORD_G711_BIND ||
           mode == SAMPLE_AUDIO_MODE_RECORD_G711 ||
           mode == SAMPLE_AUDIO_MODE_DUPLEX_G711 ||
           mode == SAMPLE_AUDIO_MODE_LOOP_G711 ||
           mode == SAMPLE_AUDIO_MODE_LOOP_OPUS;
}

static k_bool mode_uses_i2s_format(sample_audio_mode mode)
{
    return mode == SAMPLE_AUDIO_MODE_RECORD_WAV ||
           (mode >= SAMPLE_AUDIO_MODE_PLAY_WAV &&
            mode <= SAMPLE_AUDIO_MODE_BIND_PCM);
}

static k_bool mode_uses_audio3a(sample_audio_mode mode)
{
    return mode == SAMPLE_AUDIO_MODE_RECORD_WAV ||
           mode == SAMPLE_AUDIO_MODE_LOOP_PCM ||
           mode == SAMPLE_AUDIO_MODE_BIND_PCM ||
           mode == SAMPLE_AUDIO_MODE_RECORD_G711_BIND ||
           mode == SAMPLE_AUDIO_MODE_RECORD_G711 ||
           (mode >= SAMPLE_AUDIO_MODE_DUPLEX_G711 &&
            mode <= SAMPLE_AUDIO_MODE_LOOP_OPUS);
}

static k_bool agc_rate_is_supported(k_u32 sample_rate)
{
    return sample_rate == 8000 || sample_rate == 16000 ||
           sample_rate == 32000 || sample_rate == 48000;
}

static k_bool mode_has_aec_reference(sample_audio_mode mode)
{
    return mode >= SAMPLE_AUDIO_MODE_DUPLEX_G711 &&
           mode <= SAMPLE_AUDIO_MODE_LOOP_OPUS;
}

void sample_audio_config_init(sample_audio_config *config)
{
    memset(config, 0, sizeof(*config));
    config->mode = SAMPLE_AUDIO_MODE_NONE;
    config->source = SAMPLE_AUDIO_SOURCE_I2S;
    config->sample_rate = 44100;
    config->enable_codec = K_TRUE;
    config->i2s_mode = K_STANDARD_MODE;
    config->channels = 1;
    config->mono_channel = KD_I2S_IN_MONO_RIGHT_CHANNEL;
    copy_path(config->input_file, DEFAULT_WAV_FILE);
    copy_path(config->output_file, DEFAULT_WAV_FILE);
    config->input_is_default = K_TRUE;
    set_bit_width(config, 16);
}

static k_s32 apply_mode_defaults(sample_audio_config *config,
                                 k_u32 specified)
{
    if (!(specified & OPTION_INPUT) &&
        (config->mode == SAMPLE_AUDIO_MODE_PLAY_G711_BIND ||
         config->mode == SAMPLE_AUDIO_MODE_PLAY_G711 ||
         config->mode == SAMPLE_AUDIO_MODE_DUPLEX_G711) &&
        copy_path(config->input_file, DEFAULT_G711_FILE) != K_SUCCESS)
    {
        return K_FAILED;
    }
    if (!(specified & OPTION_OUTPUT) &&
        (config->mode == SAMPLE_AUDIO_MODE_RECORD_G711_BIND ||
         config->mode == SAMPLE_AUDIO_MODE_RECORD_G711 ||
         config->mode == SAMPLE_AUDIO_MODE_DUPLEX_G711) &&
        copy_path(config->output_file, DEFAULT_G711_FILE) != K_SUCCESS)
    {
        return K_FAILED;
    }
    if (!(specified & OPTION_RATE) && config->mode == SAMPLE_AUDIO_MODE_LOOP_OPUS)
    {
        config->sample_rate = 8000;
    }
    if (!(specified & OPTION_BITS) && !config->enable_codec &&
        (config->mode == SAMPLE_AUDIO_MODE_LOOP_PCM ||
         config->mode == SAMPLE_AUDIO_MODE_BIND_PCM))
    {
        return set_bit_width(config, 32);
    }
    return K_SUCCESS;
}

static k_s32 validate_option_scope(const sample_audio_config *config,
                                   k_u32 specified)
{
    if ((specified & OPTION_INPUT) &&
        !sample_audio_mode_uses_input(config->mode))
    {
        printf("mode %s does not use an input file\n",
               sample_audio_mode_name(config->mode));
        return K_FAILED;
    }
    if ((specified & OPTION_OUTPUT) &&
        !sample_audio_mode_uses_output(config->mode))
    {
        printf("mode %s does not use an output file\n",
               sample_audio_mode_name(config->mode));
        return K_FAILED;
    }
    if ((specified & OPTION_SOURCE) &&
        !sample_audio_mode_uses_capture(config->mode))
    {
        printf("mode %s does not capture audio\n",
               sample_audio_mode_name(config->mode));
        return K_FAILED;
    }
    if ((specified & OPTION_CHANNELS) &&
        config->mode != SAMPLE_AUDIO_MODE_RECORD_WAV)
    {
        printf("--channels is only valid for record-wav\n");
        return K_FAILED;
    }
    if ((specified & OPTION_MONO_CHANNEL) &&
        (config->mode != SAMPLE_AUDIO_MODE_RECORD_WAV ||
         config->source != SAMPLE_AUDIO_SOURCE_I2S))
    {
        printf("--mono-channel is only valid for I2S WAV recording\n");
        return K_FAILED;
    }
    if ((specified & OPTION_I2S_FORMAT) &&
        !mode_uses_i2s_format(config->mode))
    {
        printf("mode %s does not use --i2s-format\n",
               sample_audio_mode_name(config->mode));
        return K_FAILED;
    }
    if ((specified & OPTION_I2S_FORMAT) &&
        config->mode == SAMPLE_AUDIO_MODE_RECORD_WAV &&
        config->source == SAMPLE_AUDIO_SOURCE_PDM)
    {
        printf("PDM WAV recording does not use --i2s-format\n");
        return K_FAILED;
    }
    if ((specified & OPTION_AUDIO3A) && !mode_uses_audio3a(config->mode))
    {
        printf("mode %s does not use --3a\n",
               sample_audio_mode_name(config->mode));
        return K_FAILED;
    }
    if ((specified & OPTION_CODEC) &&
        config->source == SAMPLE_AUDIO_SOURCE_PDM &&
        (config->mode == SAMPLE_AUDIO_MODE_RECORD_WAV ||
         config->mode == SAMPLE_AUDIO_MODE_RECORD_G711_BIND ||
         config->mode == SAMPLE_AUDIO_MODE_RECORD_G711))
    {
        printf("PDM recording does not use --codec\n");
        return K_FAILED;
    }
    if ((specified & OPTION_BITS) && config->mode == SAMPLE_AUDIO_MODE_PLAY_WAV)
    {
        printf("play-wav reads the bit width from the WAV header\n");
        return K_FAILED;
    }
    if ((specified & OPTION_RATE) && config->mode == SAMPLE_AUDIO_MODE_PLAY_WAV)
    {
        printf("play-wav reads the sample rate from the WAV header\n");
        return K_FAILED;
    }
    if (config->mode == SAMPLE_AUDIO_MODE_CODEC_MENU && specified != 0)
    {
        printf("codec-menu does not accept audio stream options\n");
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 validate_config(const sample_audio_config *config,
                             k_u32 specified)
{
    if (config->mode < SAMPLE_AUDIO_MODE_MIN ||
        config->mode > SAMPLE_AUDIO_MODE_MAX)
    {
        printf("mode must be between %d and %d\n",
               SAMPLE_AUDIO_MODE_MIN, SAMPLE_AUDIO_MODE_MAX);
        return K_FAILED;
    }
    if (validate_option_scope(config, specified) != K_SUCCESS)
    {
        return K_FAILED;
    }
    if ((sample_audio_mode_uses_input(config->mode) &&
         config->input_file[0] != '/') ||
        (sample_audio_mode_uses_output(config->mode) &&
         config->output_file[0] != '/'))
    {
        printf("file paths must be absolute\n");
        return K_FAILED;
    }
    if (!sample_rate_is_supported(config->sample_rate))
    {
        printf("unsupported sample rate: %u Hz\n", config->sample_rate);
        return K_FAILED;
    }
    if (config->channels != 1 && config->channels != 2)
    {
        printf("channels must be 1 or 2\n");
        return K_FAILED;
    }
    if (config->source != SAMPLE_AUDIO_SOURCE_I2S &&
        config->source != SAMPLE_AUDIO_SOURCE_PDM)
    {
        printf("invalid audio input source\n");
        return K_FAILED;
    }
    if (config->audio3a > 7 || config->log_level < 0 || config->log_level > 7)
    {
        printf("3a mask and log level must be between 0 and 7\n");
        return K_FAILED;
    }
    if (config->audio3a != 0 && config->bit_width != KD_AUDIO_BIT_WIDTH_16)
    {
        printf("3a processing requires 16-bit audio\n");
        return K_FAILED;
    }
    if (config->source == SAMPLE_AUDIO_SOURCE_PDM &&
        (config->audio3a & ~1U) != 0)
    {
        printf("PDM input only supports ANS in --3a\n");
        return K_FAILED;
    }
    if ((config->audio3a & 2U) != 0 &&
        !agc_rate_is_supported(config->sample_rate))
    {
        printf("AGC supports 8000, 16000, 32000, or 48000 Hz\n");
        return K_FAILED;
    }
    if ((config->audio3a & 4U) != 0 &&
        (!mode_has_aec_reference(config->mode) ||
         (config->sample_rate != 8000 && config->sample_rate != 12000)))
    {
        printf("AEC requires an I2S codec loop/duplex mode at 8000 or 12000 Hz\n");
        return K_FAILED;
    }
    if (config->enable_codec && config->bit_width == KD_AUDIO_BIT_WIDTH_32 &&
        ((config->mode == SAMPLE_AUDIO_MODE_RECORD_WAV &&
          config->source == SAMPLE_AUDIO_SOURCE_I2S) ||
         config->mode == SAMPLE_AUDIO_MODE_LOOP_PCM ||
         config->mode == SAMPLE_AUDIO_MODE_BIND_PCM))
    {
        printf("the internal codec supports 16-bit or 24-bit audio\n");
        return K_FAILED;
    }
    if (!config->enable_codec &&
        (config->mode == SAMPLE_AUDIO_MODE_LOOP_PCM ||
         config->mode == SAMPLE_AUDIO_MODE_BIND_PCM) &&
        config->bit_width != KD_AUDIO_BIT_WIDTH_32)
    {
        printf("external loop-pcm and bind-pcm require 32-bit audio\n");
        return K_FAILED;
    }
    if (config->mode >= SAMPLE_AUDIO_MODE_RECORD_G711_BIND &&
        config->mode <= SAMPLE_AUDIO_MODE_LOOP_OPUS &&
        config->bit_width != KD_AUDIO_BIT_WIDTH_16)
    {
        printf("G711 and Opus modes require 16-bit audio\n");
        return K_FAILED;
    }
    if (config->mode >= SAMPLE_AUDIO_MODE_DUPLEX_G711 &&
        config->mode <= SAMPLE_AUDIO_MODE_LOOP_OPUS && !config->enable_codec)
    {
        printf("duplex and codec loopback modes require the internal codec\n");
        return K_FAILED;
    }
    if (config->mode == SAMPLE_AUDIO_MODE_LOOP_OPUS &&
        config->sample_rate != 8000)
    {
        printf("loop-opus requires an 8000 Hz sample rate\n");
        return K_FAILED;
    }
    return K_SUCCESS;
}

static k_s32 route_compat_file(sample_audio_config *config, const char *path,
                               k_u32 *specified)
{
    char *destination;
    k_u32 option;

    if (sample_audio_mode_uses_input(config->mode))
    {
        destination = config->input_file;
        option = OPTION_INPUT;
    }
    else if (sample_audio_mode_uses_output(config->mode))
    {
        destination = config->output_file;
        option = OPTION_OUTPUT;
    }
    else
    {
        printf("mode %s does not use a file\n",
               sample_audio_mode_name(config->mode));
        return K_FAILED;
    }
    if ((*specified & option) != 0)
    {
        printf("file option was specified more than once\n");
        return K_FAILED;
    }
    if (copy_path(destination, path) != K_SUCCESS)
    {
        return K_FAILED;
    }
    *specified |= option;
    return K_SUCCESS;
}

sample_audio_parse_result sample_audio_parse_arguments(
    int argc, char **argv, sample_audio_config *config)
{
    char compat_file[SAMPLE_AUDIO_FILENAME_SIZE] = "";
    k_bool compat_file_set = K_FALSE;
    k_u32 specified = 0;
    int implied_source = -1;

    if (argc < 0 || argv == NULL || config == NULL)
    {
        return SAMPLE_AUDIO_PARSE_ERROR;
    }
    sample_audio_config_init(config);
    if (argc <= 1)
    {
        return SAMPLE_AUDIO_PARSE_HELP;
    }

    for (int index = 1; index < argc; ++index)
    {
        const char *option = argv[index];
        const char *text;
        int value;

        if (option_is(option, "-h", "--help", "-help"))
        {
            return SAMPLE_AUDIO_PARSE_HELP;
        }
        if (option[0] != '-')
        {
            if (set_mode(config, option, &implied_source) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            continue;
        }
        if (option_is(option, NULL, "--type", "-type"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                set_mode(config, text, &implied_source) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
        }
        else if (option_is(option, "-i", "--input", NULL))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                copy_path(config->input_file, text) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_INPUT;
        }
        else if (option_is(option, "-o", "--output", NULL))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                copy_path(config->output_file, text) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_OUTPUT;
        }
        else if (option_is(option, "-f", "--file", "-filename"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                copy_path(compat_file, text) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            compat_file_set = K_TRUE;
        }
        else if (option_is(option, "-r", "--rate", "-samplerate"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_int(text, &value) != K_SUCCESS)
            {
                printf("invalid sample rate\n");
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            config->sample_rate = value;
            specified |= OPTION_RATE;
        }
        else if (option_is(option, "-b", "--bits", "-bitwidth"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_int(text, &value) != K_SUCCESS ||
                set_bit_width(config, value) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_BITS;
        }
        else if (option_is(option, "-c", "--channels", "-channels"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_int(text, &value) != K_SUCCESS)
            {
                printf("invalid channel count\n");
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            config->channels = value;
            specified |= OPTION_CHANNELS;
        }
        else if (option_is(option, "-s", "--source", NULL))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_source(text, &config->source) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_SOURCE;
        }
        else if (option_is(option, "-M", "--mono-channel", "-monochannel"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_mono_channel(text, &config->mono_channel) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_MONO_CHANNEL;
        }
        else if (option_is(option, "-C", "--codec", "-enablecodec"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_codec(text, &config->enable_codec) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_CODEC;
        }
        else if (option_is(option, "-m", "--i2s-format", "-i2smode"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_i2s_format(text, &config->i2s_mode) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_I2S_FORMAT;
        }
        else if (option_is(option, "-a", "--3a", "-audio3a"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_audio3a(text, &config->audio3a) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_AUDIO3A;
        }
        else if (option_is(option, "-l", "--log-level", "-loglevel"))
        {
            if (require_value(argc, argv, &index, &text) != K_SUCCESS ||
                parse_log_level(text, &config->log_level) != K_SUCCESS)
            {
                return SAMPLE_AUDIO_PARSE_ERROR;
            }
            specified |= OPTION_LOG_LEVEL;
        }
        else
        {
            printf("unknown option: %s\n", option);
            return SAMPLE_AUDIO_PARSE_ERROR;
        }
    }

    if (config->mode == SAMPLE_AUDIO_MODE_NONE)
    {
        printf("missing mode\n");
        return SAMPLE_AUDIO_PARSE_ERROR;
    }
    if (implied_source >= 0 && (specified & OPTION_SOURCE) != 0 &&
        config->source != (sample_audio_source)implied_source)
    {
        printf("mode name conflicts with --source\n");
        return SAMPLE_AUDIO_PARSE_ERROR;
    }
    if (implied_source >= 0)
    {
        config->source = (sample_audio_source)implied_source;
    }
    if (compat_file_set &&
        route_compat_file(config, compat_file, &specified) != K_SUCCESS)
    {
        return SAMPLE_AUDIO_PARSE_ERROR;
    }
    config->input_is_default = (specified & OPTION_INPUT) == 0;
    if (apply_mode_defaults(config, specified) != K_SUCCESS)
    {
        return SAMPLE_AUDIO_PARSE_ERROR;
    }
    return validate_config(config, specified) == K_SUCCESS
               ? SAMPLE_AUDIO_PARSE_OK
               : SAMPLE_AUDIO_PARSE_ERROR;
}
