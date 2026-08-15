#include "sample_audio_modes.h"

#include <stddef.h>

#include "audio_sample.h"

k_bool sample_audio_mode_is_interactive(sample_audio_mode mode)
{
    return mode >= SAMPLE_AUDIO_MODE_PLAY_WAV &&
           mode <= SAMPLE_AUDIO_MODE_LOOP_OPUS;
}

k_bool sample_audio_mode_uses_vb(sample_audio_mode mode)
{
    return mode >= SAMPLE_AUDIO_MODE_RECORD_WAV &&
           mode <= SAMPLE_AUDIO_MODE_LOOP_OPUS;
}

k_s32 sample_audio_run_mode(const sample_audio_config *config)
{
    k_audio_dev ai_dev;

    if (config == NULL || config->mode < SAMPLE_AUDIO_MODE_MIN ||
        config->mode > SAMPLE_AUDIO_MODE_MAX)
    {
        return K_FAILED;
    }

    audio_sample_enable_audio_codec(config->enable_codec);
    ai_dev = config->source == SAMPLE_AUDIO_SOURCE_PDM ? 1 : 0;

    switch (config->mode)
    {
    case SAMPLE_AUDIO_MODE_RECORD_WAV:
        if (config->source == SAMPLE_AUDIO_SOURCE_PDM)
        {
            return audio_sample_get_ai_pdm_data(
                config->output_file, config->bit_width, config->sample_rate,
                config->channels, config->audio3a);
        }
        return audio_sample_get_ai_i2s_data(
            config->output_file, config->bit_width, config->sample_rate,
            config->channels, config->mono_channel, config->i2s_mode,
            config->audio3a);
    case SAMPLE_AUDIO_MODE_PLAY_WAV:
        return audio_sample_send_ao_data(
            config->input_file, 0, 0, config->i2s_mode);

    case SAMPLE_AUDIO_MODE_LOOP_PCM:
        return audio_sample_api_ai_to_ao(
            ai_dev, 0, 0, 0, config->sample_rate, config->bit_width,
            config->i2s_mode, config->audio3a);
    case SAMPLE_AUDIO_MODE_BIND_PCM:
        return audio_sample_bind_ai_to_ao(
            ai_dev, 0, 0, 0, config->sample_rate, config->bit_width,
            config->i2s_mode, config->audio3a);

    case SAMPLE_AUDIO_MODE_RECORD_G711_BIND:
        return audio_sample_ai_encode(
            ai_dev, K_TRUE, config->sample_rate, config->bit_width, 0,
            K_PT_G711A, config->output_file, config->audio3a);
    case SAMPLE_AUDIO_MODE_PLAY_G711_BIND:
        return audio_sample_decode_ao(
            K_TRUE, config->sample_rate, config->bit_width, 0, K_PT_G711A,
            config->input_file);
    case SAMPLE_AUDIO_MODE_RECORD_G711:
        return audio_sample_ai_encode(
            ai_dev, K_FALSE, config->sample_rate, config->bit_width, 0,
            K_PT_G711A, config->output_file, config->audio3a);
    case SAMPLE_AUDIO_MODE_PLAY_G711:
        return audio_sample_decode_ao(
            K_FALSE, config->sample_rate, config->bit_width, 0, K_PT_G711A,
            config->input_file);
    case SAMPLE_AUDIO_MODE_DUPLEX_G711:
        return audio_sample_ai_aenc_adec_ao(
            ai_dev, 0, 0, 0, 0, 0, config->sample_rate, config->bit_width,
            K_PT_G711A, config->input_file, config->output_file,
            config->audio3a);

    case SAMPLE_AUDIO_MODE_LOOP_G711:
        return audio_sample_ai_aenc_adec_ao_2(
            ai_dev, 0, 0, 0, 0, 0, config->sample_rate, config->bit_width,
            K_PT_G711A, config->audio3a);
    case SAMPLE_AUDIO_MODE_LOOP_OPUS:
        return audio_sample_ai_aenc_adec_ao_opus(
            ai_dev, 0, 0, 0, 0, 0, config->sample_rate, config->bit_width,
            K_PT_OPUS, config->audio3a);

    case SAMPLE_AUDIO_MODE_CODEC_MENU:
        return audio_sample_acodec();
    default:
        return K_FAILED;
    }
}
