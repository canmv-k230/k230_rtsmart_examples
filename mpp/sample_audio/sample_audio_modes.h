#ifndef SAMPLE_AUDIO_MODES_H
#define SAMPLE_AUDIO_MODES_H

#include "sample_audio_config.h"

k_s32 sample_audio_run_mode(const sample_audio_config *config);
k_bool sample_audio_mode_is_interactive(sample_audio_mode mode);
k_bool sample_audio_mode_uses_vb(sample_audio_mode mode);

#endif
