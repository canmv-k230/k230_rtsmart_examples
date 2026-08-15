#ifndef SAMPLE_AUDIO_AUDIO_FILE_H
#define SAMPLE_AUDIO_AUDIO_FILE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdio.h>

#include "k_type.h"

#define AUDIO_FILE_PATH_SIZE 256

k_s32 audio_file_open_unique(const char *requested_path, const char *mode,
                             FILE **file, char *actual_path,
                             size_t actual_path_size);
k_s32 audio_file_find_latest(const char *requested_path, char *actual_path,
                             size_t actual_path_size);
k_s32 audio_file_discard(FILE **file, const char *path);

#ifdef __cplusplus
}
#endif

#endif
