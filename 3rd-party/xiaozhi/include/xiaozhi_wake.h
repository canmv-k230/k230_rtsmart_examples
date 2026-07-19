#ifndef XIAOZHI_WAKE_H
#define XIAOZHI_WAKE_H

#include <stddef.h>
#include <stdint.h>

#include "xiaozhi_config.h"

struct xiaozhi_wake;

struct xiaozhi_wake_config {
	const char *model_path;
	const char *task_name;
	int keyword_count;
	float threshold;
};

#ifdef __cplusplus
extern "C" {
#endif

struct xiaozhi_wake *xiaozhi_wake_create(
	const struct xiaozhi_wake_config *config);

void xiaozhi_wake_destroy(struct xiaozhi_wake *wake);

int xiaozhi_wake_reset(struct xiaozhi_wake *wake);

int xiaozhi_wake_process(struct xiaozhi_wake *wake,
				 const int16_t *samples, size_t sample_count);

#ifdef __cplusplus
}
#endif

#endif
