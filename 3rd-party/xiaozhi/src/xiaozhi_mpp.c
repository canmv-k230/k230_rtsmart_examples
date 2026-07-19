#include "xiaozhi_mpp.h"

#include <mpi_vb_api.h>

#include <stdio.h>
#include <string.h>

static int vb_initialized;

int xiaozhi_mpp_initialize(void)
{
	k_vb_config config;
	k_s32 ret;

	if (vb_initialized)
		return 0;

	/* Keep VB lifetime above audio, LVGL, and any future media modules. */
	memset(&config, 0, sizeof(config));
	config.max_pool_cnt = VB_MAX_POOLS;
	ret = kd_mpi_vb_set_config(&config);
	if (ret) {
		printf("xiaozhi: kd_mpi_vb_set_config failed ret=%d\n", ret);
		return -1;
	}

	ret = kd_mpi_vb_init();
	if (ret) {
		printf("xiaozhi: kd_mpi_vb_init failed ret=%d\n", ret);
		return -1;
	}
	vb_initialized = 1;
	return 0;
}

void xiaozhi_mpp_deinitialize(void)
{
	k_s32 ret;

	if (!vb_initialized)
		return;
	ret = kd_mpi_vb_exit();
	if (ret)
		printf("xiaozhi: kd_mpi_vb_exit failed ret=%d\n", ret);
	else
		vb_initialized = 0;
}
