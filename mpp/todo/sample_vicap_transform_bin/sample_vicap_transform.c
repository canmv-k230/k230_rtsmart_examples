/* vicap */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>

#include "k_module.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_vicap_comm.h"
#include "k_video_comm.h"
#include "k_sys_comm.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
#include "mpi_isp_api.h"
#include "mpi_sys_api.h"
#include "k_vo_comm.h"
#include "mpi_vo_api.h"
#include "vo_test_case.h"
#include "mpi_nonai_2d_api.h"
#include "k_nonai_2d_comm.h"
#include "k_connector_comm.h"
#include "mpi_connector_api.h"
#include "mpi_venc_api.h"
#include "k_venc_comm.h"



int main(int argc, char *argv[])
{
    char *path = "/sharefs/sharefs/"; //./";
    char *config_name = "imx335-1920x1080";

    kd_mpi_export_vicap_config_to_bin(path, config_name);

}
