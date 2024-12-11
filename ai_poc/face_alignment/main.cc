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
#include <iostream>
#include <thread>
#include "utils.h"
#include "vi_vo.h"
#include "face_detection.h"
#include "face_alignment.h"
#include "face_alignment_post.h"

using std::cerr;
using std::cout;
using std::endl;

std::atomic<bool> isp_stop(false);

void print_usage(const char *name)
{
    cout << "Usage: " << name << "<kmodel_det> <obj_thres> <nms_thres> <kmodel_align> <kmodel_align_post> <input_mode> <output_mode> <debug_mode>" << endl
         << "Options:" << endl
         << "  kmodel_det               人脸检测kmodel路径\n"
         << "  obj_thres                人脸检测阈值\n"
         << "  nms_thres                人脸检测nms阈值\n"
         << "  kmodel_align             人脸对齐kmodel路径\n"
         << "  kmodel_align_post        人脸对齐后处理kmodel路径\n"
         << "  input_mode               本地图片(图片路径)/ 摄像头(None) \n"
         << "  output_mode              渲染方式（depth、pncc）\n"            
         << "  debug_mode               是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "\n"
         << endl;
}

void video_proc(char *argv[])
{
    vivcap_start();
    // 设置osd参数
    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd
    memset(&vf_info, 0, sizeof(vf_info));
    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory,get isp memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    FaceDetection face_det(argv[1], atof(argv[2]),atof(argv[3]), {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[8]));
    FaceAlignment face_align(argv[4],argv[5], {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[8]));
    string output_mode(argv[7]);

    vector<FaceDetectionInfo> det_results;
    while (!isp_stop)
    {
        ScopedTiming st("total time", 1);

        {
            ScopedTiming st("read capture", atoi(argv[8]));
            // 从vivcap中读取一帧图像到dump_info
            memset(&dump_info, 0, sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }

        {
            ScopedTiming st("isp copy", atoi(argv[8]));
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        det_results.clear();

        face_det.pre_process();
        face_det.inference();
        // 旋转后图像
        face_det.post_process({SENSOR_WIDTH, SENSOR_HEIGHT}, det_results);
        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        for (int i = 0; i < det_results.size(); ++i)
        {
            face_align.pre_process(det_results[i].bbox);
            face_align.inference();
            vector<float> vertices;
            face_align.post_process({ISP_CHN0_WIDTH, ISP_CHN0_HEIGHT},vertices,false); 
            #if defined(CONFIG_BOARD_K230D_CANMV) || defined(CONFIG_BOARD_K230_CANMV_V3P0) || defined(CONFIG_BOARD_K230_CANMV_LCKFB)
            {
                ScopedTiming st("osd draw", atoi(argv[8]));
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                if(output_mode == "depth")
                    face_align.get_depth(osd_frame, vertices);
                else if(output_mode == "pncc")
                    face_align.get_pncc(osd_frame, vertices);
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            }
            #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
            {
                #if defined(STUDIO_HDMI)
                {
                    ScopedTiming st("osd draw", atoi(argv[8]));
                    if(output_mode == "depth")
                        face_align.get_depth(osd_frame, vertices);
                    else if(output_mode == "pncc")
                        face_align.get_pncc(osd_frame, vertices);
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[8]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    if(output_mode == "depth")
                        face_align.get_depth(osd_frame, vertices);
                    else if(output_mode == "pncc")
                        face_align.get_pncc(osd_frame, vertices);
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #endif
            }
            #else
            {
                ScopedTiming st("osd draw", atoi(argv[8]));
                if(output_mode == "depth")
                    face_align.get_depth(osd_frame, vertices);
                else if(output_mode == "pncc")
                    face_align.get_pncc(osd_frame, vertices);
            }
            #endif
        }

        {
            ScopedTiming st("osd copy", atoi(argv[8]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info); // K_VO_OSD0
            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();

    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 9)
    {
        print_usage(argv[0]);
        return -1;
    }

    if (strcmp(argv[6], "None") == 0)
    {
        std::thread thread_isp(video_proc, argv);
        while (getchar() != 'q')
        {
            usleep(10000);
        }

        isp_stop = true;
        thread_isp.join();
    }
    else
    {   
        cv::Mat ori_img = cv::imread(argv[6]);
        int ori_w = ori_img.cols;
        int ori_h = ori_img.rows;
        FaceDetection face_det(argv[1], atof(argv[2]),atof(argv[3]), atoi(argv[8]));
        face_det.pre_process(ori_img);
        face_det.inference();

        vector<FaceDetectionInfo> det_results;
        face_det.post_process({ori_w, ori_h}, det_results);
    
        FaceAlignment face_align(argv[4],argv[5],atoi(argv[8]));
        string output_mode(argv[7]);

        for (int i = 0; i < det_results.size(); ++i)
        {
            face_align.pre_process(ori_img, det_results[i].bbox);
            face_align.inference();
            vector<float> vertices;
            face_align.post_process({ori_w, ori_h},vertices,true); 
            if(output_mode == "depth")
                face_align.get_depth(ori_img, vertices);
            else if(output_mode == "pncc")
                face_align.get_pncc(ori_img, vertices);
        }
        cv::imwrite("face_align_result.jpg", ori_img);
    }
    return 0;
}