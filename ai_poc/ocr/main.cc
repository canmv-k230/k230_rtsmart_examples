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
#include "ocr_box.h"
#include "ocr_reco.h"

using std::cerr;
using std::cout;
using std::endl;

std::atomic<bool> isp_stop(false);

#define dict_len 6549

void print_usage(const char *name)
{
    cout << "Usage: " << name << "<kmodel_det> <threshold> <box_thresh> <input_mode> <kmodel_reco> <debug_mode>" << endl
         << "Options:" << endl
         << "  kmodel_det      ocr检测kmodel路径\n"
         << "  threshold       置信度阈值:影响检测框的大小，置信度阈值越小，检测框越大，也更容易检测到文字。\n"
         << "  box_thresh      Box阈值：影响检测框的多少，最后输出的检测框分数小于Box阈值的会被剔除，大于Box阈值的会保留，过大的Box阈值导致漏检，过小的Box阈值导致误检。\n" 
         << "  input_mode      本地图片(图片路径)/ 摄像头(None) \n"
         << "  kmodel_reco     ocr识别kmodel路径 \n"
         << "  debug_mode      是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "\n"
         << endl;
}

void video_proc(char *argv[])
{
    vivcap_start();
    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd
    memset(&vf_info, 0, sizeof(vf_info));
    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    OCRBox ocrbox(argv[1], atof(argv[2]), atof(argv[3]), {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[6]));
    OCRReco ocrreco(argv[5],dict_len,atoi(argv[6]));
    vector<ocr_det_res> results;

    while (!isp_stop)
    {
        ScopedTiming st("total time", 1);
        {
            ScopedTiming st("read capture", atoi(argv[6]));
            // VICAP_CHN_ID_1 out rgb888p
            memset(&dump_info, 0 , sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }
        {
            ScopedTiming st("isp copy", atoi(argv[6]));
            // 从vivcap中读取一帧图像到dump_info
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        results.clear();
        ocrbox.pre_process();
        ocrbox.inference();
        ocrbox.post_process({SENSOR_WIDTH, SENSOR_HEIGHT}, results);

        int matsize = SENSOR_WIDTH * SENSOR_HEIGHT;
        cv::Mat ori_img;
        std::vector<cv::Mat> sensor_bgr;
        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        #if defined(CONFIG_BOARD_K230D_CANMV)
        {
        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
        cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
        cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
        sensor_bgr.push_back(ori_img_B);
        sensor_bgr.push_back(ori_img_G);
        sensor_bgr.push_back(ori_img_R);
        cv::merge(sensor_bgr, ori_img);

        for(int i = 0; i < results.size(); i++)
        {
            vector<Point> vec;
            vector<Point2f> sort_vtd(4);
            vec.clear();
            for(int j = 0; j < 4; j++)
                vec.push_back(results[i].vertices[j]);
            cv::RotatedRect rect = cv::minAreaRect(vec);
            cv::Point2f ver[4];
            rect.points(ver);
            cv::Mat crop;
            Utils::warppersp(ori_img, crop, results[i], sort_vtd);
            ocrreco.pre_process(crop);
            ocrreco.inference();
            vector<unsigned char> results2;
            ocrreco.post_process(results2);
            Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results2,{osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
        }

        {
            ScopedTiming st("osd draw", atoi(argv[5]));
            Utils::draw_ocr_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
        }
        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
        }
        #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
        {   
            #if defined(STUDIO_HDMI)
                cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
                cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
                cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
                sensor_bgr.push_back(ori_img_B);
                sensor_bgr.push_back(ori_img_G);
                sensor_bgr.push_back(ori_img_R);
                cv::merge(sensor_bgr, ori_img);

                for(int i = 0; i < results.size(); i++)
                {
                    vector<Point> vec;
                    vector<Point2f> sort_vtd(4);
                    vec.clear();
                    for(int j = 0; j < 4; j++)
                        vec.push_back(results[i].vertices[j]);
                    cv::RotatedRect rect = cv::minAreaRect(vec);
                    cv::Point2f ver[4];
                    rect.points(ver);
                    cv::Mat crop;
                    Utils::warppersp(ori_img, crop, results[i], sort_vtd);
                    ocrreco.pre_process(crop);
                    ocrreco.inference();
                    vector<unsigned char> results2;
                    ocrreco.post_process(results2);
                    Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results2,{osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }

                {
                    ScopedTiming st("osd draw", atoi(argv[5]));
                    Utils::draw_ocr_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
            #else
            
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                
                cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
                cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
                cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
                sensor_bgr.push_back(ori_img_B);
                sensor_bgr.push_back(ori_img_G);
                sensor_bgr.push_back(ori_img_R);
                cv::merge(sensor_bgr, ori_img);

                for(int i = 0; i < results.size(); i++)
                {
                    vector<Point> vec;
                    vector<Point2f> sort_vtd(4);
                    vec.clear();
                    for(int j = 0; j < 4; j++)
                        vec.push_back(results[i].vertices[j]);
                    cv::RotatedRect rect = cv::minAreaRect(vec);
                    cv::Point2f ver[4];
                    rect.points(ver);
                    cv::Mat crop;
                    Utils::warppersp(ori_img, crop, results[i], sort_vtd);
                    ocrreco.pre_process(crop);
                    ocrreco.inference();
                    vector<unsigned char> results2;
                    ocrreco.post_process(results2);
                    Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results2,{osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }

                {
                    ScopedTiming st("osd draw", atoi(argv[5]));
                    Utils::draw_ocr_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
            
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            #endif
        }
        #else
        {
        cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
        cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
        cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
        sensor_bgr.push_back(ori_img_B);
        sensor_bgr.push_back(ori_img_G);
        sensor_bgr.push_back(ori_img_R);
        cv::merge(sensor_bgr, ori_img);

        for(int i = 0; i < results.size(); i++)
        {
            vector<Point> vec;
            vector<Point2f> sort_vtd(4);
            vec.clear();
            for(int j = 0; j < 4; j++)
                vec.push_back(results[i].vertices[j]);
            cv::RotatedRect rect = cv::minAreaRect(vec);
            cv::Point2f ver[4];
            rect.points(ver);
            cv::Mat crop;
            Utils::warppersp(ori_img, crop, results[i], sort_vtd);
            ocrreco.pre_process(crop);
            ocrreco.inference();
            vector<unsigned char> results2;
            ocrreco.post_process(results2);
            Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results2,{osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
        }

        {
            ScopedTiming st("osd draw", atoi(argv[5]));
            Utils::draw_ocr_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
        }
        }
        #endif

        {
            ScopedTiming st("osd copy", atoi(argv[5]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            //显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0

            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret) 
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
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
    if (argc != 7)
    {
        print_usage(argv[0]);
        return -1;
    }
    if (strcmp(argv[4], "None") == 0)
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
        vector<ocr_det_res> results;
        cv::Mat ori_img = cv::imread(argv[4]);
        cv::Mat draw_img = ori_img.clone();
        int ori_w = ori_img.cols;
        int ori_h = ori_img.rows;
        OCRBox ocrbox(argv[1], atof(argv[2]), atof(argv[3]), atoi(argv[6]));
        OCRReco ocrreco(argv[5],dict_len,atoi(argv[6]));
        ocrbox.pre_process(ori_img);
        ocrbox.inference();
        ocrbox.post_process({ori_w, ori_h}, results);

        for(int i = 0; i < results.size(); i++)
        {
            vector<Point> vec;
            vector<Point2f> sort_vtd(4);
            vec.clear();
            for(int j = 0; j < 4; j++)
                vec.push_back(results[i].vertices[j]);
            cv::RotatedRect rect = cv::minAreaRect(vec);
            cv::Point2f ver[4];
            rect.points(ver);
            cv::Mat crop;
            Utils::warppersp(ori_img, crop, results[i], sort_vtd);
            ocrreco.pre_process(crop);
            ocrreco.inference();
            vector<unsigned char> results2;
            ocrreco.post_process(results2);
            Utils::draw_ocr_text(int(results[i].meanx), int(results[i].meany),draw_img,results2);
        }
        Utils::draw_ocr_det_res(draw_img, results);
        results.clear();
        cv::imwrite("ocr_result.jpg", draw_img);
    }
    return 0;
}