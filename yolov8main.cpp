// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//modified 1-14-2023 Q-engineering

#include "yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>


YoloV8 yolov8;
int target_size = 640; //416; //320;  must be divisible by 32.

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return 0;
    }

    yolov8.load(target_size);       //load model (once) see yoloyV8.cpp line 246

    std::vector<Object> objects;
    std::vector<RecognitionResult> results;//保存识别结果
 
    yolov8.detect(m, objects);      //recognize the objects
    yolov8.draw(m, objects);        //show the outcome
    yolov8.form(m,objects,results);        //显示表格
    //cv::imshow("Input Image", m);
    //cv::imshow("RPi4 - 1.95 GHz - 2 GB ram",m);
    //cv::imwrite("out.jpg",m);
    cv::waitKey(0);

    return 0;
}
/*
int main(int argc, char** argv) {
    const char* vediopath = argv[1];

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [vediopath]\n", argv[0]);
        return -1;
    }

    // 打开视频文件
    cv::VideoCapture video("vediopath");

    if (!video.isOpened()) {
        std::cout << "无法打开视频文件" << std::endl;
        return -1;
    }

    // 创建窗口用于显示图像
    cv::namedWindow("Video", cv::WINDOW_NORMAL);

    // 循环读取视频帧
    cv::Mat frame;
    while (video.read(frame)) {
    
        yolov8.load(target_size);       //load model (once) see yoloyV8.cpp line 246

    	std::vector<Object> objects;
    	std::vector<RecognitionResult> results;//保存识别结果
 
    	yolov8.detect(frame, objects);      //recognize the objects
    	yolov8.draw(frame, objects);        //show the outcome
    	yolov8.form(frame,objects,results);        //显示表格
  	//cv::imshow("Input Image", m);
  	//cv::imshow("RPi4 - 1.95 GHz - 2 GB ram",m);
   	//cv::imwrite("out.jpg",m);
    	//cv::waitKey(0);
        
        // 显示当前帧
        cv::imshow("Video", frame);

        // 按下ESC键退出循环
        if (cv::waitKey(30) == ' ') {
            break;
        }
    }

    // 释放视频对象和关闭窗口
    video.release();
    cv::destroyAllWindows();

    return 0;
}
*/
