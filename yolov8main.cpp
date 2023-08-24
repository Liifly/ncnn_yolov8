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
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <unistd.h>  
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;

// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

// global variables for counting
#define CNUM 20
int total_frames = 0;
double total_time = 0.0;

void TestSORT(string seqName, bool display,const std::vector<TrackingBox>& detData,int frameCount);

YoloV8 yolov8;
int target_size = 640; //416; //320;  must be divisible by 32.

/*
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
*/
///*
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [vediopath]" << std::endl;
        return -1;
    }

    const std::string videoPath = argv[1];

    // 打开视频文件
    cv::VideoCapture video(videoPath);
    if (!video.isOpened()) {
        std::cerr << "无法打开视频文件" << std::endl;
        return -1;
    }
    
    /*
    // 打开输出文件
    std::ofstream outputFile("det.txt");
    if (!outputFile.is_open()) {
        std::cerr << "无法打开输出文件" << std::endl;
        return -1;
    }
    */

    // 获取视频的宽度和高度
    int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    //std::cout<<width<<" "<<height<<std::endl;

    // 设置输出视频的编解码器和帧率
    cv::VideoWriter outputVideo("output.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 25, cv::Size(width, height));
    //cv::VideoWriter outputVideo("output.mp4", cv::VideoWriter::fourcc('H','2','6','4'), 25, cv::Size(width, height));

    // 初始化 yolov8 模型
    yolov8.load(target_size);

    // 循环读取视频帧
    cv::Mat frame;
    int frameCount = 0;
    
    cv::namedWindow("Output Video", cv::WINDOW_NORMAL);
    
    std::vector<TrackingBox> detData;
    
    while (video.read(frame)) {
        
    

        // 检测目标
        //std::cout<<frame.size()<<" "<<std::endl;
        std::vector<Object> objects;
        std::vector<RecognitionResult> results;
        yolov8.detect(frame, objects);
        //std::cout<<"检测目标完成"<<std::endl;

        // 绘制目标框
        //yolov8.draw(frame, objects);
        //yolov8.form(frame, objects, results);
        
        /*
        // 写入目标信息到 det.txt 文件
        for (const auto& object : objects) {
            int frame_id = frameCount + 1;
            float bbox_x = object.rect.x;
            float bbox_y = object.rect.y;
            float bbox_width = object.rect.width;
            float bbox_height = object.rect.height;
            float confidence = object.prob;
            
             //std::cout<< frame_id << "," << -1 << "," << bbox_x << "," << bbox_y << ","
                       //<< bbox_width << "," << bbox_height << "," << confidence << ",-1,-1,-1" <<std::endl;

            // 写入到文件中
            outputFile << frame_id << "," << -1 << "," << bbox_x << "," << bbox_y << ","
                       << bbox_width << "," << bbox_height << "," << confidence << ",-1,-1,-1" << std::endl;
        }
        */
        
        string detLine;

	
	//char ch;
	//float tpx, tpy, tpw, tph;
	 
	detData.clear();

	for (const auto& object : objects) {
    		TrackingBox tb;
    		tb.frame = frameCount + 1;
    		tb.id = -1;
    		tb.box = Rect_<float>(Point_<float>(object.rect.x, object.rect.y), Point_<float>(object.rect.x + object.rect.width, object.rect.y + object.rect.height));
    		detData.push_back(tb);
	}
	//std::cout<<"信息转写完成"<<std::endl;
        
        // 缩小窗口尺寸
        //cv::resizeWindow("Output Video", frame.cols / 3*2, frame.rows / 3*2);

        // 保存帧为JPG图片
        std::string imageName = "frame_" + std::to_string(frameCount) + ".jpg";
        cv::imwrite(imageName, frame);
        //std::cout<<"图片保存完成"<<std::endl;

        TestSORT("yolov8", true, detData, frameCount);

        // 将帧添加到输出视频中
        //outputVideo.write(frame);
        
        // 显示输出到窗口（可选）
	//cv::imshow("Output Video", frame);
	//cv::waitKey(1); // 等待 1 毫秒，确保窗口更新

        frameCount++;
    }
    
    
    
    cout << "Total Tracking took: " << total_time << " for " << total_frames << " frames or " << ((double)total_frames / (double)total_time) << " FPS" << endl;
        
    // 声明识别结果向量
    // 保存识别结果为表格图片
    //std::vector<Object> objects;
    //std::vector<RecognitionResult> results;
    //yolov8.form(frame, objects, results);
    //cv::imwrite("result_table.jpg", frame);
    
     // 关闭输出文件
    //outputFile.close();
    
    // 指定目录和图片名称
    string directory = "/home/liifly/yolov8/build";
    string filename_pattern = "output_*.jpg";  // 假设文件名以'image_'开头，后面跟着帧序号和'.jpg'

    // 获取指定目录下特定名称的图片文件列表
    vector<string> image_files;
    string cmd = "ls " + directory + "/" + filename_pattern + " > temp.txt";
    system(cmd.c_str());
    ifstream file("temp.txt");
    string image_file;
    while (getline(file, image_file)) {
        image_files.push_back(directory + "/" + image_file);
    }
    file.close();
    
    // 按文件名排序，确保按照正确的顺序合成视频
    sort(image_files.begin(), image_files.end());

    // 设置视频编码器和输出路径
    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');  // 可根据需要选择其他编码器
    string output_path = "/home/liifly/yolov8/build/output_video.mp4";

    // 获取第一张图片的尺寸，作为视频的宽高
    Mat first_image = imread(image_files[0]);
    int h = first_image.rows;
    int w = first_image.cols;

    // 创建视频写入对象
    VideoWriter video_writer(output_path, fourcc, 25, Size(w, h));  // 后面的参数可根据需要进行调整

    // 逐帧写入视频
    for (const string& image_file : image_files) {
        Mat frame = imread(image_file);
        video_writer.write(frame);
    }

    // 释放资源
    video_writer.release();
    
    // 删除临时文件
    remove("temp.txt");
    
    cout << "视频合成完成：" << output_path << endl;

    // 释放视频对象和关闭输出视频
    video.release();
    outputVideo.release();

    return 0;
}

void TestSORT(string seqName, bool display,const std::vector<TrackingBox>& detData,int frameCount)
{
	//cout << "Processing " << seqName << "..." << endl;

	// 0. randomly generate colors, only for display
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

	string imgPath = "/home/liifly/yolov8/build/";

	if (display)
		if (eaccess(imgPath.c_str(), 0) == -1)
		{
			cerr << "Image path not found!" << endl;
			display = false;
		}
	// 3. update across frames
	static int frame_count = 0;
	static int max_age = 1;
	static int min_hits = 3;
	static double iouThreshold = 0.3;
	static vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	double cycle_time = 0.0;
	int64 start_time = 0;

	// prepare result file.
	//ofstream resultsFile;
	//string resFileName = "output/" + seqName + ".txt";
	//resultsFile.open(resFileName);

	//if (!resultsFile.is_open())
	{
		//cerr << "Error: can not create file " << resFileName << endl;
		//return;
	}
	//cout<<"checkpoint1"<<endl;
	//////////////////////////////////////////////
	// main loop
		//cout<<fi<<endl;
		total_frames++;
		//cout << frame_count << endl;

		// I used to count running time using clock(), but found it seems to conflict with cv::cvWaitkey(),
		// when they both exists, clock() can not get right result. Now I use cv::getTickCount() instead.
		start_time = getTickCount();

		if (trackers.size() == 0) // the first frame met
		{
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detData.size(); i++)
			{
				KalmanTracker trk = KalmanTracker(detData[i].box);
				trackers.push_back(trk);
			}
			// output the first frame detections
			for (unsigned int id = 0; id < detData.size(); id++)
			{
				TrackingBox tb = detData[id];
				//resultsFile << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
			}
			return;
	       }

		///////////////////////////////////////
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();

		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
				//cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}

		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		trkNum = predictedBoxes.size();
		detNum = detData.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
			;

		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

		///////////////////////////////////////
		// 3.3. updating trackers

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detData[detIdx].box);
		}

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detData[umd].box);
			trackers.push_back(tracker);
		}

		// get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}

		cycle_time = (double)(getTickCount() - start_time);
		total_time += cycle_time / getTickFrequency();

		for (auto tb : frameTrackingResult)
			//resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;

		if (display) // read image, draw results and show them
		{
			ostringstream oss;
			oss << imgPath << "frame_" << frameCount;
			Mat img = imread(oss.str() + ".jpg");
			if (img.empty())
				continue;
			
			for (auto tb : frameTrackingResult)
			{
				cv::rectangle(img, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
				
				// 在左上角添加物体ID号文本
       			 cv::Point textPos(tb.box.x, tb.box.y - 10);
       			 cv::putText(img, to_string(tb.id), textPos, cv::FONT_HERSHEY_SIMPLEX, 0.8, randColor[tb.id % CNUM], 2);
       			 
			}
			imshow(seqName, img);
			cv::waitKey(40);
			std::string imagename = "output_" + std::to_string(frameCount) + ".jpg";
       		cv::imwrite(imagename, img);
			
		}

	//resultsFile.close();

	//if (display)
		//destroyAllWindows();
}
//*/
