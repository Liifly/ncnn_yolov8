cmake_minimum_required(VERSION 3.12)
project(yolov8-ncnn)


# 导入OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 导入ncnn库
set(ncnn_DIR "/home/liifly/ncnn/build/install/lib/cmake/ncnn")
message(${ncnn_DIR})
find_package(ncnn REQUIRED)

# 添加可执行文件
add_executable(yolov8ncnn yoloV8.cpp Hungarian.cpp KalmanTracker.cpp yolov8main.cpp)

# 链接OpenCV库
target_link_libraries(yolov8ncnn ncnn ${OpenCV_LIBS})
