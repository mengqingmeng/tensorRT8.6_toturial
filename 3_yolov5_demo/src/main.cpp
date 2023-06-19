#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <fstream>
#include "yolov5_trt.h"
#include "yolov5_seg_trt.h"

#define yolov5Seg 1

int main(int argc,char** argv){
    std::string engineFilePath = "/HardDisk/DEV/CODE/tensorrt8.6_toturial/models/side_gou.engine";

    std::string segEngineFile = '/home/mqm/Workspace/code/yolov5-7.0/yolov5m-seg.engine';

    cv::Mat inImage = cv::imread("/HardDisk/DEV/CODE/tensorrt8.6_toturial/images/gou2.jpg");
    cv::Mat bus = cv::imread("/HardDisk/DEV/CODE/tensorrt8.6_toturial/images/bus.jpg");

    if(inImage.empty()) {
        std::cerr << "image empty" << std::endl;
        return -1;
    }
    double start = static_cast<double>(cv::getTickCount());

    // yolov5目标检测
    #ifdef yolov5
    Yolov5TRT* yolov5 = new Yolov5TRT;
    yolov5->init(engineFilePath,"",0.25,0.45,{"gou"});

    double initTime = ((double)cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    std::cout << "init cost:" << initTime << std::endl; 
    
    double startDetect = static_cast<double>(cv::getTickCount());
    std::vector<DetectResult> results;
    yolov5->detect(inImage,results);
    double detectTime = ((double)cv::getTickCount() - startDetect) / cv::getTickFrequency() * 1000;
    std::cout << "detect cost:" << detectTime << std::endl;

    std::cout << "result size:" << results.size() << std::endl;
    for(auto& result:results){
        std::cout  << "object label:" << result.label << " conf:" << result.conf << std::endl;
        //cv::rectangle(inImage,result.box,cv::Scalar(0,255,0),2,cv::LINE_AA);
    }    

    delete yolov5;
    #endif

    // yolov5 实例分割
    #ifdef yolov5Seg
    Yolov5SegTrt* yolov5seg = new Yolov5SegTrt;
    yolov5seg->init(segEngineFile,"",0.25,0.45,{'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush'});

    double initTime = ((double)cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    std::cout << "init cost:" << initTime << std::endl; 
    
    double startDetect = static_cast<double>(cv::getTickCount());
    std::vector<DetectResult> results;
    yolov5seg->detect(inImage,results);
    double detectTime = ((double)cv::getTickCount() - startDetect) / cv::getTickFrequency() * 1000;
    std::cout << "detect cost:" << detectTime << std::endl;

    std::cout << "result size:" << results.size() << std::endl;
    for(auto& result:results){
        std::cout  << "object label:" << result.label << " conf:" << result.conf << std::endl;
        //cv::rectangle(inImage,result.box,cv::Scalar(0,255,0),2,cv::LINE_AA);
    }    

    delete yolov5seg;
    #endif

    return 0;
}
