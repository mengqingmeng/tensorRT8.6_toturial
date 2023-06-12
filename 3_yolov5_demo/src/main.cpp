#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <fstream>
#include "yolov5_trt.h"

int main(int argc,char** argv){
    std::string engineFilePath = "/HardDisk/DEV/CODE/tensorrt8.6_toturial/models/side_gou.engine";

    cv::Mat inImage = cv::imread("/HardDisk/DEV/CODE/tensorrt8.6_toturial/images/gou2.jpg");

    if(inImage.empty()) {
        std::cerr << "image empty" << std::endl;
        return -1;
    }

    Yolov5TRT yolov5;
    yolov5.init(engineFilePath,"",0.25,0.45,{"gou"});

    std::vector<DetectResult> results;
    yolov5.detect(inImage,results);

    std::cout << "result size:" << results.size() << std::endl;
    for(auto& result:results){
        std::cout  << "object label:" << result.label << " conf:" << result.conf << std::endl;
        //cv::rectangle(inImage,result.box,cv::Scalar(0,255,0),2,cv::LINE_AA);
    }    
    return 0;
}
