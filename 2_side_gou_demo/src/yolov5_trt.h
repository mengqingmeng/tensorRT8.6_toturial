#ifndef YOLOV5_TRT
#define YOLOV5_TRT
#include <iostream>
#include "yolo_base.h"

class Yolov5TRT : public YoloBase{
    public:
        /// @brief 初始化模型
        /// @param engineFile 模型文件
        /// @param lablesFile   标签文件
        /// @param confThreshold 置信度
        /// @param objScoreThreshold 对象得分阈值
        virtual void initConfig(const std::string& engine_file,const std::string& lables_file,const float confThreshold,float obj_score_threshold) override;
        /// @brief 目标检测
        /// @param image 输入图像
        /// @param results 输出结果
        virtual void detect(const cv::Mat& image,std::vector<DetectResult> results) override;

        /// @brief 改变置信度
        /// @param conf 置信度
        virtual void changeConf(float conf) override;
};
#endif
