#ifndef YOLOV5_TRT
#define YOLOV5_TRT
#include <iostream>
#include "yolo_base.h"
#include <NvInfer.h>
#include "NvOnnxParser.h"

class Yolov5TRT : public YoloBase{
    public:
        Yolov5TRT();

        ~Yolov5TRT();

    protected:
        virtual void doDetect(const cv::Mat& inImage,std::vector<DetectResult>& results) override;
        virtual void doInit(const std::string& engineFile) override;

    private:
        /// @brief TRT 运行时
        nvinfer1::IRuntime* m_runtime;

        /// @brief TRT 引擎
        nvinfer1::ICudaEngine* m_engine;

        /// @brief TRT 上下文
        nvinfer1::IExecutionContext* m_context;
        
        void* m_cudaBuffers[2] = {NULL,NULL};
        cudaStream_t m_cudaStream;

         /// @brief 模型输入索引
        int m_inputIndex;

        /// @brief 模型输出索引
        int m_outputIndex;

        /// @brief 初始化TensortRT模型
        /// @param engineFilePath 模型路径
        void initTensorRT(const std::string& engineFilePath);
};
#endif
