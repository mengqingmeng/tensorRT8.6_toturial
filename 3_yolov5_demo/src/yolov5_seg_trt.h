#ifndef YOLOV5_SEG_TRT
#define YOLOV5_SEG_TRT
#include <iostream>
#include "yolo_base.h"
#include <NvInfer.h>
#include "NvOnnxParser.h"

class Yolov5SegTrt : public YoloBase{
    public:
        Yolov5SegTrt();
        ~Yolov5SegTrt();

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
        
        //void* m_cudaBuffers[3] = {NULL,NULL,NULL};

        std::vector<void*> m_cudaBuffers;

        cudaStream_t m_cudaStream;

         /// @brief 模型输入索引
        int m_inputIndex;

        /// @brief 模型输出索引
        int m_outputIndex;

        /// @brief 掩膜输出索引
        int m_maskIndex;

        int m_maskChannels;

        int m_maskWidth;

        int m_maskHeight;

        int m_maskSize;

        /// @brief 初始化TensortRT模型
        /// @param engineFilePath 模型路径
        void initTensorRT(const std::string& engineFilePath);
}

#endif