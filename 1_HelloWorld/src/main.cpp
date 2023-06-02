#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include <numeric>
#include <fstream>

void testInstall(){
    using namespace nvinfer1;
    using namespace nvonnxparser;
    using namespace cv;
     
    class Logger:public ILogger{
        void log(Severity severity, const char* msg) noexcept{
            if (severity <= Severity::kINFO) {
                std::cout << msg << std::endl;
            }
        }
    } gLogger;

    auto builder = createInferBuilder(gLogger);
    builder->getLogger()->log(nvinfer1::ILogger::Severity::kERROR,"Create Builder...");

}

// 1. 日志模块
using Severity = nvinfer1::ILogger::Severity;

// logger接口
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

// logger实现
void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

// 2. 计算模型的输入、输出所占内存大小，返回单位为字节（byte）
size_t get_memory_size(const nvinfer1::Dims& dims, const int32_t elem_size){
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

struct Result {
    float score;
    cv::Rect box;
    int class_id;
};

void run(){

    Logger logger;
    // 1. 初始化资源
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    // 1.2 读取
    // 从磁盘读取engine文件到内存
    std::ifstream engine_file("", std::ios::binary);
    if (engine_file.fail()) {
        std::cout << "Failed to read model file." << std::endl;
    }

    engine_file.seekg(0, std::ifstream::end);
    auto fsize = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(fsize);
    engine_file.read(engineData.data(), fsize);

    // 1.3 初始化引擎
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine { runtime->deserializeCudaEngine(engineData.data(), fsize) };
    if (mEngine.get() == nullptr) {
        std::cout << "Failed to deserialize CUDA engine." << std::endl;
    }
    
    // 2. 输入图像读取与预处理

    // 3. 引擎推演

    // 4. 输出后处理
}

int main(int argc, char const* argv[]){
    // if (argc != 3) {
    //     std::cout << "Run like this:\n    " << argv[0] << "yolov5s.engine input.jpg" << std::endl;
    //     return -1;
    // }

    testInstall();
    std::cout << "hello cmake" << std::endl;
    
}