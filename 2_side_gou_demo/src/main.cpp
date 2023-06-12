#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "NvOnnxParser.h"
#include <numeric>
#include <fstream>

class BoxItem {
public: 
	cv::Rect box;
	int label;
	float conf;
	BoxItem()
	{
		label = -1;
        conf = 0;
	}
};


int main(int argc,char** argv){

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

    //1 创建runtime
    std::unique_ptr<nvinfer1::IRuntime> runtime{createInferRuntime(gLogger)};

    // 2 读取模型文件
    std::ifstream engine_file("/HardDisk/DEV/CODE/tensorrt8.6_toturial/models/side_gou.engine", std::ios::binary);
    if (engine_file.fail()) {
        std::cout << "Failed to read model file." << std::endl;
        return -1;
    }

    engine_file.seekg(0, std::ifstream::end);
    auto fsize = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(fsize);
    engine_file.read(engineData.data(), fsize);
    engine_file.close();
    
    // 3 创建引擎
    std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(engineData.data(),fsize)};
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        return -1;
    }

    // 3.1 创建执行上下文
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if(context.get() == nullptr){
        std::cerr << "Failed to create CUDA context." << std::endl;
        return -1;
    }

    double start = static_cast<double>(getTickCount());
    // 4 获取输入、输出的索引
    int input_index = engine->getBindingIndex("images");
    int output_index = engine->getBindingIndex("output0");

    // 4.1 获取输入维度信息 NCHW = 1 * 3 * 320 * 320
    int input_batch = engine->getBindingDimensions(input_index).d[0];
    int input_channel = engine->getBindingDimensions(input_index).d[1];
    int input_height = engine->getBindingDimensions(input_index).d[2];
    int input_width = engine->getBindingDimensions(input_index).d[3];
    int input_size = input_batch * input_channel * input_height * input_width;
    std::cout << "input,batch:" << input_batch << " height:" << input_height << " width:" << input_width << std::endl;

    // 4.2 获取输出维度
    int output_batch = engine->getBindingDimensions(output_index).d[0];
    int output_height = engine->getBindingDimensions(output_index).d[1];
    int output_width = engine->getBindingDimensions(output_index).d[2];
    int output_size = output_batch * output_height * output_width;
    std::cout << "output,batch:" <<output_batch <<  " height:" << output_height << " width:" << output_width << std::endl;

    // 5 为输入输出创建缓存
    std::cout << "input/output:" << engine->getNbBindings() << std::endl;
    // 输入
    void* buffers[2] = {NULL,NULL};
    cudaMalloc(&buffers[0],input_size * sizeof(float));
    // 输出
    cudaMalloc(&buffers[1],output_size * sizeof(float));
    
    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cv::Mat inImage = cv::imread("/HardDisk/DEV/CODE/tensorrt8.6_toturial/images/gou2.jpg");
    if(inImage.empty()) {
        std::cerr << "image empty" << std::endl;
        return -1;
    }

    cv::Mat rgb,blob;
    cv::cvtColor(inImage,rgb,cv::COLOR_BGR2RGB);
   // 原始图像大小
	size_t srcWidth = inImage.cols;
	size_t srcHeight = inImage.rows;

	// 计算目标图像大小与原始图像宽高的比例，并取其中的较小值
	float ratioW = 1.0 * input_width / srcWidth;
	float ratioH = 1.0 * input_height / srcHeight;
	float ratio = std::min(ratioW, ratioH);

	// 计算图像(真实图像)调整后的大小
	cv::Size newCenterImgSize = cv::Size(ratio * srcWidth, ratio * srcHeight);
	if (newCenterImgSize.empty()) {
		throw "调整后图片大小为空";
	}
	cv::resize(inImage, blob, newCenterImgSize);

	// 计算填充像素数
	int paddingW = input_width - newCenterImgSize.width;
	int paddingH = input_height - newCenterImgSize.height;

	float paddingTop = 0.0, paddingBootom = 0.0, paddingLeft = 0.0, paddingRight = 0.0;
	paddingTop = paddingH / 2;
	paddingBootom = paddingH - paddingTop;
	paddingLeft = paddingW / 2;
	paddingRight = paddingW - paddingLeft;

	// 填充
	cv::copyMakeBorder(blob, blob, paddingTop, paddingBootom, paddingLeft, paddingRight, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat tensor = cv::dnn::blobFromImage(blob,1.0f/255.f,cv::Size(input_width,input_height),cv::Scalar(),true);

    //cv::subtract(blob,cv::Scalar(),blob);
    //cv::divide(blob,cv::Scalar(),blob);
    std::cout << "input size, width:" << blob.cols << " height:" << blob.rows << std::endl; 
    //cv::Mat tensor = cv::dnn::blobFromImage(blob); // HWC -> CHW

    std::vector<float> prob;
    prob.resize(output_size);

    // 内存到显存
    cudaMemcpyAsync(buffers[input_index],tensor.ptr<float>(),input_size*sizeof(float),cudaMemcpyHostToDevice,stream);

    // 推理
    bool refResult = context->enqueueV2(buffers,stream,nullptr);
    std::cout << "ref result: " << refResult << std::endl;

    // 显存到内存
    cudaMemcpyAsync(prob.data(),buffers[output_index],output_size*sizeof(float),cudaMemcpyDeviceToHost,stream);

    int probSize  = prob.size();
    // 后处理 6300 * 6 
    cv::Mat probMat(output_height,output_width,CV_32F,(float*)prob.data());

    // cv::imshow("sourceImage",inImage);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	int net_width = 1 + 5;  //输出的网络宽度是类别数+5

    for(int i=0;i<probMat.rows;i++){
        float confidence = probMat.at<float>(i,4);
        if(confidence < 0.25)
            continue;
        cv::Mat classesScores=probMat.row(i).colRange(5,net_width);
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classesScores, 0, &score, 0, &classIdPoint);
        if (score > 0.45) {
            //rect [x,y,w,h]
            float cx = probMat.at<float>(i,0);  //cx
            float cy = probMat.at<float>(i,1);   //cy
            float w = probMat.at<float>(i,2);   //w
            float h = probMat.at<float>(i,3);  //h
            int left = (cx - 0.5 * w - paddingLeft) / ratio;
            int top = (cy - 0.5 * h - paddingTop) / ratio;
            classIds.push_back(classIdPoint.x);
            confidences.push_back(confidence * score);
            boxes.push_back(cv::Rect(left, top, int(w / ratio), int(h / ratio)));
        }
    }

    cv::Mat outImage = inImage.clone();
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		BoxItem boxItem;
		boxItem.box = box;
		boxItem.conf = confidences[idx];
		boxItem.label = classIds[idx];
		std::cout  << "object label:" << boxItem.label << " conf:" << boxItem.conf << std::endl;

        cv::rectangle(outImage,box,cv::Scalar(0,255,0),2,cv::LINE_AA);
	}

    std::cout << "cost time:" << ((double)getTickCount() - start) / getTickFrequency()  * 1000<< " ms" << std::endl;
    cv::imwrite("/HardDisk/DEV/CODE/tensorrt8.6_toturial/images/gou2_output.jpg",outImage);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if(!context){
        context->destroy();
    }
    if(!engine){
        engine->destroy();
    }

    if(!runtime){
        runtime->destroy();
    }

    if(!buffers[0]){
        delete[] buffers;
    }
    return 0;
}
