#include "yolov5_seg_trt.h"

Yolov5SegTrt:Yolov5SegTrt(){

}

Yolov5SegTrt::~Yolov5SegTrt()
{
    cudaStreamSynchronize(m_cudaStream);
    cudaStreamDestroy(m_cudaStream);

    for(auto it=m_cudaBuffers.begin();it !=m_cudaBuffers.end();it++){
        if(it){
            delete *it;
            *it = nullptr;
        }
    }

    if(!m_context){
        delete m_context;
    }
    if(!m_engine){
        delete m_engine;
    }

    if(!m_runtime){
        delete m_runtime;
    }
}

void Yolov5SegTrt::doDetect(const cv::Mat &inImage, std::vector<DetectResult> &results)
{
    // 输出
    std::vector<float> prob,mdata;
    prob.resize(m_outputSize);
    mdata.resize(m_maskSize);

    // 内存到显存
    cudaMemcpyAsync(m_cudaBuffers[m_inputIndex],inImage.ptr<float>(),m_inputSize*sizeof(float),cudaMemcpyHostToDevice,m_cudaStream);

    // 推理
    bool refResult = m_context->enqueueV2(m_cudaBuffers,m_cudaStream,nullptr);
    std::cout << "ref result: " << refResult << std::endl;

    // 显存到内存
    cudaMemcpyAsync(prob.data(),m_cudaBuffers[m_outputIndex],m_outputSize*sizeof(float),cudaMemcpyDeviceToHost,m_cudaStream);
    cudaMemcpyAsync(mdata.data(),m_cudaBuffers[m_maskIndex],m_maskSize*sizeof(float),cudaMemcpyDeviceToHost,m_cudaStream);

    // 后处理 6300 * 6 
    cv::Mat probMat(m_outputHeight,m_outputWidth,CV_32F,(float*)prob.data());
    // 32 * 25600
    cv::Mat maskMat(m_maskChannels,m_maskWidth * m_maskHeight,CV_32F,(float*)mdata.data());

    std::vector<int> classIds;//结果id数组  
    std::vector<cv::Mat> masks; 
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	int net_width = m_labelNames.size() + 5;  //输出的网络宽度是类别数+5

    int sx = 160.0 / m_inputWidth;
    int sy = 160.0 / m_inputHeight;

    for(int i=0;i<probMat.rows;i++){
        float confidence = probMat.at<float>(i,4);
        if(confidence < m_confThreshold)
            continue;
        cv::Mat classesScores=probMat.row(i).colRange(5,net_width);
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classesScores, 0, &score, 0, &classIdPoint);
        if (score > m_objScoreThreshold) {
            //rect [x,y,w,h]
            float cx = probMat.at<float>(i,0);  //cx
            float cy = probMat.at<float>(i,1);   //cy
            float w = probMat.at<float>(i,2);   //w
            float h = probMat.at<float>(i,3);  //h
            int left = (cx - 0.5 * w - m_paddingLeft) / m_ratio;
            int top = (cy - 0.5 * h - m_paddingTop) / m_ratio;
            classIds.emplace_back(classIdPoint.x);
            confidences.emplace_back(confidence * score);
            boxes.emplace_back(cv::Rect(left, top, int(w / m_ratio), int(h / m_ratio)));

            // 1 * 32
            cv::Mat mask =probMat.row(i).colRange(m_outputWidth - 32, m_outputWidth);
            masks.emplace_back(mask);
        }
    }

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		DetectResult result;
		result.box = box;
		result.conf = confidences[idx];
		result.label = classIds[idx];

        // 解析语义分割mask 
        cv::Mat mask = masks[idx];

        int x1 = std::max(0,box.x);
        int y1 = std::max(0,box.y);
        int x2 = std::max(0,box.br().x);
        int y2 = std::max(0,box.br().y);

        cv::Mat m = mask * maskMat; // 1*32 * 32*25600 -> 1*25600
        for(int col = 0; col<m.cols;col++){
            m.at<float>(0,col) = sigmodFunction(m.at<float>(0,col)); // 0-160
        }

        cv::Mat m1 = m.reshape(1,160);
        
		results.emplace_back(result);
	}
}

void Yolov5SegTrt::doInit(const std::string &engineFile)
{
    initTensorRT(engineFile);
}

void Yolov5SegTrt::initTensorRT(const std::string &engineFilePath)
{
    class Logger:public nvinfer1::ILogger{
        void log(Severity severity, const char* msg) noexcept{
            if (severity <= Severity::kINFO) {
                std::cout << msg << std::endl;
            }
        }
    } gLogger;

    if(engineFilePath.empty()){
        throw std::invalid_argument("engine file path is empty!");
    }
    std::ifstream engineStream(engineFilePath, std::ios::binary);

    if (engineStream.fail()) {
        //std::cerr << "Failed to read model file:" << engineFile << std::endl;
        throw std::invalid_argument("Failed to read model file:" + engineFilePath);
    }

    engineStream.seekg(0, std::ifstream::end);
    auto fsize = engineStream.tellg();
    engineStream.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(fsize);
    engineStream.read(engineData.data(), fsize);
    engineStream.close();

     //1 创建runtimex
    m_runtime = nvinfer1::createInferRuntime(gLogger);

    // 2 创建引擎
    m_engine = m_runtime->deserializeCudaEngine(engineData.data(),fsize);
    if (!m_engine) {
        throw std::runtime_error("Failed to deserialize CUDA engine.");
    }

    // 3 创建执行上下文
    m_context = m_engine->createExecutionContext();
    if(m_context == nullptr){
       throw std::runtime_error("Failed to create CUDA context.");
    }

    // 4 获取输入、输出的索引
    m_inputIndex = m_engine->getBindingIndex("images");
    m_outputIndex = m_engine->getBindingIndex("output0");
    m_maskIndex = m_engine->getBindingIndex("output1");

    // 4.1 获取输入维度信息 NCHW = 1 * 3 * 320 * 320
    int input_batch = m_engine->getBindingDimensions(m_inputIndex).d[0];
    int input_channel = m_engine->getBindingDimensions(m_inputIndex).d[1];
    m_inputHeight = m_engine->getBindingDimensions(m_inputIndex).d[2];
    m_inputWidth = m_engine->getBindingDimensions(m_inputIndex).d[3];
    m_inputSize = input_batch * input_channel * m_inputHeight * m_inputWidth;
    std::cout << "input,batch:" << input_batch << " height:" << m_inputHeight << " width:" << m_inputWidth << std::endl;

    // 4.2 获取输出维度
    int output_batch = m_engine->getBindingDimensions(m_outputIndex).d[0];
    m_outputHeight = m_engine->getBindingDimensions(m_outputIndex).d[1];
    m_outputWidth = m_engine->getBindingDimensions(m_outputIndex).d[2];
    m_outputSize = output_batch * m_outputHeight * m_outputWidth;
    std::cout << "output,batch:" <<output_batch <<  " height:" << m_outputHeight << " width:" << m_outputWidth << std::endl;

    // 掩码尺寸
    m_maskChannels = m_engine->getBindingDimensions(m_maskIndex)[1];
    m_maskWidth = m_engine->getBindingDimensions(m_maskIndex)[2];
    m_maskHeight = m_engine->getBindingDimensions(m_maskIndex)[3];
    m_maskSize = m_maskChannels * m_maskWidth * m_maskHeight;
     // 输入
    cudaMalloc(&m_cudaBuffers[m_inputIndex],m_inputSize * sizeof(float));
    // 输出
    cudaMalloc(&m_cudaBuffers[m_outputIndex],m_outputSize * sizeof(float));
    // 掩码
    cudaMalloc(&m_cudaBuffers[m_maskIndex],m_maskSize * sizeof(float));

    // 创建cuda流
    cudaStreamCreate(&m_cudaStream);
}
