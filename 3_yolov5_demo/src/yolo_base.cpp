#include "yolo_base.h"

YoloBase::~YoloBase()
{
}

void YoloBase::changeConf(float conf)
{
    if(m_version == VERSION::V5){
        m_confThreshold = conf;
    }else if (m_version == VERSION::V8){
        m_objScoreThreshold = conf;
    }
}

void YoloBase::changeNMSConf(float conf)
{
    m_nmsThreshold = conf;
}

float YoloBase::sigmodFunction(float a)
{
     return 1. / (1. + exp(-a))
}

void YoloBase::init(const std::string &engineFile, const std::string &lablesFile, const float confThreshold, const float objScoreThreshold,const std::vector<std::string>& labelNames)
{   
    m_confThreshold = confThreshold;
    m_objScoreThreshold = objScoreThreshold;

    // 处理标签名称
    readFileLines(lablesFile,m_labelNames);
    if(m_labelNames.empty()){
        if(labelNames.empty()){
            throw std::invalid_argument("labels should not be empty!");
        }
        m_labelNames = labelNames;
    }

    // 初始化模型
    doInit(engineFile);
}

void YoloBase::readFileLines(const std::string &filePath,std::vector<std::string>& lines)
{
    if(filePath.empty()) return;
    std::ifstream file(filePath);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            lines.emplace_back(line);
        }
        file.close();
    }
}

void YoloBase::detect(const cv::Mat& inImage,std::vector<DetectResult>& results){
    if(inImage.empty()){
        throw std::invalid_argument("input image is empty!");
    }

    cv::Mat rgb,blob;
    cv::cvtColor(inImage,rgb,cv::COLOR_BGR2RGB);
   // 原始图像大小
	size_t srcWidth = inImage.cols;
	size_t srcHeight = inImage.rows;

	// 计算目标图像大小与原始图像宽高的比例，并取其中的较小值
	float ratioW = 1.0 * m_inputWidth / srcWidth;
	float ratioH = 1.0 * m_inputHeight / srcHeight;
	m_ratio = std::min(ratioW, ratioH);

	// 计算图像(真实图像)调整后的大小
	cv::Size newCenterImgSize = cv::Size(m_ratio * srcWidth, m_ratio * srcHeight);
	if (newCenterImgSize.empty()) {
        throw std::invalid_argument("resized image is empty!");
	}
	cv::resize(inImage, blob, newCenterImgSize);

	// 计算填充像素数
	int paddingW = m_inputWidth - newCenterImgSize.width;
	int paddingH = m_inputHeight - newCenterImgSize.height;

	float paddingBootom = 0.0, paddingRight = 0.0;
	m_paddingTop = paddingH / 2;
	paddingBootom = paddingH - m_paddingTop;
	m_paddingLeft = paddingW / 2;
	paddingRight = paddingW - m_paddingLeft;

    // 填充
	cv::copyMakeBorder(blob, blob, m_paddingTop, paddingBootom, m_paddingLeft, paddingRight, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat tensor = cv::dnn::blobFromImage(blob,1.0f/255.f,cv::Size(m_inputWidth,m_inputHeight),cv::Scalar(),true);

    doDetect(tensor,results);
}
