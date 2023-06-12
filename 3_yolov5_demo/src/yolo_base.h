#ifndef YOLO_BASE
#define YOLO_BASE
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

/// @brief  深度学习推理结果
class DetectResult {
public: 
	cv::Rect box; // 目标检测框
	int label;  // 标签名称
	float conf; // 置信度
	DetectResult()
	{
		label = -1;
        conf = 0;
	}
};

/// @brief yolo版本
enum class VERSION{
    V5 = 5,
    V8 = 8
};

/// @brief 推理类型
enum class TYPE{
    OPENCV,
    OPENVINO,
    TENSORRT
};

class YoloBase{
    public:
        /// @brief 初始化模型
        /// @param engineFile 模型文件
        /// @param lablesFile   标签文件
        /// @param confThreshold 置信度
        /// @param objScoreThreshold 对象得分阈值
        /// @param labelNames 标签名称列表，如果有标签文件，以标签文件为准，否则使用此列表的标签名称
        virtual void init(const std::string& engineFile,
                                const std::string& lablesFile,
                                const float confThreshold,
                                const float objScoreThreshold,
                                const std::vector<std::string>& labelNames = {}
                                );

        /// @brief 目标检测
        /// @param image 输入图像
        /// @param results 输出结果
        virtual void detect(const cv::Mat& image,std::vector<DetectResult>& results);

        /// @brief 析构函数
        virtual ~YoloBase();

        /// @brief 改变置信度
        virtual void changeConf(float conf);

        /// @brief 改变NMS置信度
        /// @param conf 
        virtual void changeNMSConf(float conf);

    protected:
        /// @brief IOU置信度
        float m_confThreshold = 0.25; 

        /// @brief 目标对象置信度
        float m_objScoreThreshold = 0.25;

        /// @brief NMS置信度
        float m_nmsThreshold = 0.45;

        /// @brief 输入图像尺寸-高
        int m_inputHeight = 640;

        /// @brief 输入图像尺寸-宽
        int m_inputWidth = 640;

        /// @brief 输入尺寸
        int m_inputSize;

        /// @brief 输出尺寸
        int m_outputSize;
        
        /// @brief 模型输出高度
        int m_outputHeight;

        /// @brief 模型输出宽度
        int m_outputWidth;

        /// @brief 图像最小比例
        float m_ratio;

        /// @brief 图像左边填充尺寸
        float m_paddingLeft;

        /// @brief 图像顶面填充尺寸
        float m_paddingTop;

        /// @brief 标签名称列表
        std::vector<std::string> m_labelNames;

        /// @brief 版本
        VERSION m_version;

        /// @brief 推理类型
        TYPE m_type;

        /// @brief 按行读取文件内容
        /// @param filePath 文件路径
        /// @param lines 文件行内容
        void readFileLines(const std::string& filePath,std::vector<std::string>& lines);

        /// @brief 子类重写，实现真正的目标检测
        /// @param inImage 输入图像
        /// @param results 输出结果
        virtual void doDetect(const cv::Mat& inImage, std::vector<DetectResult>& results) = 0;

        /// @brief 子类重写，实现真正的初始化模型
        /// @param modelPath 
        virtual void doInit(const std::string& modelPath) = 0;
};
#endif