#pragma once
#include"common.h"
#include<Eigen/Eigen>


class IntrinsicDecom
{
public:
	IntrinsicDecom();
	~IntrinsicDecom();

	cv::Mat RollingGuidanceFilter(const cv::Mat& color, float sigma_s=30, float sigma_r=10, int iter = 4);
	void doIntrinsicDecomposition(cv::Mat&I, cv::Mat&S, cv::Mat&renNormal, float sigma_c = 0.0001, float sigma_i = 0.8, float sigma_n=0.5);
	void run(cv::Mat&in_color, cv::Mat&RenNormal);
	
private:
	cv::Mat m_ori_colorImg;

	cv::Mat getLaplacian(cv::Mat&S);
	cv::Mat getChromaticity(cv::Mat&I);
};

