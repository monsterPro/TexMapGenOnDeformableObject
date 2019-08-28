#include "IntrinsicDecom.h"
#include<ANN/ANN.h>


IntrinsicDecom::IntrinsicDecom()
{
}


IntrinsicDecom::~IntrinsicDecom()
{
}

cv::Mat IntrinsicDecom::RollingGuidanceFilter(const cv::Mat& color, float sigma_s, float sigma_r, int iter /*= 4*/)
{
	//cv::Mat filtered_r, filtered_g, filtered_b;
	//std::vector<cv::Mat> channels(3);
	//cv::Mat res(color.rows, color.cols, color.type(), 0.f);

	////split img
	//cv::split(color, channels);
	////b
	//filtered_b = channels[0];
	////g
	//filtered_g = channels[1];
	////r
	//filtered_r = channels[2];

	cv::cuda::GpuMat d_src(color);
	cv::cuda::GpuMat d_dst;

	for (int i = 0; i < iter; i++) {
		cv::cuda::bilateralFilter(d_src, d_dst, sigma_s * 2, sigma_r, sigma_s);
		d_dst.copyTo(d_src);

		//cv::Mat iterRes(d_src);
		//cv::imshow(std::to_string(i), iterRes);
		//cv::waitKey(0);
	}
	
	cv::Mat res(d_src);
	//cv::imshow("filtered", res);
	//cv::waitKey(0);
	return res;
}

cv::Mat IntrinsicDecom::getChromaticity(cv::Mat&I)
{
	cv::Mat newI;
	if (I.type() != CV_64FC3) {
		I.convertTo(newI, CV_64FC3, 1 / 255.);
	}
	else
		I.copyTo(newI);
	cv::Mat intensity;
	cv::Mat Isquared = I.mul(I);
	std::vector < cv::Mat> channels;
	cv::split(Isquared, channels);
	cv::max(cv::max(channels[2], cv::max(channels[1], channels[0])), pow(0.1, 10));
	////b
	//filtered_b = channels[0];
	////g
	//filtered_g = channels[1];
	////r
	//filtered_r = channels[2];
	//

}

void IntrinsicDecom::doIntrinsicDecomposition(cv::Mat&I, cv::Mat&S, cv::Mat&renNormal, float sigma_c /*= 0.0001*/, float sigma_i /*= 0.8*/, float sigma_n /*= 0.5*/)
{
	
}

void IntrinsicDecom::run(cv::Mat&in_color, cv::Mat&RenNormal)
{
	ANNpointArray dataPts;
	ANNpoint queryPt;
	ANNidxArray nnIdx;
	ANNdistArray dists;
	ANNkd_tree * kdTree;

	queryPt = annAllocPt(5);
	dataPts = annAllocPts(34, 5);
	nnIdx = new ANNidx[5];
	dists = new ANNdist[5];

	cv::Mat S= RollingGuidanceFilter(in_color,5, 40,6);

	doIntrinsicDecomposition(in_color, S, RenNormal);


	annClose();
}