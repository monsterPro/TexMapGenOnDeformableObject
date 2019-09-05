#pragma once

#ifndef MAPPER4D_H
#define MAPPER4D_H

#include <iostream>
#include <fstream>
#include <memory>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include "any.hpp"

#include "TriangleAndVertex.h"

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

using namespace std;
using namespace OpenMesh;


// For temporal sampling...
// not use

static float CalcBlurMetric(const cv::Mat& img)
{
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_BGR2GRAY);
	//img_gray = img;
	img_gray.convertTo(img_gray, CV_32F, 1 / 255.0f);

	int r = img_gray.rows;
	int c = img_gray.cols;

	cv::Mat Hv, Hh;
	Hv = cv::Mat::ones(1, 5, CV_32F) / 5.0f;
	cv::transpose(Hv, Hh);

	cv::Mat B_Ver, B_Hor;
	cv::filter2D(img_gray, B_Ver, CV_32F, Hv);
	cv::filter2D(img_gray, B_Hor, CV_32F, Hh);

	cv::Mat D_F_Ver, D_F_Hor;
	D_F_Ver = cv::abs(img_gray(cv::Rect(0, 0, c, r - 1)) - img_gray(cv::Rect(0, 1, c, r - 1)));
	D_F_Hor = cv::abs(img_gray(cv::Rect(0, 0, c - 1, r)) - img_gray(cv::Rect(1, 0, c - 1, r)));

	cv::Mat D_B_Ver, D_B_Hor;
	D_B_Ver = cv::abs(B_Ver(cv::Rect(0, 0, c, r - 1)) - B_Ver(cv::Rect(0, 1, c, r - 1)));
	D_B_Hor = cv::abs(B_Hor(cv::Rect(0, 0, c - 1, r)) - B_Hor(cv::Rect(1, 0, c - 1, r)));

	cv::Mat V_Ver, V_Hor;
	V_Ver = cv::max(0, D_F_Ver - D_B_Ver);
	V_Hor = cv::max(0, D_F_Hor - D_B_Hor);

	float S_D_Ver = (float)cv::sum(D_F_Ver(cv::Rect(1, 1, c - 2, r - 2)))[0];
	float S_D_Hor = (float)cv::sum(D_F_Hor(cv::Rect(1, 1, c - 2, r - 2)))[0];
	float S_V_Ver = (float)cv::sum(V_Ver(cv::Rect(1, 1, c - 2, r - 2)))[0];
	float S_V_Hor = (float)cv::sum(V_Hor(cv::Rect(1, 1, c - 2, r - 2)))[0];

	float blur = std::max((S_D_Ver - S_V_Ver) / S_D_Ver, (S_D_Hor - S_V_Hor) / S_D_Hor);
	return blur;
}
static float CalcBlurMetric(const cv::Mat& img, int count)
{
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_BGR2GRAY);
	//img_gray = img;
	img_gray.convertTo(img_gray, CV_32F, 1 / 255.0f);

	int r = img_gray.rows;
	int c = img_gray.cols;

	cv::Mat Hv, Hh;
	Hv = cv::Mat::ones(1, 5, CV_32F) / 5.0f;
	cv::transpose(Hv, Hh);

	cv::Mat B_Ver, B_Hor;
	cv::filter2D(img_gray, B_Ver, CV_32F, Hv);
	cv::filter2D(img_gray, B_Hor, CV_32F, Hh);

	cv::Mat D_F_Ver, D_F_Hor;
	D_F_Ver = cv::abs(img_gray(cv::Rect(0, 0, c, r - 1)) - img_gray(cv::Rect(0, 1, c, r - 1)));
	D_F_Hor = cv::abs(img_gray(cv::Rect(0, 0, c - 1, r)) - img_gray(cv::Rect(1, 0, c - 1, r)));

	cv::Mat D_B_Ver, D_B_Hor;
	D_B_Ver = cv::abs(B_Ver(cv::Rect(0, 0, c, r - 1)) - B_Ver(cv::Rect(0, 1, c, r - 1)));
	D_B_Hor = cv::abs(B_Hor(cv::Rect(0, 0, c - 1, r)) - B_Hor(cv::Rect(1, 0, c - 1, r)));

	cv::Mat V_Ver, V_Hor;
	V_Ver = cv::max(0, D_F_Ver - D_B_Ver);
	V_Hor = cv::max(0, D_F_Hor - D_B_Hor);

	float S_D_Ver = (float)cv::sum(D_F_Ver(cv::Rect(1, 1, c - 2, r - 2)))[0];
	float S_D_Hor = (float)cv::sum(D_F_Hor(cv::Rect(1, 1, c - 2, r - 2)))[0];
	float S_V_Ver = (float)cv::sum(V_Ver(cv::Rect(1, 1, c - 2, r - 2)))[0];
	float S_V_Hor = (float)cv::sum(V_Hor(cv::Rect(1, 1, c - 2, r - 2)))[0];

	float blur = std::max((S_D_Ver - S_V_Ver) / S_D_Ver, (S_D_Hor - S_V_Hor) / S_D_Hor);
	return blur / count;
}
static cv::Mat TransferColor(cv::Mat& content_b, cv::Scalar style_mean, cv::Scalar style_std)
{
	cv::Mat3f content;
	content_b.convertTo(content, CV_32FC3, 1.0 / 255.0f);

	cv::Mat3f content_lab;
	cv::cvtColor(content, content_lab, CV_BGR2Lab);
	cv::Scalar content_mean, content_std;
	cv::meanStdDev(content_lab, content_mean, content_std);
	//printf("%f %f %f, %f %f %f\n", style_mean[0], style_mean[1], style_mean[2], style_std[0], style_std[1], style_std[2]);
	content_lab = (content_lab - content_mean);
	cv::Mat1f content_labs[3];
	cv::split(content_lab, content_labs);
	for (int i = 0; i < 3; i++)
		content_labs[i] = content_labs[i] + style_mean[i]; /// content_std[i] * style_std[i]
	cv::merge(content_labs, 3, content_lab);
	cv::Mat3f res_f;
	cv::cvtColor(content_lab, res_f, CV_Lab2BGR);
	cv::Mat res;
	res_f.convertTo(res, CV_8UC3, 255.0);
	content_b = res.clone();
	return res;
}
static cv::Mat TransferColor(cv::Mat& content_b, cv::Mat& content_c, cv::Scalar style_mean, cv::Scalar style_std)
{
	cv::Mat3f content;
	content_b.convertTo(content, CV_32FC3, 1.0 / 255.0f);

	cv::Mat3f content_lab;
	cv::cvtColor(content, content_lab, CV_BGR2Lab);
	cv::Scalar content_mean, content_std;
	cv::meanStdDev(content_c, content_mean, content_std);
	//printf("%f %f %f, %f %f %f\n", style_mean[0], style_mean[1], style_mean[2], style_std[0], style_std[1], style_std[2]);
	content_lab = (content_lab - content_mean);
	cv::Mat1f content_labs[3];
	cv::split(content_lab, content_labs);
	for (int i = 0; i < 3; i++)
		content_labs[i] = content_labs[i] + style_mean[i]; /// content_std[i] * style_std[i]
	cv::merge(content_labs, 3, content_lab);
	cv::Mat3f res_f;
	cv::cvtColor(content_lab, res_f, CV_Lab2BGR);
	cv::Mat res;
	res_f.convertTo(res, CV_8UC3, 255.0);
	content_b = res.clone();
	return res;
}
static void TransferColor(vector<cv::Mat>& content_b, vector<cv::Mat>& content_c)
{
	cv::Scalar style_mean;
	cv::Scalar style_std;
	vector<cv::Scalar> content_mean_vec;
	vector<cv::Scalar> content_std_vec;
	vector<cv::Mat3f> content_lab_vec;
	for (int i = 0; i < content_c.size(); i++) {
		cv::Mat3f content, content_target;
		content_b[i].convertTo(content_target, CV_32FC3, 1.0 / 255.0f);
		content_c[i].convertTo(content, CV_32FC3, 1.0 / 255.0f);
		cv::Mat3f content_lab, content_lab_target;
		cv::cvtColor(content, content_lab, CV_BGR2Lab);
		cv::cvtColor(content_target, content_lab_target, CV_BGR2Lab);
		cv::Scalar content_mean, content_std;
		cv::meanStdDev(content_lab, content_mean, content_std);
		content_mean_vec.push_back(content_mean);
		content_std_vec.push_back(content_std);
		style_mean += content_mean;
		style_std += content_std;
		content_lab_target = (content_lab_target - content_mean);
		content_lab_vec.push_back(content_lab_target);
	}
	style_mean[0] /= content_c.size();
	style_mean[1] /= content_c.size();
	style_mean[2] /= content_c.size();
	style_std[0] /= content_c.size();
	style_std[1] /= content_c.size();
	style_std[2] /= content_c.size();
	for (int i = 0; i < content_b.size(); i++) {
		cv::Mat1f content_labs[3];
		cv::split(content_lab_vec[i], content_labs);
		for (int i = 0; i < 3; i++)
			content_labs[i] = content_labs[i] + style_mean[i]; /// content_std[i] * style_std[i]
		cv::merge(content_labs, 3, content_lab_vec[i]);
		cv::Mat3f res_f;
		cv::cvtColor(content_lab_vec[i], res_f, CV_Lab2BGR);
		cv::Mat res;
		res_f.convertTo(res, CV_8UC3, 255.0);
		content_b[i] = res.clone();
	}
}

namespace TexMap {

	class Mapper4D {
	public:
		Mapper4D() {}
		~Mapper4D() {}

		Mapper4D(string templateMeshPath, string NR_MeshPathAndPrefix, string streamPath, int startIdx, int endIdx);

		void ConstructVertree_majorVote();

		void SetPropVec(vector<vector<float2>> propVec, int layer);

		void SpatiotemporalSampleAndCalpos(string streamPath, string NR_MeshPathAndPrefix, int startIdx, int endIdx);

		void temporalframeSample(string streamPath, vector<cv::Mat>& outColor, vector<cv::Mat>& outDepth, vector<float> &outBlur, vector<int>& outIdx, int startIdx, int endIdx);

		void Get_VT_layer(vector<hostVertex> &hVs, vector<hostTriangle> &hTs, int layer);
		
		void SaveResult(string outDir, string FileName);

		void GetNumInfo(int * nV, int * nT, int * nI);

		void GetNumInfo_layer(int * nV, int * nT, int layer);

		void ShowResult();

		std::vector<cv::Mat> depthImages_dummy;
		std::vector<cv::Mat> colorImages_dummy;

		std::vector<cv::Mat> colorImages;
		std::vector<cv::Mat> depthImages;
		cv::Mat _Pose = cv::Mat::eye(4, 4, CV_32F);

	private:

		string m_templateMeshPath;
		string m_streamPath;
		string m_NR_MeshPathAndPrefix;

		MyMesh template_mesh;
		OpenMesh::IO::Options opt;

		cv::Mat _IR_CO;
		cv::Mat _CO_IR;

		unsigned int imgNum = 0;
		unsigned int cW, cH, dW, dH;
		vector<cv::Mat> c_dImages_temporal;
		vector<cv::Mat> c_dImages;
		vector<MyMesh> layer_mesh_vec;

		vector<hostVertex> _Vertices;
		vector<hostTriangle> _Triangles;

		vector<vector<hostVertex>> layer_Vertices;
		vector<vector<hostTriangle>> layer_Triangles;

		int numTri;
		int numVer;

		vector<int> layer_numTri;
		vector<int> layer_numVer;

		vector<vector<float2>> propAccum;
		vector<vector<bool>> propValid;

		float Distance(float3 &_Pos, float3 &_Pos_s) {
			return sqrtf((_Pos.x - _Pos_s.x) * (_Pos.x - _Pos_s.x) + (_Pos.y - _Pos_s.y) * (_Pos.y - _Pos_s.y) + (_Pos.z - _Pos_s.z) * (_Pos.z - _Pos_s.z));
		}

		float Distance(float2 &_Pos, float2 &_Pos_s) {
			return sqrtf((_Pos.x - _Pos_s.x) * (_Pos.x - _Pos_s.x) + (_Pos.y - _Pos_s.y) * (_Pos.y - _Pos_s.y));
		}

		void Assign(float3 &_Pos, MyMesh::FaceVertexIter &_fv, MyMesh &mesh) {
			_Pos.x = mesh.point(*_fv)[0];
			_Pos.y = mesh.point(*_fv)[1];
			_Pos.z = mesh.point(*_fv)[2];
		}

		void Transform(float3 &_Src, cv::Mat &_P, float3 &_Des) {
			_Des.x = _Src.x * _P.at<float>(0, 0) + _Src.y * _P.at<float>(0, 1) + _Src.z * _P.at<float>(0, 2) + _P.at<float>(0, 3);
			_Des.y = _Src.x * _P.at<float>(1, 0) + _Src.y * _P.at<float>(1, 1) + _Src.z * _P.at<float>(1, 2) + _P.at<float>(1, 3);
			_Des.z = _Src.x * _P.at<float>(2, 0) + _Src.y * _P.at<float>(2, 1) + _Src.z * _P.at<float>(2, 2) + _P.at<float>(2, 3);
		}
		void PointToPixel_IR(float3 &_Point, float2 &_Pixel) {
			_Pixel.x = _Point.x / _Point.z * IRFLX + IRCPX;
			_Pixel.y = _Point.y / _Point.z * IRFLY + IRCPY;
		}
		void PointToPixel_CO(float3 &_Point, float2 &_Pixel) {
			_Pixel.x = _Point.x / _Point.z * COFLX + COCPX;
			_Pixel.y = _Point.y / _Point.z * COFLY + COCPY;
		}

		void PixelToPoint_IR(float3 &_Pixel, float3 &_Point) {
			_Point.x = (_Pixel.x - IRCPX) / IRFLX * _Pixel.z;
			_Point.y = (_Pixel.y - IRCPY) / IRFLY * _Pixel.z;
			_Point.z = _Pixel.z;
		}
		void PixelToPoint_CO(float3 &_Pixel, float3 &_Point) {
			_Point.x = (_Pixel.x - COCPX) / COFLX * _Pixel.z;
			_Point.y = (_Pixel.y - COCPY) / COFLY * _Pixel.z;
			_Point.z = _Pixel.z;
		}
	};
}


#endif MAPPER4D_H