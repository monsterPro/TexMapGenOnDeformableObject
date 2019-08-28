#pragma once

#include"common.h"
#include"JGLib/JGRLibCommon.h"
#include"JGLib/Graphic_util.h"
#include"Shaders/Shader.h"
#include"JGLib/Mesh.h"

class RenderToTexture
{
public:
	RenderToTexture();
	~RenderToTexture();
	
	void init(std::string pathToDeformedMesh, std::string pathToColorImg, std::string pathToRenNormal, std::string pathToCalibParam, std::string rootPath, int d_width, int d_height, int c_width, int c_height);
	
	void run(unsigned int nFrame, unsigned int ndigits=3);
	void renderToFrameBuffer(jgr::Meshf& deformedMesh);
	void bindFBO();
	void unbindFBO();

	void setClearColor(const std::vector<std::array<float, 4>>& clearColors);
	void clearFramebuffer();

	cv::Mat getRenVertexFromTexture()const;
	cv::Mat getRenNormalFromTexture()const;


	enum TexType { warpVertex = 0, warpNormal, renDepth};

private:
	GLuint m_fbo;
	GLuint m_rboDepth;//render buffer object
	GLuint m_VAO, m_vertex_buffer, m_indices_buffer;
	GLuint m_renderedTexture[2];

	//two values to initialize two color buffers, one value to initialize depth buffer.
	std::vector<std::array<float, 4>> m_clearColors;

	CShaderProgram m_normalRenderer, m_normalDepthRenderer;

	std::string m_pathTodeformedMesh, m_pathToColorImg, m_pathToRenNormal, m_pathToCalibParam, m_rootPath;
	float m_near_clip, m_far_clip;
	//calib parameters
	Eigen::Matrix3f m_c_proj_mat;
	Eigen::Matrix3f m_d_proj_mat;
	Eigen::Matrix4f m_d2c_mat;
	int m_d_width, m_d_height, m_c_width, m_c_height;

	glm::mat4 m_cProjMat, m_dProjMat, m_viewMat, m_modelMat, m_d2cMat;

	void readCalibParamFromText(const std::string& path);
	void setProjectionMatrices();
	void loadShaders();
};

