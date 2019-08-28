
#include "RenderToTexture.h"
#include"JGLib/Util.h"
#include"JGLib/MatrixConversion.h"
#include"IntrinsicDecom.h"

RenderToTexture::RenderToTexture()
{
}


RenderToTexture::~RenderToTexture()
{
	m_normalRenderer.deleteProgram();
	m_normalDepthRenderer.deleteProgram();

	glDeleteVertexArrays(1, &m_VAO);
	glDeleteBuffers(1, &m_vertex_buffer);
	glDeleteBuffers(1, &m_indices_buffer);
	jgr::SDK_CHECK_ERROR_GL_HOST();



	glDeleteRenderbuffers(1, &m_rboDepth);
	glDeleteTextures(2, m_renderedTexture);
	glDeleteFramebuffers(1, &m_fbo);
	jgr::SDK_CHECK_ERROR_GL_HOST();

}

void RenderToTexture::loadShaders()
{
	//get path of this executable file
	TCHAR szExeFileName[MAX_PATH];
	GetModuleFileName(NULL, szExeFileName, MAX_PATH);
	size_t index = std::string(szExeFileName).find_last_of("\\");
	std::string shaderPath;
	index = std::string(szExeFileName).substr(0, index).find_last_of("\\");
	shaderPath = std::string(szExeFileName).substr(0, index) + "\\Shaders\\";



	CShader sh_vert, sh_frag;

	if (!sh_vert.loadShader(shaderPath + "normalRenderShader.vert", GL_VERTEX_SHADER) ||
		!sh_frag.loadShader(shaderPath + "normalRenderShader.frag", GL_FRAGMENT_SHADER))
		throw jgr::JGRLIB_EXCEPTION("Load Shader Error");
	jgr::SDK_CHECK_ERROR_GL_HOST();

	m_normalRenderer.createProgram();
	jgr::SDK_CHECK_ERROR_GL_HOST();
	if (!m_normalRenderer.addShaderToProgram(&sh_vert) || !m_normalRenderer.addShaderToProgram(&sh_frag))
		throw jgr::JGRLIB_EXCEPTION("Add Shader to Program Error");
	if (!m_normalRenderer.linkProgram())
		throw jgr::JGRLIB_EXCEPTION("Link Program Error");
	jgr::SDK_CHECK_ERROR_GL_HOST();

	sh_vert.deleteShader();
	sh_frag.deleteShader();
	jgr::SDK_CHECK_ERROR_GL_HOST();


	if (!sh_vert.loadShader(shaderPath + "normaDepthlRenderShader.vert", GL_VERTEX_SHADER) ||
		!sh_frag.loadShader(shaderPath + "normaDepthlRenderShader.frag", GL_FRAGMENT_SHADER))
		throw jgr::JGRLIB_EXCEPTION("Load Shader Error");
	jgr::SDK_CHECK_ERROR_GL_HOST();

	m_normalDepthRenderer.createProgram();
	jgr::SDK_CHECK_ERROR_GL_HOST();
	if (!m_normalDepthRenderer.addShaderToProgram(&sh_vert) || !m_normalDepthRenderer.addShaderToProgram(&sh_frag))
		throw jgr::JGRLIB_EXCEPTION("Add Shader to Program Error");
	if (!m_normalDepthRenderer.linkProgram())
		throw jgr::JGRLIB_EXCEPTION("Link Program Error");
	jgr::SDK_CHECK_ERROR_GL_HOST();

	sh_vert.deleteShader();
	sh_frag.deleteShader();
	jgr::SDK_CHECK_ERROR_GL_HOST();

}

void RenderToTexture::readCalibParamFromText(const std::string& path)
{
	std::string c_intrPath, d_intrPath, d2cPath;
	d2cPath = path + "Extrinsic.txt";
	c_intrPath = path + "ColorIntrinsic.txt";
	d_intrPath = path + "DepthIntrinsic.txt";

	int cnt = 0;
	std::ifstream file;
	file.open(c_intrPath, std::ifstream::in);
	if (file.is_open()) {
		std::string line, value;
		while (getline(file, line))
		{
			std::stringstream linestream(line);
			while (getline(linestream, value, ' ')) {
				m_c_proj_mat(cnt / 3, cnt - (cnt / 3) * 3) = std::stof(value);
				cnt++;
			}
		}
	}
	file.close();

	file.open(d_intrPath, std::ifstream::in);
	cnt = 0;
	if (file.is_open()) {
		std::string line, value;
		while (getline(file, line))
		{
			std::stringstream linestream(line);
			while (getline(linestream, value, ' ')) {
				m_d_proj_mat(cnt / 3, cnt - (cnt / 3) * 3) = std::stof(value);
				cnt++;
			}
		}
	}
	file.close();

	file.open(d2cPath, std::ifstream::in);
	cnt = 0;
	if (file.is_open()) {
		std::string line, value;
		while (getline(file, line))
		{
			std::stringstream linestream(line);
			while (getline(linestream, value, ' ')) {
				m_d2c_mat(cnt / 4, cnt - (cnt / 4) * 4) = std::stof(value);
				cnt++;
			}
		}
	}
	file.close();

	std::cout << m_c_proj_mat << std::endl;
	std::cout << m_d_proj_mat << std::endl;
	std::cout << m_d2c_mat << std::endl;
}

void RenderToTexture::bindFBO()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
	jgr::SDK_CHECK_ERROR_GL_HOST();
}

void RenderToTexture::unbindFBO()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	jgr::SDK_CHECK_ERROR_GL_HOST();
}


void RenderToTexture::setClearColor(const std::vector<std::array<float,4>>& clearColors)
{
	m_clearColors = clearColors;
}

void RenderToTexture::clearFramebuffer()
{
	glClearBufferfv(GL_COLOR, 0, &m_clearColors[warpVertex][0]);
	glClearBufferfv(GL_COLOR, 1, &m_clearColors[warpNormal][0]);
	glClearBufferfv(GL_DEPTH, 0, &m_clearColors[renDepth][0]); //only x value is used for "clear" operation
	jgr::SDK_CHECK_ERROR_GL_HOST();

}


cv::Mat RenderToTexture::getRenVertexFromTexture() const
{
	cv::Mat warp_vertex_data(m_c_height, m_c_width, CV_32FC4);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	jgr::SDK_CHECK_ERROR_GL_HOST();
	glReadPixels(0, 0, m_c_width, m_c_height, GL_RGBA, GL_FLOAT, warp_vertex_data.data);
	jgr::SDK_CHECK_ERROR_GL_HOST();
	cv::Mat warp_vertex_data_vflipped;
	cv::flip(warp_vertex_data, warp_vertex_data_vflipped, 0);
	return warp_vertex_data_vflipped;
}

cv::Mat RenderToTexture::getRenNormalFromTexture() const
{
	cv::Mat warp_normal_data(m_c_height, m_c_width, CV_32FC4);
	glReadBuffer(GL_COLOR_ATTACHMENT1);
	jgr::SDK_CHECK_ERROR_GL_HOST();
	glReadPixels(0, 0, m_c_width, m_c_height, GL_RGBA, GL_FLOAT, warp_normal_data.data);
	jgr::SDK_CHECK_ERROR_GL_HOST();
	cv::Mat warp_normal_data_vflipped;
	cv::flip(warp_normal_data, warp_normal_data_vflipped, 0);
	return warp_normal_data_vflipped;
}


void RenderToTexture::renderToFrameBuffer(jgr::Meshf& deformedMesh)
{
	//Preparing data part
	glBindVertexArray(m_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);	
	glBufferData(GL_ARRAY_BUFFER, deformedMesh.m_vertices.size() * sizeof(jgr::Vertex<float>), &deformedMesh.m_vertices[0], GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indices_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, deformedMesh.m_indices.size() * sizeof(unsigned int), &deformedMesh.m_indices[0], GL_DYNAMIC_DRAW);
	jgr::SDK_CHECK_ERROR_GL_HOST();


	//position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(jgr::Vertex<float>), (void*)0);

	//normal
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(jgr::Vertex<float>), (void*)offsetof(jgr::Vertex<float>, Normal));
	jgr::SDK_CHECK_ERROR_GL_HOST();
	
	//Render part
	glViewport(0, 0, m_c_width, m_c_height);
	//The call order of glClear() and glDrawBuffers&glClearBufferfv is important !
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glDisable(GL_DEPTH_CLAMP);

	GLenum draw_buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, draw_buffers);
	clearFramebuffer();

	
	m_normalRenderer.useProgram();
	//set uniform variables
	glUniformMatrix4fv(glGetUniformLocation(m_normalRenderer.getProgramID(), "viewMat"), 1, GL_FALSE, glm::value_ptr(m_viewMat));
	glUniformMatrix4fv(glGetUniformLocation(m_normalRenderer.getProgramID(), "c_projectionMat"), 1, GL_FALSE, glm::value_ptr(m_cProjMat));
	glUniformMatrix4fv(glGetUniformLocation(m_normalRenderer.getProgramID(), "d2cMat"), 1, GL_FALSE, glm::value_ptr(m_d2cMat));
	jgr::SDK_CHECK_ERROR_GL_HOST();

	glDrawElements(GL_TRIANGLES, deformedMesh.m_indices.size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);


}

void RenderToTexture::run(unsigned int nFrame, unsigned int ndigits)
{
	bindFBO();

	std::string filename ="Frame_"+jgr::util::zeroPad(nFrame, 3);

	jgr::Meshf deformedMesh = jgr::MeshIO<float>::loadFromFile(m_pathTodeformedMesh + filename+".off");
	deformedMesh.calcVertexNormal();
	deformedMesh.setupIndicesFromTriFacesIndices();
	renderToFrameBuffer(deformedMesh);

	
	//save filtered color image
	//cv::Mat ori_colorImg = cv::imread(m_pathToColorImg + filename + ".png");// , cv::IMREAD_UNCHANGED);
	//cv::cuda::GpuMat d_src(ori_colorImg);
	//cv::cuda::GpuMat d_dst;

	//cv::cuda::bilateralFilter(d_src, d_dst, 60, 10, 30);
	//cv::Mat dst(d_dst);
	//cv::imwrite(m_rootPath + "filteredColor/" + filename + ".png", dst);

	//save results
	cv::Mat Ren_vertices, Ren_normals;
	Ren_vertices = getRenVertexFromTexture();
	Ren_normals = getRenNormalFromTexture();

	cv::Mat smoothDepth(m_c_height, m_c_width, CV_16U);
	cv::Mat smoothNormal(m_c_height, m_c_width, CV_8UC3);
	for (int y = 0; y < m_c_height; y++)
		for (int x = 0; x < m_c_width; x++) {
			smoothDepth.at<unsigned short>(y, x) = Ren_vertices.at<cv::Vec4f>(y, x)[2] * 1000;
			if (smoothDepth.at<unsigned short>(y, x) == 0)
				smoothNormal.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			else
				smoothNormal.at<cv::Vec3b>(y, x) = cv::Vec3b((Ren_normals.at<cv::Vec4f>(y, x)[2]+1.)/2. *255, (Ren_normals.at<cv::Vec4f>(y, x)[1] + 1.) / 2. * 255, (Ren_normals.at<cv::Vec4f>(y, x)[0] + 1.) / 2. * 255);
		}

	filename = "Frame_" + jgr::util::zeroPad(nFrame, 3);
	cv::imwrite(m_pathToRenNormal + filename+ ".png", smoothNormal);
	cv::imwrite(m_rootPath + "renderedDepth/" + filename+ ".png", smoothDepth);


	IntrinsicDecom intrin;
	filename = "Frame_" + jgr::util::zeroPad(nFrame, ndigits);
	intrin.run(cv::imread(m_pathToColorImg + filename + ".png", cv::IMREAD_UNCHANGED), Ren_normals);


	unbindFBO();

}

void RenderToTexture::init(std::string pathToDeformedMesh, std::string pathToColorImg, std::string pathToRenNormal, std::string pathToCalibParam, std::string rootPath, int d_width, int d_height, int c_width, int c_height)
{

	m_pathTodeformedMesh = pathToDeformedMesh;
	m_pathToCalibParam = pathToCalibParam;
	m_pathToColorImg = pathToColorImg;
	m_pathToRenNormal = pathToRenNormal;
	m_rootPath = rootPath;

	std::cout << "m_rootPath :" << m_rootPath << std::endl;
	std::cout << "m_pathTodeformedMesh :" << m_pathTodeformedMesh << std::endl;
	std::cout << "m_pathToCalibParam :"<< m_pathToCalibParam << std::endl;
	std::cout << "m_pathToColorImg :"<< m_pathToColorImg << std::endl;
	std::cout << "m_pathToRenNormal : " << m_pathToRenNormal << std::endl;
	//make folder to save rendered normal maps
	if (!jgr::util::directoryExists(pathToRenNormal)) {
		jgr::util::makeDirectory(pathToRenNormal);
	}

	//filtered image
	if (!jgr::util::directoryExists(rootPath+"filteredColor/")) {
		jgr::util::makeDirectory(rootPath+"filteredColor/");
	}
	
	//rendered depth
	if (!jgr::util::directoryExists(rootPath + "renderedDepth/")) {
		jgr::util::makeDirectory(rootPath + "renderedDepth/");
	}

	m_d_width = d_width;
	m_d_height = d_height;
	m_c_width = c_width;
	m_c_height = c_height;

	m_near_clip = 0.002;
	m_far_clip = 10.0f;

	m_cProjMat = m_dProjMat = m_viewMat = m_modelMat = m_d2cMat= glm::mat4(1.);

	readCalibParamFromText(pathToCalibParam);
	m_d2cMat = jgr::toGLMMat4(m_d2c_mat);

	setProjectionMatrices();
	loadShaders();



	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_vertex_buffer);
	glGenBuffers(1, &m_indices_buffer);
	
	//generate and bind the framebuffer object
	glGenFramebuffers(1, &m_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
	jgr::SDK_CHECK_ERROR_GL_HOST();

	glGenTextures(2, m_renderedTexture);

	glBindTexture(GL_TEXTURE_2D, m_renderedTexture[warpVertex]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_c_width, m_c_height, 0, GL_RGBA, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	jgr::SDK_CHECK_ERROR_GL_HOST();

	glBindTexture(GL_TEXTURE_2D, m_renderedTexture[warpNormal]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_c_width, m_c_height, 0, GL_RGBA, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	jgr::SDK_CHECK_ERROR_GL_HOST();

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_renderedTexture[warpVertex], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_renderedTexture[warpNormal], 0);
	jgr::SDK_CHECK_ERROR_GL_HOST();

	//create render buffer object to attach to declared framebuffer object
	glGenRenderbuffers(1, &m_rboDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, m_rboDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, m_c_width, m_c_height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_rboDepth);
	jgr::SDK_CHECK_ERROR_GL_HOST();

	// Set draw buffers
	GLenum draw_buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, draw_buffers);

	// Check completeness of this framebuffer
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		throw jgr::JGRLIB_EXCEPTION("fatal error: framebuffer is not complete !");
	
	//unbinding
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_clearColors.resize(3);
	std::array<float, 4> init_val = { 0.f, 0.f, 0.f, 1.f };
	m_clearColors[warpVertex] = init_val;
	m_clearColors[warpNormal] = init_val;
	init_val[0] = 1.f;
	m_clearColors[renDepth] = init_val;


}

void RenderToTexture::setProjectionMatrices() {
	float d_fx, d_fy, d_cx, d_cy;
	float c_fx, c_fy, c_cx, c_cy;
	
	
	d_fx = m_d_proj_mat(0, 0);
	d_fy = m_d_proj_mat(1, 1);
	d_cx = m_d_proj_mat(0, 2);
	d_cy = m_d_proj_mat(1, 2);

	c_fx = m_c_proj_mat(0, 0);
	c_fy = m_c_proj_mat(1, 1);
	c_cx = m_c_proj_mat(0, 2);
	c_cy = m_c_proj_mat(1, 2);
	

	//glm's matrix is stored in column-major order
	m_dProjMat[0][0] = 2 * d_fx / m_d_width;
	m_dProjMat[1][0] = 0.0f;
	m_dProjMat[2][0] = (m_d_width - 2 * d_cx) / m_d_width;
	m_dProjMat[3][0] = 0.0f;

	m_dProjMat[0][1] = 0.0f;
	m_dProjMat[1][1] = 2 * d_fy / m_d_height;
	m_dProjMat[2][1] = (2 * d_cy - m_d_height) / m_d_height;
	m_dProjMat[3][1] = 0.0f;

	m_dProjMat[0][2] = 0.0f;
	m_dProjMat[1][2] = 0.0f;
	m_dProjMat[2][2] = -(m_far_clip + m_near_clip) / (m_far_clip - m_near_clip);
	m_dProjMat[3][2] = -2 * m_near_clip*m_far_clip / (m_far_clip - m_near_clip);

	m_dProjMat[0][3] = 0.0f;
	m_dProjMat[1][3] = 0.0f;
	m_dProjMat[2][3] = -1.0f;
	m_dProjMat[3][3] = 0.0f;



	m_cProjMat[0][0] = 2 * c_fx / m_c_width;
	m_cProjMat[1][0] = 0.0f;
	m_cProjMat[2][0] = (m_c_width - 2 * c_cx) / m_c_width;
	m_cProjMat[3][0] = 0.0f;

	m_cProjMat[0][1] = 0.0f;
	m_cProjMat[1][1] = 2 * c_fy / m_c_height;
	m_cProjMat[2][1] = (2 * c_cy - m_c_height) / m_c_height;
	m_cProjMat[3][1] = 0.0f;

	m_cProjMat[0][2] = 0.0f;
	m_cProjMat[1][2] = 0.0f;
	m_cProjMat[2][2] = -(m_far_clip + m_near_clip) / (m_far_clip - m_near_clip);
	m_cProjMat[3][2] = -2 * m_near_clip*m_far_clip / (m_far_clip - m_near_clip);

	m_cProjMat[0][3] = 0.0f;
	m_cProjMat[1][3] = 0.0f;
	m_cProjMat[2][3] = -1.0f;
	m_cProjMat[3][3] = 0.0f;

}