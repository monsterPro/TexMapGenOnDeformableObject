#pragma once
#include"JGLib/Graphic_util.h"
#include"Shaders/Shader.h"
#include"JGLib/MatrixConversion.h"
#include"JGLib/ImageUtils.h"


//render 2D images in main window.
class Render
{
public:
	Render(int d_W, int d_H, int c_W, int c_H);
	~Render();
	void loadShaders();
	void setFrameBufferSize(int fb_width, int fb_height);
	void draw(const jgr::Image<jgr::rgb>& ColorImg, const jgr::Image<jgr::rgb>& DepthImg );
private:
	CShaderProgram m_imageShProg;
	
	
	glm::mat4 m_viewMat, m_projectionMat, m_modelMat;
	float m_far_clip, m_near_clip;

	//framebuffer width & height
	int m_FrameBufferHeight, m_FrameBufferWidth;


	//Texture 
	GLuint m_VAO_2Dimages, m_VBO_2Dimages[2], m_texturesForImages[2];
	int m_d_Img_width, m_d_Img_height;
	

};

