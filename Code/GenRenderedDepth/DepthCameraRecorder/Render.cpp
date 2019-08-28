#include "Render.h"
#include"JGLib/JGRLibCommon.h"
#include"JGLib/MatrixConversion.h"
#include"JGLib/Graphic_util.h"
//for texture display
float vertices[] = {
	// positions            // texture coords
	1.f,  1.0f, -1.0f,     1.0f, 1.0f,   // top right
	1.0f, -1.0f, -1.0f,     1.0f, 0.0f,   // bottom right
	-1.0f, -1.0f, -1.0f,     0.0f, 0.0f,   // bottom left
	-1.0f,  1.0f, -1.0f,     0.0f, 1.0f    // top left 
};
unsigned int indices[] = {
	0, 1, 3, // first triangle
	1, 2, 3  // second triangle
};

Render::Render(int d_W, int d_H, int c_W, int c_H)
{
	//buffer objects for texture display
	glGenVertexArrays(1, &m_VAO_2Dimages);
	glGenBuffers(2, m_VBO_2Dimages);
	glBindVertexArray(m_VAO_2Dimages);
	// Position attribute
	glBindBuffer(GL_ARRAY_BUFFER, m_VBO_2Dimages[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	//index 
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VBO_2Dimages[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));


	//textures
	//color image
	glGenTextures(2, m_texturesForImages);
	glBindTexture(GL_TEXTURE_2D, m_texturesForImages[0]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, c_W, c_H, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	//glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	//depth image
	glBindTexture(GL_TEXTURE_2D, m_texturesForImages[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, d_W, d_H, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	//glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);


}


Render::~Render()
{
	glDeleteVertexArrays(1,&m_VAO_2Dimages);
	glDeleteBuffers(2, m_VBO_2Dimages);
	m_imageShProg.deleteProgram();
}

void Render::loadShaders()
{
	//get path of this executable file
	TCHAR szExeFileName[MAX_PATH];
	GetModuleFileName(NULL, szExeFileName, MAX_PATH);
	size_t index = std::string(szExeFileName).find_last_of("\\");
	std::string shaderPath;
	shaderPath = std::string(szExeFileName).substr(0, index) + "\\Shaders\\";
	


	CShader sh_vert, sh_frag;

	if (!sh_vert.loadShader(shaderPath + "VertexShader.vert", GL_VERTEX_SHADER) ||
		!sh_frag.loadShader(shaderPath + "FragmentShader.frag", GL_FRAGMENT_SHADER))
		throw jgr::JGRLIB_EXCEPTION("Load Shader Error");
	jgr::SDK_CHECK_ERROR_GL_HOST();
	
	m_imageShProg.createProgram();
	jgr::SDK_CHECK_ERROR_GL_HOST();
	if (!m_imageShProg.addShaderToProgram(&sh_vert) || !m_imageShProg.addShaderToProgram(&sh_frag))
		throw jgr::JGRLIB_EXCEPTION("Add Shader to Program Error");
	if (!m_imageShProg.linkProgram())
		throw jgr::JGRLIB_EXCEPTION("Link Program Error");
	jgr::SDK_CHECK_ERROR_GL_HOST();

	sh_vert.deleteShader();
	sh_frag.deleteShader();
	jgr::SDK_CHECK_ERROR_GL_HOST();

}

void Render::setFrameBufferSize(int fb_width, int fb_height)
{
	m_FrameBufferWidth = fb_width;
	m_FrameBufferHeight = fb_height;
}

void Render::draw(const jgr::Image<jgr::rgb>& ColorImg, const jgr::Image<jgr::rgb>& DepthImg)
{
	m_imageShProg.useProgram();
	glBindVertexArray(m_VAO_2Dimages);
	jgr::SDK_CHECK_ERROR_GL_HOST();
	glActiveTexture(GL_TEXTURE0);

	glBindTexture(GL_TEXTURE_2D, m_texturesForImages[0]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ColorImg.size.x, ColorImg.size.y, GL_RGB, GL_UNSIGNED_BYTE, &ColorImg.data[0].r);
	glViewport(0, m_FrameBufferHeight - ColorImg.size.y, ColorImg.size.x, ColorImg.size.y);
	
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	jgr::SDK_CHECK_ERROR_GL_HOST();

	glBindTexture(GL_TEXTURE_2D, m_texturesForImages[1]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DepthImg.size.x, DepthImg.size.y, GL_RGB, GL_UNSIGNED_BYTE, &DepthImg.data[0].r);
	glViewport(ColorImg.size.x, m_FrameBufferHeight - DepthImg.size.y, DepthImg.size.x, DepthImg.size.y);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	jgr::SDK_CHECK_ERROR_GL_HOST();

}
