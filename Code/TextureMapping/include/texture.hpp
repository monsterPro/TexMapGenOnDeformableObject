#ifndef TEXTURE_HPP
#define TEXTURE_HPP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <opencv2\opencv.hpp>

#include <GLFW/glfw3.h>
#include "Optimizer.cuh"
// Load a .BMP file using our custom loader
GLuint loadBMP_custom(const char * imagepath);
GLuint loadPNG_custom(const char * imagepath);
GLuint loadPNG_edgesmooth(const char * imagepath, vector<glm::vec2> tex_vec, vector<GLuint> elements);
//// Since GLFW 3, glfwLoadTexture2D() has been removed. You have to use another texture loading library, 
//// or do it yourself (just like loadBMP_custom and loadDDS)
//// Load a .TGA file using GLFW's own loader
//GLuint loadTGA_glfw(const char * imagepath);

// Load a .DDS file using GLFW's own loader
GLuint loadDDS(const char * imagepath);


#endif