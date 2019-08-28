
#include<iostream>
#include"SensorInterface/RGBDSensor.h"
#include"SensorInterface/KinectOne.h"
#include"SensorInterface/OpenNIControl.h"
#include<algorithm>
#include<GLFW/glfw3.h>
#include"Render.h"
#include"RenderToTexture.h"
#include"JGLib/stringUtil.h"
#include"JGLib/Util.h"
#include"IntrinsicDecom.h"

int g_d_W, g_d_H, g_c_W, g_c_H;

//Rendering window setting
Render *g_pRender = nullptr;
unsigned int SCR_WIDTH ;
unsigned int SCR_HEIGHT;
int g_FrameBufferWidth, g_FrameBufferHeight;
char g_keys[1024];

RenderToTexture *g_pRenToTexture = nullptr;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);


jgr::Image<jgr::rgb> colorImgForDisp;
jgr::Image<jgr::rgb> depthImgForDisp;


//read and write paths
std::string g_pathToDeformedMesh, g_pathToColorImg, g_pathToCalibParam, g_pathToRenderedNormal, g_rootPath;

int g_nStartFrame, g_nEndFrame;

int main(int argc, char** argv)
{
	g_c_W = 1920;
	g_c_H = 1080;
	g_d_W = 512;
	g_d_H = 424;

	SCR_WIDTH = g_c_W;
	SCR_HEIGHT = g_c_H;

	if (argc < 2) {
		//g_pathToDeformedMesh = "D:/DataSet/Tshinghua_DataSet/SVR_L0/Sidekick_SaveResults";
		g_pathToDeformedMesh = "D:/Developed3Dprograms/PG2019/TexGen_non_rigid_0522";
		g_pathToDeformedMesh = g_pathToDeformedMesh + "/NonRigid/";
		//g_rootPath = "D:/DataSet/Tshinghua_DataSet/SVR_L0/Sidekick/";
		g_rootPath = "D:/DataSet/Postech_DataSet/190503_hyomin_dance/";
		g_pathToCalibParam = g_rootPath;
		g_pathToColorImg = g_rootPath + "color/";
		g_pathToRenderedNormal = g_rootPath + "renderedNormal/";
		g_nStartFrame = 0;
		g_nEndFrame = 400;
		
	}
	else
	{
		g_pathToDeformedMesh = std::string(argv[1]);
		g_rootPath = ::string(argv[2]);
		g_pathToCalibParam = g_rootPath;
		g_pathToColorImg = g_rootPath + "color/";
		g_pathToRenderedNormal = g_rootPath + "renderedNormal/";
		g_nStartFrame = atoi(argv[3]);
		g_nEndFrame = atoi(argv[4]);

	}
	//Rerdering Window Setting
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif
														 // glfw window creation
														 // --------------------

														 //Get current video mode of a primary monitor.
	const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Recorder", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwGetFramebufferSize(window, &g_FrameBufferWidth, &g_FrameBufferHeight);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, &key_callback);
	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}


	//colorImgForDisp.alloc(jgr::uint2(g_pRGBDSensor->getColorWidth(), g_pRGBDSensor->getColorHeight()));
	//depthImgForDisp.alloc(jgr::uint2(g_pRGBDSensor->getDepthWidth(), g_pRGBDSensor->getDepthHeight()));
	

	/*g_pRender = new Render(g_d_W, g_d_H, g_c_W, g_c_H);
	g_pRender->setFrameBufferSize(g_FrameBufferWidth, g_FrameBufferHeight);
	g_pRender->loadShaders();*/

	g_pRenToTexture = new RenderToTexture();
	g_pRenToTexture->init(g_pathToDeformedMesh, g_pathToColorImg, g_pathToRenderedNormal, g_pathToCalibParam, g_rootPath, g_d_W, g_d_H, g_c_W, g_c_H);
	int nFrame = g_nStartFrame;
	//save filtered color image


	while (!glfwWindowShouldClose(window))
	{
		
		// render
		// ------
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		//Z-buffering
		glEnable(GL_DEPTH_TEST);
		// Accept fragment if it closer to the camera than the former one
		glDepthFunc(GL_LESS);


		glfwPollEvents();

		processInput(window);



		g_pRenToTexture->run(nFrame++,6);
		
	//	g_pRender->draw(colorImgForDisp, depthImgForDisp);
			
		
		//glfwWaitEventsTimeout(0.2);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);

		if (nFrame >= g_nEndFrame) {
			break;
		}
		
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();

	delete g_pRenToTexture;
	//delete g_pRender;
	return 0;
}

// process all input : query GLFW whether relevant keys are pressed / released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
	g_pRender->setFrameBufferSize(width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	// When a user press the escape key, we set the WindowShouldClose property to true,
	//closing the application.
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS || (key == GLFW_KEY_Q && action == GLFW_PRESS)) {
	
		
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	if (key >= 0 && key < 1024)
	{

		if (action == GLFW_PRESS)
			g_keys[key] = true;
		else if (action == GLFW_RELEASE)
			g_keys[key] = false;


		//Pause
		if (key == GLFW_KEY_P && action == GLFW_PRESS) {
			
		}
	}
}
