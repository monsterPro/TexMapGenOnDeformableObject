#include "texture.hpp"

inline bool exists_test0(const std::string& name) {
	ifstream f(name.c_str());
	return f.good();
}

GLuint loadBMP_custom(const char * imagepath){

    //printf("Reading image %s\n", imagepath);

    // Data read from the header of the BMP file
    unsigned char header[54];
    unsigned int dataPos;
    unsigned int imageSize;
    unsigned int width, height;
    // Actual RGB data
    unsigned char * data;

    // Open the file
    FILE * file = fopen(imagepath,"rb");
    if (!file)                              {printf("%s could not be opened. Are you in the right directory ?\n", imagepath); getchar(); return 0;}

    // Read the header, i.e. the 54 first bytes

    // If less than 54 bytes are read, problem
    if ( fread(header, 1, 54, file)!=54 ){
        printf("Less than 54. Not a correct BMP file\n");
        return 0;
    }
    // A BMP files always begins with "BM"
    if ( header[0]!='B' || header[1]!='M' ){
        printf("Doesn't begin with BM. Not a correct BMP file\n");
        return 0;
    }
    // Make sure this is a 24bpp file
    if ( *(int*)&(header[0x1E])!=0  )         {printf("Not 24bpp. Not a correct BMP file\n");    return 0;}
    if ( *(int*)&(header[0x1C])!=24 )         {printf("Not 24bpp. Not a correct BMP file\n");    return 0;}

    // Read the information about the image
    dataPos    = *(int*)&(header[0x0A]);
    imageSize  = *(int*)&(header[0x22]);
    width      = *(int*)&(header[0x12]);
    height     = *(int*)&(header[0x16]);

    // Some BMP files are misformatted, guess missing information
    if (imageSize==0)    imageSize=width*height*3; // 3 : one byte for each Red, Green and Blue component
    if (dataPos==0)      dataPos=54; // The BMP header is done that way

    // Create a buffer
    data = new unsigned char [imageSize];

    // Read the actual data from the file into the buffer
    fread(data,1,imageSize,file);

    // Everything is in memory now, the file wan be closed
    fclose (file);

    // Create one OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);

    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Give the image to OpenGL
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

    // OpenGL has now copied the data. Free our own version
    delete [] data;

    // Poor filtering, or ...
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // ... nice trilinear filtering.
    /*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);*/
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glGenerateMipmap(GL_TEXTURE_2D);

    // Return the ID of the texture we just created
    return textureID;
}

GLuint loadPNG_custom(const char * imagepath) {

	//printf("Reading image %s\n", imagepath);

	// Data read from the header of the BMP file
	unsigned int width, height;
	// Actual RGB data
	unsigned char * data;

	cv::Mat image = cv::imread(imagepath, CV_LOAD_IMAGE_UNCHANGED);
	//cv::cvtColor(image, image, CV_BGRA2BGR);
	width = image.cols;
	height = image.rows;

	//dilation
	int radius = 8;
	cv::Mat out(image.rows, image.cols, CV_8UC4);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			out.at<cv::Vec4b>(y, x) = image.at<cv::Vec4b>(y, x);
			if (image.at<cv::Vec4b>(y, x)[3] == 255) continue;

			int cnt = 0, totalCnt = 0;
			double weightSum = 0;
			cv::Vec4d Ave = cv::Vec4d(0, 0, 0, 0);
			for (int dy = -radius; dy <= radius; dy++)
				for (int dx = -radius; dx <= radius; dx++)
				{
					if (0 <= y + dy && y + dy < image.rows && 0 <= x + dx && x + dx < image.cols && dx*dx + dy*dy <= radius*radius) {
						totalCnt++;
						if (image.at<cv::Vec4b>(y + dy, x + dx)[3] == 255) {
							//cout << Img.at<Vec3b>(y + dy, x + dx) << endl;
							Ave = Ave + cv::Vec4d(image.at<cv::Vec4b>(y + dy, x + dx)) / (double)(dx*dx + dy*dy);
							weightSum += 1. / (dx*dx + dy*dy);
							cnt++;
						}
					}
				}
			if (cnt != 0) {
				out.at<cv::Vec4b>(y, x) = cv::Vec4b(Ave / weightSum);
				out.at<cv::Vec4b>(y, x)[3] = 255;
			}

		}
	}
	cv::imwrite((string)imagepath + "dil.png", out);
	cv::cuda::GpuMat d_src(out);
	cv::cuda::GpuMat d_dst;

	cv::cuda::bilateralFilter(d_src, d_dst, 60, 20, 30);
	cv::Mat dst(d_dst);
	cv::imwrite((string)imagepath + "bil.png", dst);
	cv::cvtColor(dst, image, CV_BGRA2BGR);


	data = new unsigned char[width * height * 3];
	memcpy(data, image.data, sizeof(unsigned char) * width * height * 3);
	// Create one OpenGL texture
	GLuint textureID;
	glGenTextures(1, &textureID);

	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Give the image to OpenGL
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

	// OpenGL has now copied the data. Free our own version
	delete[] data;

	// Poor filtering, or ...
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// ... nice trilinear filtering.
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);*/
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glGenerateMipmap(GL_TEXTURE_2D);

	// Return the ID of the texture we just created
	return textureID;
}

GLuint loadPNG_edgesmooth(const char * imagepath, vector<glm::vec2> tex_vec, vector<GLuint> elements) {

	//printf("Reading image %s\n", imagepath);

	// Data read from the header of the BMP file
	unsigned int width, height;
	// Actual RGB data
	unsigned char * data;

	cv::Mat image = cv::imread(imagepath, CV_LOAD_IMAGE_UNCHANGED);
	width = image.cols;
	height = image.rows;
	if (exists_test0((string)imagepath + "edgesmoo.png") && is_viewer) {
		image = cv::imread((string)imagepath + "edgesmoo.png");
		cv::cvtColor(image, image, CV_BGRA2BGR);
	}
	else {
		//cv::cvtColor(image, image, CV_BGRA2BGR);

		//dilation
		int radius = 5;
		cv::Mat out(image.rows, image.cols, CV_8UC4);
		for (int y = 0; y < image.rows; y++) {
			for (int x = 0; x < image.cols; x++) {
				out.at<cv::Vec4b>(y, x) = image.at<cv::Vec4b>(y, x);
				if (image.at<cv::Vec4b>(y, x)[3] == 255) continue;

				int cnt = 0, totalCnt = 0;
				double weightSum = 0;
				cv::Vec4d Ave = cv::Vec4d(0, 0, 0, 0);
				for (int dy = -radius; dy <= radius; dy++)
					for (int dx = -radius; dx <= radius; dx++)
					{
						if (0 <= y + dy && y + dy < image.rows && 0 <= x + dx && x + dx < image.cols && dx*dx + dy*dy <= radius*radius) {
							totalCnt++;
							if (image.at<cv::Vec4b>(y + dy, x + dx)[3] == 255) {
								//cout << Img.at<Vec3b>(y + dy, x + dx) << endl;
								Ave = Ave + cv::Vec4d(image.at<cv::Vec4b>(y + dy, x + dx)) / (double)(dx*dx + dy*dy);
								weightSum += 1. / (dx*dx + dy*dy);
								cnt++;
							}
						}
					}
				if (cnt != 0) {
					out.at<cv::Vec4b>(y, x) = cv::Vec4b(Ave / weightSum);
					out.at<cv::Vec4b>(y, x)[3] = 255;
				}

			}
		}
		cv::imwrite((string)imagepath + "dil.png", out);
		//cv::cvtColor(out, image, CV_BGRA2BGR);
		cv::Mat4b tmpImage = out.clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
		float thresh_edge = 5.0 / image.rows;
		for (int i = 0; i < elements.size(); i += 3) {
			float minX, maxX, minY, maxY;
			minX = min(tex_vec[elements[i]].x, min(tex_vec[elements[i + 1]].x, tex_vec[elements[i + 2]].x));
			minY = min(tex_vec[elements[i]].y, min(tex_vec[elements[i + 1]].y, tex_vec[elements[i + 2]].y));
			maxX = max(tex_vec[elements[i]].x, max(tex_vec[elements[i + 1]].x, tex_vec[elements[i + 2]].x));
			maxY = max(tex_vec[elements[i]].y, max(tex_vec[elements[i + 1]].y, tex_vec[elements[i + 2]].y));

			for (int y = minY * image.rows; y < maxY * image.rows; y++) {
				for (int x = minX * image.cols; x < maxX * image.cols; x++) {
					glm::vec2 point((float)x / image.cols, (float)y / image.rows);
					glm::vec2 v1(tex_vec[elements[i]]);
					glm::vec2 v2(tex_vec[elements[i + 1]]);
					glm::vec2 v3(tex_vec[elements[i + 2]]);
					float d1 = glm::length((point - v1) - glm::dot((point - v1), (v2 - v1)) / glm::dot((v2 - v1), (v2 - v1)) * (v2 - v1));
					float d2 = glm::length((point - v2) - glm::dot((point - v2), (v3 - v2)) / glm::dot((v3 - v2), (v3 - v2)) * (v3 - v2));
					float d3 = glm::length((point - v3) - glm::dot((point - v3), (v1 - v3)) / glm::dot((v1 - v3), (v1 - v3)) * (v1 - v3));
					if (d1 < thresh_edge || d2 < thresh_edge || d3 < thresh_edge) {
						float dd = min(d1, min(d2, d3));
						out.at<cv::Vec4b>(y, x) = (thresh_edge - dd) / thresh_edge * tmpImage.at<cv::Vec4b>(y, x) + dd / thresh_edge *out.at<cv::Vec4b>(y, x);
					}
				}
			}

		}

		cv::imwrite((string)imagepath + "edgesmoo.png", out);

		cv::cuda::GpuMat d_src(out);
		cv::cuda::GpuMat d_dst;

		//cv::cuda::bilateralFilter(d_src, d_dst, 30, 20, 30);
		cv::cuda::bilateralFilter(d_src, d_dst, 5, 20, 10);
		cv::Mat dst(d_dst);
		cv::imwrite((string)imagepath + "bil.png", dst);



		//cv::cvtColor(dst, image, CV_BGRA2BGR);
		cv::cvtColor(out, image, CV_BGRA2BGR);
	}
	


	data = new unsigned char[width * height * 3];
	memcpy(data, image.data, sizeof(unsigned char) * width * height * 3);
	// Create one OpenGL texture
	GLuint textureID;
	glGenTextures(1, &textureID);

	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Give the image to OpenGL
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

	// OpenGL has now copied the data. Free our own version
	delete[] data;

	// Poor filtering, or ...
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// ... nice trilinear filtering.
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);*/
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glGenerateMipmap(GL_TEXTURE_2D);

	// Return the ID of the texture we just created
	return textureID;
}

// Since GLFW 3, glfwLoadTexture2D() has been removed. You have to use another texture loading library,
// or do it yourself (just like loadBMP_custom and loadDDS)
//GLuint loadTGA_glfw(const char * imagepath){
//
//  // Create one OpenGL texture
//  GLuint textureID;
//  glGenTextures(1, &textureID);
//
//  // "Bind" the newly created texture : all future texture functions will modify this texture
//  glBindTexture(GL_TEXTURE_2D, textureID);
//
//  // Read the file, call glTexImage2D with the right parameters
//  glfwLoadTexture2D(imagepath, 0);
//
//  // Nice trilinear filtering.
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//  glGenerateMipmap(GL_TEXTURE_2D);
//
//  // Return the ID of the texture we just created
//  return textureID;
//}


#define FOURCC_DXT1 0x31545844 // Equivalent to "DXT1" in ASCII
#define FOURCC_DXT3 0x33545844 // Equivalent to "DXT3" in ASCII
#define FOURCC_DXT5 0x35545844 // Equivalent to "DXT5" in ASCII

GLuint loadDDS(const char * imagepath){

    unsigned char header[124];

    FILE *fp;

    /* try to open the file */
    fp = fopen(imagepath, "rb");
    if (fp == NULL){
        printf("%s could not be opened. Are you in the right directory ? Don't forget to read the FAQ !\n", imagepath); getchar();
        return 0;
    }

    /* verify the type of file */
    char filecode[4];
    fread(filecode, 1, 4, fp);
    if (strncmp(filecode, "DDS ", 4) != 0) {
        fclose(fp);
        return 0;
    }

    /* get the surface desc */
    fread(&header, 124, 1, fp);

    unsigned int height      = *(unsigned int*)&(header[8 ]);
    unsigned int width       = *(unsigned int*)&(header[12]);
    unsigned int linearSize  = *(unsigned int*)&(header[16]);
    unsigned int mipMapCount = *(unsigned int*)&(header[24]);
    unsigned int fourCC      = *(unsigned int*)&(header[80]);


    unsigned char * buffer;
    unsigned int bufsize;
    /* how big is it going to be including all mipmaps? */
    bufsize = mipMapCount > 1 ? linearSize * 2 : linearSize;
    buffer = (unsigned char*)malloc(bufsize * sizeof(unsigned char));
    fread(buffer, 1, bufsize, fp);
    /* close the file pointer */
    fclose(fp);

    unsigned int components  = (fourCC == FOURCC_DXT1) ? 3 : 4;
    unsigned int format;
    switch(fourCC)
    {
    case FOURCC_DXT1:
        format = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
        break;
    case FOURCC_DXT3:
        format = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
        break;
    case FOURCC_DXT5:
        format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
        break;
    default:
        free(buffer);
        return 0;
    }

    // Create one OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);

    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, textureID);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    unsigned int blockSize = (format == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT) ? 8 : 16;
    unsigned int offset = 0;

    /* load the mipmaps */
    for (unsigned int level = 0; level < mipMapCount && (width || height); ++level)
    {
        unsigned int size = ((width+3)/4)*((height+3)/4)*blockSize;
        glCompressedTexImage2D(GL_TEXTURE_2D, level, format, width, height,
            0, size, buffer + offset);

        offset += size;
        width  /= 2;
        height /= 2;

        // Deal with Non-Power-Of-Two textures. This code is not included in the webpage to reduce clutter.
        if(width < 1) width = 1;
        if(height < 1) height = 1;

    }

    free(buffer);

    return textureID;


}
