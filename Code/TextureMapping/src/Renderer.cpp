#include "Renderer.h"

using namespace TexMap;

Renderer::Renderer() {

}
Renderer::Renderer(Optimizer* optimizer) {
	this->optimizer = optimizer;
}

glm::vec3 Renderer::get_arcball_vector(int x, int y) {
	glm::vec3 P = glm::vec3(1.0 * x / screen_width * 2 - 1.0,
		1.0 * y / screen_height * 2 - 1.0,
		0);
	P.y = -P.y;
	float OP_squared = P.x * P.x + P.y * P.y;

	if (OP_squared <= 1 * 1) {
		P.z = sqrt(1 * 1 - OP_squared);    // Pythagore
	}
	else {
		P = glm::normalize(P);    // nearest point
	}

	return P;
}

void Renderer::ScreenCapture(const char *strFilePath)
{
	//비트맵 파일 처리를 위한 헤더 구조체
	BITMAPFILEHEADER	BMFH;
	BITMAPINFOHEADER	BMIH;

	int nWidth = 0;
	int nHeight = 0;
	unsigned long dwQuadrupleWidth = 0;		//LJH 추가, 가로 사이즈가 4의 배수가 아니라면 4의 배수로 만들어서 저장

	GLbyte *pPixelData = NULL;				//front buffer의 픽셀 값들을 얻어 오기 위한 버퍼의 포인터

											//4의 배수인지 아닌지 확인해서 4의 배수가 아니라면 4의 배수로 맞춰준다.
	dwQuadrupleWidth = (screen_width % 4) ? ((screen_width)+(4 - (screen_width % 4))) : (screen_width);

	//비트맵 파일 헤더 처리
	BMFH.bfType = 0x4D42;		//B(42)와 M(4D)에 해당하는 ASCII 값을 넣어준다.
								//바이트 단위로 전체파일 크기
	BMFH.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (dwQuadrupleWidth * 3 * screen_height);
	//영상 데이터 위치까지의 거리
	BMFH.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	//비트맵 인포 헤더 처리
	BMIH.biSize = sizeof(BITMAPINFOHEADER);		//이 구조체의 크기
	BMIH.biWidth = screen_width;							//픽셀 단위로 영상의 폭
	BMIH.biHeight = screen_height;							//영상의 높이
	BMIH.biPlanes = 1;								//비트 플레인 수(항상 1)
	BMIH.biBitCount = 24;								//픽셀당 비트수(컬러, 흑백 구별)
	BMIH.biCompression = BI_RGB;							//압축 유무
	BMIH.biSizeImage = dwQuadrupleWidth * 3 * screen_height;	//영상의 크기
	BMIH.biXPelsPerMeter = 0;								//가로 해상도
	BMIH.biYPelsPerMeter = 0;								//세로 해상도
	BMIH.biClrUsed = 0;								//실제 사용 색상수
	BMIH.biClrImportant = 0;								//중요한 색상 인덱스

	pPixelData = new GLbyte[dwQuadrupleWidth * 3 * screen_height];	//LJH 수정

																	//프런트 버퍼로 부터 픽셀 정보들을 얻어온다.
	glReadPixels(
		0, 0,					//캡처할 영역의 좌측상단 좌표
		screen_width, screen_height,		//캡처할 영역의 크기
		GL_BGR,					//캡처한 이미지의 픽셀 포맷
		GL_UNSIGNED_BYTE,		//캡처한 이미지의 데이터 포맷
		pPixelData				//캡처한 이미지의 정보를 담아둘 버퍼 포인터
	);

	{//저장 부분
		FILE *outFile = fopen(strFilePath, "wb");
		if (outFile == NULL)
		{
			//에러 처리
			//printf( "에러" );
			//fclose( outFile );
		}

		fwrite(&BMFH, sizeof(char), sizeof(BITMAPFILEHEADER), outFile);			//파일 헤더 쓰기
		fwrite(&BMIH, sizeof(char), sizeof(BITMAPINFOHEADER), outFile);			//인포 헤더 쓰기
		fwrite(pPixelData, sizeof(unsigned char), BMIH.biSizeImage, outFile);	//glReadPixels로 읽은 데이터 쓰기

		fclose(outFile);	//파일 닫기
	}

	if (pPixelData != NULL)
	{
		delete pPixelData;
	}
}

void Renderer::onSpecial(int key, int x, int y) {
	int modifiers = glutGetModifiers();

	if ((modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT) {
		strife = 1;
	}
	else {
		strife = 0;
	}

	if ((modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT) {
		speed_factor = 0.1;
	}
	else {
		speed_factor = 1;
	}

	switch (key) {
	case GLUT_KEY_F1:
		view_mode = MODE_OBJECT;
		break;

	case GLUT_KEY_F2:
		view_mode = MODE_CAMERA;
		break;

	case GLUT_KEY_F3:
		view_mode = MODE_LIGHT;
		break;

	case GLUT_KEY_LEFT:
		//rotY_direction = 1;
		main_object4D.down_time_stamp();
		break;

	case GLUT_KEY_RIGHT:
		//rotY_direction = -1;
		main_object4D.up_time_stamp();
		break;

	case GLUT_KEY_UP:
		transZ_direction = 1;
		break;

	case GLUT_KEY_DOWN:
		transZ_direction = -1;
		break;

	case GLUT_KEY_PAGE_UP:
		rotX_direction = -1;
		break;

	case GLUT_KEY_PAGE_DOWN:
		rotX_direction = 1;
		break;

	case GLUT_KEY_HOME:
		init_view();
		break;

	case GLUT_KEY_END:
		main_object4D.set_time_stamp(0);
		break;
	}
}

void Renderer::onKeyboard(unsigned char key, int x, int y) {
	switch (key) {
	case ' ':
		object4D_play = !object4D_play;
		break;

	case 'N':
	case 'n':
		main_object4D._flags.y == 0 ? main_object4D._flags.y = 1 : main_object4D._flags.y = 0;
		break;

	case 'M':
	case 'm':
		main_object4D._flags.x == 0 ? main_object4D._flags.x = 1 : main_object4D._flags.x = 0;
		break;

	case 'A':
	case 'a':
		rotY_direction = 1;
		break;

	case 'D':
	case 'd':
		rotY_direction = -1;
		break;

	case 'W':
	case 'w':
		transZ_direction = 1;
		break;

	case 'S':
	case 's':
		transZ_direction = -1;
		break;

	case 'C':
	case 'c':
		tmp_s = glm::to_string(transforms[MODE_CAMERA]);
		tmp_s.erase(remove(tmp_s.begin(), tmp_s.end(), '('), tmp_s.end());
		tmp_s.erase(remove(tmp_s.begin(), tmp_s.end(), ')'), tmp_s.end());
		cout << tmp_s << endl;
		break;

	case 'O':
	case 'o':
		tmp_s = glm::to_string(main_object.object2world);
		tmp_s.erase(remove(tmp_s.begin(), tmp_s.end(), '('), tmp_s.end());
		tmp_s.erase(remove(tmp_s.begin(), tmp_s.end(), ')'), tmp_s.end());
		cout << tmp_s << endl;
		break;
		////////////////////////////////////////////
		////////// 과제 위해 특별히 수정
		////////////////////////////////////////////
	case 'r':
	case 'R':
		ScreenCapture(string(data_root_path + "/capture/view_" + viewnum + currentTime() + ".png").c_str());
		cout << string(data_root_path + "/capture/view_" + viewnum + currentTime() + ".png") << "   Saved" << endl;
		break;

	case '1':		
		main_object.object2world = glm::mat4(1);
		main_object4D.object2world = glm::mat4(1);
		transforms[MODE_CAMERA] = glm::lookAt(
			glm::vec3(0.0, 0.0, 0.0) - objmov,   // eye
			glm::vec3(0.0, 0.0, 1.0),   // direction
			glm::vec3(0.0, -1.0, 0.0));  // up
		viewnum = "1";
		break;

	case '2':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[2]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[2]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[2]);
		viewnum = "2";
		break;

	case '3':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[3]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[3]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[3]);
		viewnum = "3";
		break;
	case '4':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[4]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[4]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[4]);
		viewnum = "4";
		break;
	case '5':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[5]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[5]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[5]);
		viewnum = "5";
		break;
	case '6':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[6]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[6]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[6]);
		viewnum = "6";
		break;
	case '7':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[7]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[7]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[7]);
		viewnum = "7";
		break;
	case '8':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[8]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[8]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[8]);
		viewnum = "8";
		break;
	case '9':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[9]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[9]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[9]);
		viewnum = "9";
		break;
	case '0':
		transforms[MODE_CAMERA] = glm::make_mat4(rendering_spots_c[0]);
		main_object.object2world = glm::make_mat4(rendering_spots_o[0]);
		main_object4D.object2world = glm::make_mat4(rendering_spots_o[0]);
		viewnum = "0";
		break;
	case 27:
		exit(0);
		break;
	}
}

void Renderer::onSpecialUp(int key, int x, int y) {
	switch (key) {

	case GLUT_KEY_UP:
	case GLUT_KEY_DOWN:
		transZ_direction = 0;
		break;

	case GLUT_KEY_PAGE_UP:
	case GLUT_KEY_PAGE_DOWN:
		rotX_direction = 0;
		break;
	}
}

void Renderer::onKeyboardUp(unsigned char key, int x, int y) {
	switch (key) {
	case 'A':
	case 'a':
	case 'D':
	case 'd':
		rotY_direction = 0;
		break;

	case 'W':
	case 'w':
	case 'S':
	case 's':
		transZ_direction = 0;
		break;
	}
}

void Renderer::onDisplay() {
	logic();
	draw();
	glutSwapBuffers();
}

void Renderer::onMouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		arcball_on = true;
		last_mx = cur_mx = x;
		last_my = cur_my = y;
	}
	else {
		arcball_on = false;
	}
}

void Renderer::onMotion(int x, int y) {
	if (arcball_on) {  // if left button is pressed
		cur_mx = x;
		cur_my = y;
	}
}

void Renderer::onReshape(int width, int height) {
	screen_width = width;
	screen_height = height;
	glViewport(0, 0, screen_width, screen_height);
}
int Renderer::init_resources(string ScanMeshPathAndPrefix, string tex_filename, string bmp_filename, const char* vshader_filename, const char* fshader_filename, int startIdx, int endIdx) {
	
	glm::vec3 light_position = glm::vec3(0.0, 1.0, 2.0);
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, 0.1, 0.0));
	light_bbox.object2world = glm::translate(glm::mat4(1), light_position);

	/* Compile and link shaders */
	GLint link_ok = GL_FALSE;
	GLint validate_ok = GL_FALSE;

	GLuint vs, fs;

	if ((vs = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return 0;
	}

	if ((fs = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return 0;
	}

	program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);

	if (!link_ok) {
		fprintf(stderr, "glLinkProgram:");
		print_log(program);
		return 0;
	}

	glValidateProgram(program);
	glGetProgramiv(program, GL_VALIDATE_STATUS, &validate_ok);

	if (!validate_ok) {
		fprintf(stderr, "glValidateProgram:");
		print_log(program);
	}

	objmov = main_object4D.upload_UVAtlas(ScanMeshPathAndPrefix, bmp_filename, tex_filename, startIdx, endIdx);

	const char* attribute_name;
	attribute_name = "v_coord";
	attribute_v_coord = glGetAttribLocation(program, attribute_name);

	if (attribute_v_coord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_normal";
	attribute_v_normal = glGetAttribLocation(program, attribute_name);

	if (attribute_v_normal == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_tex_coord";
	attribute_v_texcoord = glGetAttribLocation(program, attribute_name);
	if (attribute_v_texcoord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	const char* uniform_name;
	uniform_name = "m";
	uniform_m = glGetUniformLocation(program, uniform_name);

	if (uniform_m == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v";
	uniform_v = glGetUniformLocation(program, uniform_name);

	if (uniform_v == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "p";
	uniform_p = glGetUniformLocation(program, uniform_name);

	if (uniform_p == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "m_3x3_inv_transp";
	uniform_m_3x3_inv_transp = glGetUniformLocation(program, uniform_name);

	if (uniform_m_3x3_inv_transp == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v_inv";
	uniform_v_inv = glGetUniformLocation(program, uniform_name);

	if (uniform_v_inv == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "flags";
	uniform_flags = glGetUniformLocation(program, uniform_name);

	if (uniform_flags == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	fps_start = glutGet(GLUT_ELAPSED_TIME);

	return 1;

}
int Renderer::init_resources(string ScanMeshPathAndPrefix, string bmp_filename, char* vshader_filename, char* fshader_filename, int startIdx, int endIdx) {
	//load_obj(model_filename, &main_object);
	// mesh position initialized in init_view()

	glm::vec3 light_position = glm::vec3(0.0, 1.0, 2.0);
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, 0.1, 0.0));
	light_bbox.object2world = glm::translate(glm::mat4(1), light_position);

	main_object4D.upload(ScanMeshPathAndPrefix, bmp_filename, startIdx, endIdx);
	//main_object.upload(bmp_filename);
	//light_bbox.upload();
	/* Compile and link shaders */
	GLint link_ok = GL_FALSE;
	GLint validate_ok = GL_FALSE;

	GLuint vs, fs;

	if ((vs = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return 0;
	}

	if ((fs = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return 0;
	}

	program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);

	if (!link_ok) {
		fprintf(stderr, "glLinkProgram:");
		print_log(program);
		return 0;
	}

	glValidateProgram(program);
	glGetProgramiv(program, GL_VALIDATE_STATUS, &validate_ok);

	if (!validate_ok) {
		fprintf(stderr, "glValidateProgram:");
		print_log(program);
	}

	const char* attribute_name;
	attribute_name = "v_coord";
	attribute_v_coord = glGetAttribLocation(program, attribute_name);

	if (attribute_v_coord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_normal";
	attribute_v_normal = glGetAttribLocation(program, attribute_name);

	if (attribute_v_normal == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_tex_coord";
	attribute_v_texcoord = glGetAttribLocation(program, attribute_name);
	if (attribute_v_texcoord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	const char* uniform_name;
	uniform_name = "m";
	uniform_m = glGetUniformLocation(program, uniform_name);

	if (uniform_m == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v";
	uniform_v = glGetUniformLocation(program, uniform_name);

	if (uniform_v == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "p";
	uniform_p = glGetUniformLocation(program, uniform_name);

	if (uniform_p == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "m_3x3_inv_transp";
	uniform_m_3x3_inv_transp = glGetUniformLocation(program, uniform_name);

	if (uniform_m_3x3_inv_transp == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v_inv";
	uniform_v_inv = glGetUniformLocation(program, uniform_name);

	if (uniform_v_inv == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	fps_start = glutGet(GLUT_ELAPSED_TIME);

	return 1;

}

int Renderer::init_resources(char* bmp_filename, char* vshader_filename, char* fshader_filename) {
	//load_obj(model_filename, &main_object);
	// mesh position initialized in init_view()

	glm::vec3 light_position = glm::vec3(0.0, 1.0, 2.0);
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, 0.1, 0.0));
	light_bbox.object2world = glm::translate(glm::mat4(1), light_position);

	main_object.upload(bmp_filename);
	light_bbox.upload();
	/* Compile and link shaders */
	GLint link_ok = GL_FALSE;
	GLint validate_ok = GL_FALSE;

	GLuint vs, fs;

	if ((vs = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return 0;
	}

	if ((fs = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return 0;
	}

	program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);

	if (!link_ok) {
		fprintf(stderr, "glLinkProgram:");
		print_log(program);
		return 0;
	}

	glValidateProgram(program);
	glGetProgramiv(program, GL_VALIDATE_STATUS, &validate_ok);

	if (!validate_ok) {
		fprintf(stderr, "glValidateProgram:");
		print_log(program);
	}

	const char* attribute_name;
	attribute_name = "v_coord";
	attribute_v_coord = glGetAttribLocation(program, attribute_name);

	if (attribute_v_coord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_normal";
	attribute_v_normal = glGetAttribLocation(program, attribute_name);

	if (attribute_v_normal == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_tex_coord";
	attribute_v_texcoord = glGetAttribLocation(program, attribute_name);
	if (attribute_v_texcoord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_img_tex_coord";
	attribute_v_img_texcoord = glGetAttribLocation(program, attribute_name);
	if (attribute_v_img_texcoord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}
	
	const char* uniform_name;
	uniform_name = "m";
	uniform_m = glGetUniformLocation(program, uniform_name);

	if (uniform_m == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v";
	uniform_v = glGetUniformLocation(program, uniform_name);

	if (uniform_v == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "p";
	uniform_p = glGetUniformLocation(program, uniform_name);

	if (uniform_p == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "m_3x3_inv_transp";
	uniform_m_3x3_inv_transp = glGetUniformLocation(program, uniform_name);

	if (uniform_m_3x3_inv_transp == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v_inv";
	uniform_v_inv = glGetUniformLocation(program, uniform_name);

	if (uniform_v_inv == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	fps_start = glutGet(GLUT_ELAPSED_TIME);

	return 1;

}
int Renderer::init_resources(char* vshader_filename, char* fshader_filename) {
	//load_obj(model_filename, &main_object);
	// mesh position initialized in init_view()

	glm::vec3 light_position = glm::vec3(0.0, 1.0, 2.0);
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, 0.1, 0.0));
	light_bbox.object2world = glm::translate(glm::mat4(1), light_position);

	//main_object.upload(optimizer->GetAtlas(), ATLAS_SIDE);
	char* v_path = "D:/texture_CGI16/TexGen_non_rigid_0522/src/atlas.v.glsl";
	char* f_path = "D:/texture_CGI16/TexGen_non_rigid_0522/src/atlas.f.glsl";
	char* v_path2 = "D:/texture_CGI16/TexGen_non_rigid_0522/src/atlas2.v.glsl";
	char* f_path2 = "D:/texture_CGI16/TexGen_non_rigid_0522/src/atlas2.f.glsl";
	uchar *data;
	data = new uchar[3 * ATLAS_SIDE * ATLAS_SIDE];
	
	//GenAtlas2(v_path2, f_path2, data, true);
	//GenAtlas(v_path, f_path, data, true);
	//GenAtlas_UVAtlas(v_path, f_path, "D:/3D_data/L0/190529_gunhee_tex.obj",data, true);

	
	main_object.upload(data, ATLAS_SIDE);
	free(data);
	light_bbox.upload();
	/* Compile and link shaders */
	GLint link_ok = GL_FALSE;
	GLint validate_ok = GL_FALSE;

	GLuint vs, fs;

	if ((vs = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return 0;
	}

	if ((fs = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return 0;
	}

	program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);

	if (!link_ok) {
		fprintf(stderr, "glLinkProgram:");
		print_log(program);
		return 0;
	}

	glValidateProgram(program);
	glGetProgramiv(program, GL_VALIDATE_STATUS, &validate_ok);

	if (!validate_ok) {
		fprintf(stderr, "glValidateProgram:");
		print_log(program);
	}

	const char* attribute_name;
	attribute_name = "v_coord";
	attribute_v_coord = glGetAttribLocation(program, attribute_name);

	if (attribute_v_coord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_normal";
	attribute_v_normal = glGetAttribLocation(program, attribute_name);

	if (attribute_v_normal == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_tex_coord";
	attribute_v_texcoord = glGetAttribLocation(program, attribute_name);

	if (attribute_v_texcoord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_img_coord";
	attribute_v_img_texcoord = glGetAttribLocation(program, attribute_name);

	if (attribute_v_img_texcoord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	const char* uniform_name;
	uniform_name = "m";
	uniform_m = glGetUniformLocation(program, uniform_name);

	if (uniform_m == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v";
	uniform_v = glGetUniformLocation(program, uniform_name);

	if (uniform_v == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "p";
	uniform_p = glGetUniformLocation(program, uniform_name);

	if (uniform_p == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "m_3x3_inv_transp";
	uniform_m_3x3_inv_transp = glGetUniformLocation(program, uniform_name);

	if (uniform_m_3x3_inv_transp == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v_inv";
	uniform_v_inv = glGetUniformLocation(program, uniform_name);

	if (uniform_v_inv == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "flags";
	uniform_flags = glGetUniformLocation(program, uniform_name);

	if (uniform_flags == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}


	fps_start = glutGet(GLUT_ELAPSED_TIME);

	return 1;

}
int Renderer::init_resources_UVAtlas(const char* vshader_filename, const char* fshader_filename, string tex_filename) {
	
	string v_path_atlas = "./shaders/" + shader_atlas + ".v.glsl";
	string f_path_atlas = "./shaders/" + shader_atlas + ".f.glsl";
	string v_path_sub_atlas = "./shaders/" + shader_sub_atlas + ".v.glsl";
	string f_path_sub_atlas = "./shaders/" + shader_sub_atlas + ".f.glsl";
	string v_path_mask = "./shaders/" + shader_mask + ".v.glsl";
	string f_path_mask = "./shaders/" + shader_mask + ".f.glsl";
	uchar *data;
	data = new uchar[3 * ATLAS_SIDE * ATLAS_SIDE];

	GenAtlas_UVAtlas(v_path_atlas.c_str(), f_path_atlas.c_str(), tex_filename, data, true);
	if (RECORD_UNIT) {
		GenAtlas_UVAtlas_texel(v_path_sub_atlas.c_str(), f_path_sub_atlas.c_str(), tex_filename, data, true);
		GenAtlas_UVAtlas_mask(v_path_mask.c_str(), f_path_mask.c_str(), tex_filename, data, true);
	}
	
	return 1;

}

int Renderer::init_resources(uchar* data, char* vshader_filename, char* fshader_filename) {
	//load_obj(model_filename, &main_object);
	// mesh position initialized in init_view()

	glm::vec3 light_position = glm::vec3(0.0, 1.0, 2.0);
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, -0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, -0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(0.1, 0.1, 0.1, 0.0));
	light_bbox.vertices.push_back(glm::vec4(-0.1, 0.1, 0.1, 0.0));
	light_bbox.object2world = glm::translate(glm::mat4(1), light_position);

	main_object.upload(data, ATLAS_SIDE);
	light_bbox.upload();
	/* Compile and link shaders */
	GLint link_ok = GL_FALSE;
	GLint validate_ok = GL_FALSE;

	GLuint vs, fs;

	if ((vs = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return 0;
	}

	if ((fs = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return 0;
	}

	program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);

	if (!link_ok) {
		fprintf(stderr, "glLinkProgram:");
		print_log(program);
		return 0;
	}

	glValidateProgram(program);
	glGetProgramiv(program, GL_VALIDATE_STATUS, &validate_ok);

	if (!validate_ok) {
		fprintf(stderr, "glValidateProgram:");
		print_log(program);
	}

	const char* attribute_name;
	attribute_name = "v_coord";
	attribute_v_coord = glGetAttribLocation(program, attribute_name);

	if (attribute_v_coord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_normal";
	attribute_v_normal = glGetAttribLocation(program, attribute_name);

	if (attribute_v_normal == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	attribute_name = "v_tex_coord";
	attribute_v_texcoord = glGetAttribLocation(program, attribute_name);
	if (attribute_v_texcoord == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
		return 0;
	}

	const char* uniform_name;
	uniform_name = "m";
	uniform_m = glGetUniformLocation(program, uniform_name);

	if (uniform_m == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v";
	uniform_v = glGetUniformLocation(program, uniform_name);

	if (uniform_v == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "p";
	uniform_p = glGetUniformLocation(program, uniform_name);

	if (uniform_p == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "m_3x3_inv_transp";
	uniform_m_3x3_inv_transp = glGetUniformLocation(program, uniform_name);

	if (uniform_m_3x3_inv_transp == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	uniform_name = "v_inv";
	uniform_v_inv = glGetUniformLocation(program, uniform_name);

	if (uniform_v_inv == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		return 0;
	}

	fps_start = glutGet(GLUT_ELAPSED_TIME);

	return 1;

}

void Renderer::load_obj(string filename) {
	Mesh* mesh;
	mesh = &main_object;

	ifstream in(filename, ios::in);

	if (!in) {
		cerr << "Cannot open " << filename << endl;
		exit(1);
	}

	vector<int> nb_seen;


	string line;

	while (getline(in, line)) {
		if (line.substr(0, 2) == "v ") {
			istringstream s(line.substr(2));
			glm::vec4 v;
			s >> v.x;
			s >> v.y;
			s >> v.z;
			/*v.x /= 1000;
			v.y /= 1000;
			v.z /= 1000;*/
			v.w = 1.0;
			mesh->vertices.push_back(v);
		}
		else if (line.substr(0, 3) == "vt ") {
			istringstream s(line.substr(3));
			glm::vec2 vt;
			s >> vt.x;
			s >> vt.y;
			mesh->tex_coords.push_back(vt);
		}
		else if (line.substr(0, 2) == "f ") {
			GLuint a, b, c, at, bt, ct;
			if (line.find('/')) {
				replace(line.begin(), line.end(), '/', ' ');
				istringstream s(line.substr(2));
				s >> a;
				s >> at;
				s >> b;
				s >> bt;
				s >> c;
				s >> ct;
			}
			else {
				istringstream s(line.substr(2));
				s >> a;
				s >> b;
				s >> c;
			}
			a--;
			b--;
			c--;
			mesh->elements.push_back(a);
			mesh->elements.push_back(b);
			mesh->elements.push_back(c);
		}
		else if (line[0] == '#') {
			/* ignoring this line */
		}
		else {
			/* ignoring this line */
		}
	}

	mesh->normals.resize(mesh->vertices.size(), glm::vec3(0.0, 0.0, 0.0));
	nb_seen.resize(mesh->vertices.size(), 0);

	glm::vec3 sum = glm::vec3(.0, .0, .0);

	for (unsigned int i = 0; i < mesh->elements.size(); i += 3) {
		GLuint ia = mesh->elements[i];
		GLuint ib = mesh->elements[i + 1];
		GLuint ic = mesh->elements[i + 2];
		glm::vec3 normal = glm::normalize(glm::cross(
			glm::vec3(mesh->vertices[ib]) - glm::vec3(mesh->vertices[ia]),
			glm::vec3(mesh->vertices[ic]) - glm::vec3(mesh->vertices[ia])));

		int v[3];
		v[0] = ia;
		v[1] = ib;
		v[2] = ic;

		for (int j = 0; j < 3; j++) {
			GLuint cur_v = v[j];
			nb_seen[cur_v]++;

			if (nb_seen[cur_v] == 1) {
				mesh->normals[cur_v] = normal;
			}
			else {
				// average
				mesh->normals[cur_v].x = mesh->normals[cur_v].x * (1.0 - 1.0 / nb_seen[cur_v]) + normal.x * 1.0 / nb_seen[cur_v];
				mesh->normals[cur_v].y = mesh->normals[cur_v].y * (1.0 - 1.0 / nb_seen[cur_v]) + normal.y * 1.0 / nb_seen[cur_v];
				mesh->normals[cur_v].z = mesh->normals[cur_v].z * (1.0 - 1.0 / nb_seen[cur_v]) + normal.z * 1.0 / nb_seen[cur_v];
				mesh->normals[cur_v] = glm::normalize(mesh->normals[cur_v]);
			}
		}
	}
	for (unsigned int i = 0; i < mesh->vertices.size(); i++) {
		sum.x += glm::vec3(mesh->vertices[i]).x;
		sum.y += glm::vec3(mesh->vertices[i]).y;
		sum.z += glm::vec3(mesh->vertices[i]).z;

	}
	sum /= mesh->vertices.size();

	for (unsigned int i = 0; i < mesh->vertices.size(); i++) {
		mesh->vertices[i] -= glm::vec4(sum, 0.0);
		//mesh->vertices[i] += glm::vec4(0.0,2.0,0.0, 0.0);
	}

}
void Renderer::load_obj_tex(string filename) {
	Mesh* mesh;
	mesh = &main_object_tex;

	ifstream in(filename, ios::in);

	if (!in) {
		cerr << "Cannot open " << filename << endl;
		exit(1);
	}

	vector<int> nb_seen;


	string line;

	while (getline(in, line)) {
		if (line.substr(0, 2) == "v ") {
			istringstream s(line.substr(2));
			glm::vec4 v;
			s >> v.x;
			s >> v.y;
			s >> v.z;
			/*v.x /= 1000;
			v.y /= 1000;
			v.z /= 1000;*/
			v.w = 1.0;
			mesh->vertices.push_back(v);
		}
		else if (line.substr(0, 3) == "vt ") {
			istringstream s(line.substr(3));
			glm::vec2 vt;
			s >> vt.x;
			s >> vt.y;
			mesh->tex_coords.push_back(vt);
		}
		else if (line.substr(0, 2) == "f ") {
			GLuint a, b, c, at, bt, ct;
			if (line.find('/')) {
				replace(line.begin(), line.end(), '/', ' ');
				istringstream s(line.substr(2));
				s >> a;
				s >> at;
				s >> b;
				s >> bt;
				s >> c;
				s >> ct;
			}
			else {
				istringstream s(line.substr(2));
				s >> a;
				s >> b;
				s >> c;
			}
			a--;
			b--;
			c--;
			at--;
			bt--;
			ct--;
			mesh->elements.push_back(a);
			mesh->elements.push_back(b);
			mesh->elements.push_back(c);
			mesh->tex_elements.push_back(at);
			mesh->tex_elements.push_back(bt);
			mesh->tex_elements.push_back(ct);
		}
		else if (line[0] == '#') {
			/* ignoring this line */
		}
		else {
			/* ignoring this line */
		}
	}

	mesh->normals.resize(mesh->vertices.size(), glm::vec3(0.0, 0.0, 0.0));
	nb_seen.resize(mesh->vertices.size(), 0);

	glm::vec3 sum = glm::vec3(.0, .0, .0);

	for (unsigned int i = 0; i < mesh->elements.size(); i += 3) {
		GLuint ia = mesh->elements[i];
		GLuint ib = mesh->elements[i + 1];
		GLuint ic = mesh->elements[i + 2];
		glm::vec3 normal = glm::normalize(glm::cross(
			glm::vec3(mesh->vertices[ib]) - glm::vec3(mesh->vertices[ia]),
			glm::vec3(mesh->vertices[ic]) - glm::vec3(mesh->vertices[ia])));

		int v[3];
		v[0] = ia;
		v[1] = ib;
		v[2] = ic;

		for (int j = 0; j < 3; j++) {
			GLuint cur_v = v[j];
			nb_seen[cur_v]++;

			if (nb_seen[cur_v] == 1) {
				mesh->normals[cur_v] = normal;
			}
			else {
				// average
				mesh->normals[cur_v].x = mesh->normals[cur_v].x * (1.0 - 1.0 / nb_seen[cur_v]) + normal.x * 1.0 / nb_seen[cur_v];
				mesh->normals[cur_v].y = mesh->normals[cur_v].y * (1.0 - 1.0 / nb_seen[cur_v]) + normal.y * 1.0 / nb_seen[cur_v];
				mesh->normals[cur_v].z = mesh->normals[cur_v].z * (1.0 - 1.0 / nb_seen[cur_v]) + normal.z * 1.0 / nb_seen[cur_v];
				mesh->normals[cur_v] = glm::normalize(mesh->normals[cur_v]);
			}
		}
	}
	for (unsigned int i = 0; i < mesh->vertices.size(); i++) {
		sum.x += glm::vec3(mesh->vertices[i]).x;
		sum.y += glm::vec3(mesh->vertices[i]).y;
		sum.z += glm::vec3(mesh->vertices[i]).z;

	}
	sum /= mesh->vertices.size();

	for (unsigned int i = 0; i < mesh->vertices.size(); i++) {
		mesh->vertices[i] -= glm::vec4(sum, 0.0);
		//mesh->vertices[i] += glm::vec4(0.0,2.0,0.0, 0.0);
	}

}

void Renderer::init_view() {
	main_object.object2world = glm::mat4(1);
	main_object4D.object2world = glm::mat4(1);
	//transforms[MODE_CAMERA] = glm::lookAt(
	//	glm::vec3(0.0, 0.0, 4.0),   // eye
	//	glm::vec3(0.0, 0.0, 0.0),   // direction
	//	glm::vec3(0.0, 1.0, 0.0));  // up
	transforms[MODE_CAMERA] = glm::lookAt(
		glm::vec3(0.0, 0.0, 0.0) - objmov,   // eye
		glm::vec3(0.0, 0.0, 1.0),   // direction
		glm::vec3(0.0, -1.0, 0.0));  // up
}
void Renderer::draw() {
	glClearColor(1.0,1.0,1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(program);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glEnable(GL_TEXTURE_2D);
	main_object4D.draw();
	//main_object.draw();
	//ground.draw();
	//light_bbox.draw_bbox();
	glDisable(GL_TEXTURE_2D);
	//main_object.drawwireframe();
}
void Renderer::logic() {
	/* FPS count */
	{
		fps_frames++;
		int delta_t = glutGet(GLUT_ELAPSED_TIME) - fps_start;

		if (delta_t > 1000) {
			fps_now = 1000.0 * fps_frames / delta_t;
			//cout << 1000.0 * fps_frames / delta_t << " fps" << endl;
			fps_frames = 0;
			fps_start = glutGet(GLUT_ELAPSED_TIME);
		}
	}
	if (object4D_play && fps_frames % ((int)(fps_now / fps_play) + 1) == 0) {
		string flag;
		string filename;
		if (main_object4D._flags[1]) {
			if(main_object4D._flags[0])
				flag = "sl_color_render";
			else
				flag = "ml_color_render";
		}
		else {
			if (main_object4D._flags[0])
				flag = "naive_color_render";
			else
				flag = "geo_render";
		}
		if (RECORD_UNIT) {
			filename = data_root_path + "/unit_test/" + unit_test_path + "/" + flag + "/Frame" + zeroPadding(main_object4D.get_time_stamp(), 3) + ".png";
			ScreenCapture(filename.c_str());
		}
		main_object4D.up_time_stamp();
	}

	/* Handle keyboard-based transformations */
	int delta_t = glutGet(GLUT_ELAPSED_TIME) - last_ticks;
	last_ticks = glutGet(GLUT_ELAPSED_TIME);

	float delta_transZ = transZ_direction * delta_t / 1000.0 * 5 * speed_factor;  // 5 units per second
	float delta_transX = 0, delta_transY = 0, delta_rotY = 0, delta_rotX = 0;

	if (strife) {
		delta_transX = rotY_direction * delta_t / 1000.0 * 3 * speed_factor;  // 3 units per second
		delta_transY = rotX_direction * delta_t / 1000.0 * 3 * speed_factor;  // 3 units per second
	}
	else {
		delta_rotY = rotY_direction * delta_t / 1000.0 * 120 * speed_factor;  // 120° per second
		delta_rotX = -rotX_direction * delta_t / 1000.0 * 120 * speed_factor;  // 120° per second
	}

	if (view_mode == MODE_OBJECT) {
		main_object4D.object2world = glm::rotate(main_object4D.object2world, glm::radians(delta_rotY), glm::vec3(0.0, 1.0, 0.0));
		main_object4D.object2world = glm::rotate(main_object4D.object2world, glm::radians(delta_rotX), glm::vec3(1.0, 0.0, 0.0));
		main_object4D.object2world = glm::translate(main_object4D.object2world, glm::vec3(0.0, 0.0, delta_transZ));
		main_object.object2world = glm::rotate(main_object.object2world, glm::radians(delta_rotY), glm::vec3(0.0, 1.0, 0.0));
		main_object.object2world = glm::rotate(main_object.object2world, glm::radians(delta_rotX), glm::vec3(1.0, 0.0, 0.0));
		main_object.object2world = glm::translate(main_object.object2world, glm::vec3(0.0, 0.0, delta_transZ));
	}
	else if (view_mode == MODE_CAMERA) {
		// Camera is reverse-facing, so reverse Z translation and X rotation.
		// Plus, the View matrix is the inverse of the camera2world (it's
		// world->camera), so we'll reverse the transformations.
		// Alternatively, imagine that you transform the world, instead of positioning the camera.
		if (strife) {
			transforms[MODE_CAMERA] = glm::translate(glm::mat4(1.0), glm::vec3(delta_transX, 0.0, 0.0)) * transforms[MODE_CAMERA];
		}
		else {
			glm::vec3 y_axis_world = glm::mat3(transforms[MODE_CAMERA]) * glm::vec3(0.0, 1.0, 0.0);
			transforms[MODE_CAMERA] = glm::rotate(glm::mat4(1.0), glm::radians(-delta_rotY), y_axis_world) * transforms[MODE_CAMERA];
		}

		if (strife) {
			transforms[MODE_CAMERA] = glm::translate(glm::mat4(1.0), glm::vec3(0.0, delta_transY, 0.0)) * transforms[MODE_CAMERA];
		}
		else {
			transforms[MODE_CAMERA] = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, delta_transZ)) * transforms[MODE_CAMERA];
		}

		transforms[MODE_CAMERA] = glm::rotate(glm::mat4(1.0), glm::radians(delta_rotX), glm::vec3(1.0, 0.0, 0.0)) * transforms[MODE_CAMERA];
	}

	/* Handle arcball */
	if (view_mode == MODE_CAMERA) {
		if (cur_mx != last_mx || cur_my != last_my) {
			glm::vec3 va = get_arcball_vector(last_mx, last_my);
			glm::vec3 vb = get_arcball_vector(cur_mx, cur_my);
			int dx = cur_mx - last_mx;
			int dy = cur_my - last_my;
			float angle = acos(min(1.0f, glm::dot(va, vb)));
			glm::vec3 axis_in_camera_coord = glm::mat3(transforms[MODE_CAMERA]) * glm::vec3(dy, -dx, 0.0);
			transforms[MODE_CAMERA] = glm::rotate(glm::mat4(1.0), -angle, axis_in_camera_coord)*transforms[MODE_CAMERA];
			last_mx = cur_mx;
			last_my = cur_my;
		}
	}
	else if (view_mode == MODE_OBJECT) {
		if (cur_mx != last_mx || cur_my != last_my) {
			glm::vec3 va = get_arcball_vector(last_mx, last_my);
			glm::vec3 vb = get_arcball_vector(cur_mx, cur_my);
			float angle = acos(min(1.0f, glm::dot(va, vb)));
			glm::vec3 axis_in_camera_coord = glm::cross(va, vb);
			glm::mat3 camera2object4D = glm::inverse(glm::mat3(transforms[MODE_CAMERA]) * glm::mat3(main_object4D.object2world));
			glm::vec3 axis_in_object_coord4D = camera2object4D * axis_in_camera_coord;
			glm::mat3 camera2object = glm::inverse(glm::mat3(transforms[MODE_CAMERA]) * glm::mat3(main_object.object2world));
			glm::vec3 axis_in_object_coord = camera2object * axis_in_camera_coord;
			main_object4D.object2world = glm::rotate(main_object4D.object2world, angle, axis_in_object_coord4D);
			main_object.object2world = glm::rotate(main_object.object2world, angle, axis_in_object_coord);
			last_mx = cur_mx;
			last_my = cur_my;
		}
	}


	// Model
	// Set in onDisplay() - cf. main_object.object2world

	// View
	glm::mat4 world2camera = transforms[MODE_CAMERA];

	// Projection
	glm::mat4 camera2screen = glm::perspective(45.0f, 1.0f * screen_width / screen_height, 0.1f, 100.0f);

	glUseProgram(program);
	glUniformMatrix4fv(uniform_v, 1, GL_FALSE, glm::value_ptr(world2camera));
	glUniformMatrix4fv(uniform_p, 1, GL_FALSE, glm::value_ptr(camera2screen));

	glm::mat4 v_inv = glm::inverse(world2camera);
	glUniformMatrix4fv(uniform_v_inv, 1, GL_FALSE, glm::value_ptr(v_inv));

	glutPostRedisplay();
}
void Renderer::free_resources() {
	glDeleteProgram(program);
}

Renderer* currentInstance;

void Wrap_C_onSpecial(int key, int x, int y) { currentInstance->onSpecial(key, x, y); }
void Wrap_C_onKeyboard(unsigned char key, int x, int y) { currentInstance->onKeyboard(key, x, y); }
void Wrap_C_onSpecialUp(int key, int x, int y) { currentInstance->onSpecialUp(key, x, y); }
void Wrap_C_onKeyboardUp(unsigned char key, int x, int y) { currentInstance->onKeyboardUp(key, x, y); }
void Wrap_C_onDisplay() { currentInstance->onDisplay(); }
void Wrap_C_onMouse(int button, int state, int x, int y) { currentInstance->onMouse(button, state, x, y); }
void Wrap_C_onMotion(int x, int y) { currentInstance->onMotion(x, y); }
void Wrap_C_onReshape(int width, int height) { currentInstance->onReshape(width, height); }

void Renderer::mainloop() {
	currentInstance = this;
	glutDisplayFunc(Wrap_C_onDisplay);
	glutSpecialFunc(Wrap_C_onSpecial);
	glutSpecialUpFunc(Wrap_C_onSpecialUp);
	glutKeyboardFunc(Wrap_C_onKeyboard);
	glutKeyboardUpFunc(Wrap_C_onKeyboardUp);
	glutMouseFunc(Wrap_C_onMouse);
	glutMotionFunc(Wrap_C_onMotion);
	glutReshapeFunc(Wrap_C_onReshape);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LEQUAL);
	//glDepthRange(1, 0);
	last_ticks = glutGet(GLUT_ELAPSED_TIME);
	//----------------------------------
	//----------------------------------
	glutMainLoop();
}

void atlas_onDisplay() {
	glutSwapBuffers();
}

void Renderer::GenAtlas_UVAtlas(const char* vshader_filename, const char* fshader_filename, string tex_filename, uchar* result, bool draw, int width, int height) {

	load_obj_tex(tex_filename);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);



	GLuint atlas_prog, atlas_shader_v, atlas_shader_f;
	GLuint atlas_texID, img_texID;

	atlas_prog = glCreateProgram();

	if ((atlas_shader_v = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return;
	}

	if ((atlas_shader_f = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return;
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_POLYGON_SMOOTH);

	glAttachShader(atlas_prog, atlas_shader_v);
	glAttachShader(atlas_prog, atlas_shader_f);
	glLinkProgram(atlas_prog);

	glGenFramebuffers(1, &fb);
	glGenRenderbuffers(1, &rb);
	glBindRenderbuffer(GL_RENDERBUFFER, rb);
	glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);
	glFramebufferRenderbuffer(
		GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);


	//glClearColor(0.5f, 0.5f, 1.0f, 1.0f);
	glClearColor(0.f, 0.f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glViewport(0, 0, width, height);
	glUseProgram(atlas_prog);

	atlas_texID = glGetUniformLocation(atlas_prog, "bufImg");
	img_texID = glGetUniformLocation(atlas_prog, "tempImg");

	optimizer->PrepareRend_UVAtlas();

	uint nImage, width_img, height_img;
	optimizer->GetNumber(&nImage, &width_img, &height_img);

	glm::vec2 uniform_imgsize = glm::vec2(width_img, height_img);
	glUniform2fv(glGetUniformLocation(atlas_prog, "img_size"), 1, glm::value_ptr(uniform_imgsize));

	GLuint texture;
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE1);
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);*/
	//glBindTexture(GL_TEXTURE_2D, texture);
	/*glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);*/


	GLuint tex;

	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);*/
	//glBindTexture(GL_TEXTURE_2D, tex);
	//glBindTexture(GL_TEXTURE_2D, 0);

	GLuint uvBuf, uvImgBuf;
	glGenBuffers(1, &uvBuf);
	glGenBuffers(1, &uvImgBuf);
	GLuint tbo;
	glGenBuffers(1, &tbo);
	/*glGenBuffers(1, &uvBuf);
	glGenBuffers(1, &uvImgBuf);
	vector<float> atlas_coord;
	vector<float> img_coord;

	optimizer->GetAtlasInfoi(&atlas_coord, &img_coord, 0);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), img_coord.data(), GL_STATIC_DRAW);*/

	//system("pause");
	//Sleep(10);

	//uchar* tmpImg = new uchar[width_img*height_img * 3 * 8];
	/*for (int i = 0; i < 8; i++) {
	cv::Mat colortemp = cv::imread("D:\\3D data\\multiview_opt\\stream\\0\\Color_"+to_string(i)+".png");
	cv::cvtColor(colortemp, colortemp, CV_BGRA2RGB);
	memcpy(&tmpImg[width_img*height_img * 3 * i], colortemp.data, sizeof(uchar)*width_img*height_img * 3);
	}*/

	//memcpy(tmpImg, optimizer->hCImages, sizeof(uchar)*width_img*height_img * 3 * 8);

	float aaa = 0.0;
	vector<float> atlas_coord;
	vector<float> img_coord;
	vector<int> tri_idx;

	//GLuint tbo[8];
	//glGenBuffers(8, tbo);
	//for (uint i = 0; i < 8; i++) {
	//	glBindBuffer(GL_TEXTURE_BUFFER, tbo[i]);
	//	glBufferData(GL_TEXTURE_BUFFER, sizeof(float)*width_img*height_img * 3, &optimizer->hCImages[i*width_img*height_img * 3], GL_STATIC_DRAW);

	//	//glGenTextures(1, &tbo_tex);

	//	glBindBuffer(GL_TEXTURE_BUFFER, 0);
	//
	//}

	for (uint i = 0; i < nImage; i++) {

		StopWatchInterface *get_timer = NULL;
		sdkCreateTimer(&get_timer);
		sdkStartTimer(&get_timer);

		/*float *atlas_coord = new float;
		float *img_coord = new float;*/

		optimizer->GetAtlasInfoi_UVAtlas(&atlas_coord, &img_coord, &tri_idx, i);
		for (int ti = 0; ti < tri_idx.size(); ti++) {
			atlas_coord[ti * 6] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3]].x;
			atlas_coord[ti * 6 + 1] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3]].y;
			atlas_coord[ti * 6 + 2] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 1]].x;
			atlas_coord[ti * 6 + 3] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 1]].y;
			atlas_coord[ti * 6 + 4] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 2]].x;
			atlas_coord[ti * 6 + 5] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 2]].y;
		}
		/*uchar* tmpImg = new uchar[width_img*height_img * 3];
		memcpy(tmpImg, &optimizer->hCImages[i*width_img*height_img * 3], sizeof(uchar)*width_img*height_img * 3);*/

		//cv::Mat3b ch4Image;
		//ch4Image.create(height_img, width_img);
		//memcpy(ch4Image.data, tmpImg, sizeof(uchar)*width_img*height_img * 3);
		////cv::resize(ch4Image, ch4Image, cv::Size(800, 800));
		//for (int kkk = 0; kkk < atlas_coord.size() / 2; kkk += 9) {
		//	cv::line(ch4Image, cv::Point(img_coord[kkk], img_coord[kkk + 1]), cv::Point(img_coord[kkk + 3], img_coord[kkk + 4]), cv::Scalar(255, 0, 0));
		//}
		//cv::imshow("111", ch4Image);
		//cv::waitKey(0);
		//GLuint uvBuf, uvImgBuf;
		glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * atlas_coord.size(), atlas_coord.data());
		//glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), atlas_coord.data(), GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * img_coord.size(), img_coord.data());
		//glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), img_coord.data(), GL_STREAM_DRAW);
		//cout << img_coord.size() << "  ,  " << atlas_coord.size() << endl;

		/*uchar* tmp_buf = new uchar[8000 * 8000 * 3];
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height,
		GL_RGB, GL_UNSIGNED_BYTE, tmp_buf);*/
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, tmp_buf);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glUniform1i(atlas_texID, 1);


		///////
		//free(tmp_buf);

		/*GLuint tex;

		glGenTextures(1, &tex);*/
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		/*glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGB, width_img, height_img,
		0, GL_RGB, GL_UNSIGNED_BYTE, &tmpImg[i*width_img*height_img * 3]);*/

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_img, height_img, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		/*glBindBuffer(GL_TEXTURE_BUFFER, tbo);
		glBufferData(GL_TEXTURE_BUFFER, sizeof(float)*width_img*height_img * 3, NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) *width_img*height_img * 3, &optimizer->hCImages[i*width_img*height_img * 3]);
		glBindTexture(GL_TEXTURE_BUFFER, tex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB, tbo);*/
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_img, height_img, GL_RGB, GL_UNSIGNED_BYTE, &optimizer->hCImages[i*width_img*height_img * 3]);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_img, height_img, GL_RGB, GL_UNSIGNED_BYTE, &tmpImg[i*width_img*height_img * 3]);

		sdkStopTimer(&get_timer);
		float get_time = sdkGetAverageTimerValue(&get_timer) / 1000.0f;
		aaa += get_time * 1000;
		printf("%d - time!!!!!!!! : %fms\n", i, get_time * 1000);
		glUniform1i(img_texID, 0);


		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		//glClear(GL_DEPTH_BUFFER_BIT);
		//glDrawArrays(GL_TRIANGLE_FAN, 0, atlas_coord.size() / 6);
		glDrawArrays(GL_TRIANGLES, 0, atlas_coord.size() / 2);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		/*glDeleteBuffers(1, &uvBuf);
		glDeleteBuffers(1, &uvImgBuf);
		glDeleteTextures(1, &tex);*/
		atlas_coord.clear();
		img_coord.clear();
		tri_idx.clear();
		//free(tmpImg);
		/*free(atlas_coord);
		free(img_coord);*/
		glDeleteBuffers(1, &tbo);

	}
	printf("loop time!!!!!!!! : %fms\n", aaa);
	glDeleteBuffers(1, &uvBuf);
	glDeleteBuffers(1, &uvImgBuf);
	//glDeleteTextures(1, &tex);
	/*glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_RGB, GL_UNSIGNED_BYTE, result);*/

	/*vector<uchar> buf(3 * width * height);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_BGR, GL_UNSIGNED_BYTE, buf.data());
	cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, buf.data(), sizeof(uchar) * 8000 * 8000 * 3);
	cv::imshow("111", ch3Image);
	cv::waitKey(0);*/
	/*cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, result, sizeof(uchar) * 8000 * 8000 * 3);
	cv::resize(ch3Image, ch3Image, cv::Size(800, 800));
	cv::imshow("111", ch3Image);
	cv::waitKey(10);*/


	if (draw) {
		vector<uchar> buf(4 * width * height);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, buf.data());
		cv::Mat4b ch4Image;
		ch4Image.create(ATLAS_SIDE, ATLAS_SIDE);
		memcpy(ch4Image.data, buf.data(), sizeof(uchar)*ATLAS_SIDE *ATLAS_SIDE *4);
		cv::imwrite(data_root_path + "/atlas/" + atlas_path + "/" + optimizer->opt_mode + ".png", ch4Image);
		cv::resize(ch4Image, ch4Image, cv::Size(800, 800));
		cv::imshow("Atlas", ch4Image);
		cv::waitKey(10);
	}

	glGenBuffers(1, &pb_Atlas);
	glReadBuffer(GL_FRONT);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pb_Atlas);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 3, 0, GL_DYNAMIC_COPY);
	//glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);


	/*glDeleteRenderbuffers(GL_RENDERBUFFER, &rb);
	glDeleteFramebuffers(GL_FRAMEBUFFER, &fb);*/
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glUseProgram(program);


	sdkStopTimer(&timer);
	float whole_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
	printf("drawing time: %fms\n", whole_time * 1000);
	//return textureID;
}

void Renderer::GenAtlas_UVAtlas_texel(const char* vshader_filename, const char* fshader_filename, string tex_filename, uchar* result, bool draw, int width, int height) {

	load_obj_tex(tex_filename);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);



	GLuint atlas_prog, atlas_shader_v, atlas_shader_f;
	GLuint atlas_texID, img_texID;

	atlas_prog = glCreateProgram();

	if ((atlas_shader_v = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return;
	}

	if ((atlas_shader_f = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return;
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_POLYGON_SMOOTH);

	glAttachShader(atlas_prog, atlas_shader_v);
	glAttachShader(atlas_prog, atlas_shader_f);
	glLinkProgram(atlas_prog);

	glGenFramebuffers(1, &fb);
	glGenRenderbuffers(1, &rb);
	glBindRenderbuffer(GL_RENDERBUFFER, rb);
	glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);
	glFramebufferRenderbuffer(
		GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);


	//glClearColor(0.5f, 0.5f, 1.0f, 1.0f);
	//glClearColor(0.f, 0.f, 0.0f, 1.0f);
	glClearColor(1.f, 1.f, 1.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glViewport(0, 0, width, height);
	glUseProgram(atlas_prog);

	atlas_texID = glGetUniformLocation(atlas_prog, "bufImg");
	img_texID = glGetUniformLocation(atlas_prog, "tempImg");

	optimizer->PrepareRend_UVAtlas();

	uint nImage, width_img, height_img;
	optimizer->GetNumber(&nImage, &width_img, &height_img);

	glm::vec2 uniform_imgsize = glm::vec2(width_img, height_img);
	glUniform2fv(glGetUniformLocation(atlas_prog, "img_size"), 1, glm::value_ptr(uniform_imgsize));

	GLuint texture;
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE1);
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);*/
	//glBindTexture(GL_TEXTURE_2D, texture);
	/*glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);*/


	GLuint tex;

	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);*/
	//glBindTexture(GL_TEXTURE_2D, tex);
	//glBindTexture(GL_TEXTURE_2D, 0);

	GLuint uvBuf, uvImgBuf;
	glGenBuffers(1, &uvBuf);
	glGenBuffers(1, &uvImgBuf);
	GLuint tbo;
	glGenBuffers(1, &tbo);
	/*glGenBuffers(1, &uvBuf);
	glGenBuffers(1, &uvImgBuf);
	vector<float> atlas_coord;
	vector<float> img_coord;

	optimizer->GetAtlasInfoi(&atlas_coord, &img_coord, 0);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), img_coord.data(), GL_STATIC_DRAW);*/

	//system("pause");
	//Sleep(10);

	//uchar* tmpImg = new uchar[width_img*height_img * 3 * 8];
	/*for (int i = 0; i < 8; i++) {
	cv::Mat colortemp = cv::imread("D:\\3D data\\multiview_opt\\stream\\0\\Color_"+to_string(i)+".png");
	cv::cvtColor(colortemp, colortemp, CV_BGRA2RGB);
	memcpy(&tmpImg[width_img*height_img * 3 * i], colortemp.data, sizeof(uchar)*width_img*height_img * 3);
	}*/

	//memcpy(tmpImg, optimizer->hCImages, sizeof(uchar)*width_img*height_img * 3 * 8);

	float aaa = 0.0;
	vector<float> atlas_coord;
	vector<float> img_coord;
	vector<int> tri_idx;

	//GLuint tbo[8];
	//glGenBuffers(8, tbo);
	//for (uint i = 0; i < 8; i++) {
	//	glBindBuffer(GL_TEXTURE_BUFFER, tbo[i]);
	//	glBufferData(GL_TEXTURE_BUFFER, sizeof(float)*width_img*height_img * 3, &optimizer->hCImages[i*width_img*height_img * 3], GL_STATIC_DRAW);

	//	//glGenTextures(1, &tbo_tex);

	//	glBindBuffer(GL_TEXTURE_BUFFER, 0);
	//
	//}

	for (uint i = 0; i < nImage; i++) {

		StopWatchInterface *get_timer = NULL;
		sdkCreateTimer(&get_timer);
		sdkStartTimer(&get_timer);

		/*float *atlas_coord = new float;
		float *img_coord = new float;*/

		optimizer->GetAtlasInfoi_UVAtlas(&atlas_coord, &img_coord, &tri_idx, i);
		for (int ti = 0; ti < tri_idx.size(); ti++) {
			atlas_coord[ti * 6] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3]].x;
			atlas_coord[ti * 6 + 1] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3]].y;
			atlas_coord[ti * 6 + 2] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 1]].x;
			atlas_coord[ti * 6 + 3] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 1]].y;
			atlas_coord[ti * 6 + 4] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 2]].x;
			atlas_coord[ti * 6 + 5] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 2]].y;
		}
		/*uchar* tmpImg = new uchar[width_img*height_img * 3];
		memcpy(tmpImg, &optimizer->hCImages[i*width_img*height_img * 3], sizeof(uchar)*width_img*height_img * 3);*/

		//cv::Mat3b ch4Image;
		//ch4Image.create(height_img, width_img);
		//memcpy(ch4Image.data, tmpImg, sizeof(uchar)*width_img*height_img * 3);
		////cv::resize(ch4Image, ch4Image, cv::Size(800, 800));
		//for (int kkk = 0; kkk < atlas_coord.size() / 2; kkk += 9) {
		//	cv::line(ch4Image, cv::Point(img_coord[kkk], img_coord[kkk + 1]), cv::Point(img_coord[kkk + 3], img_coord[kkk + 4]), cv::Scalar(255, 0, 0));
		//}
		//cv::imshow("111", ch4Image);
		//cv::waitKey(0);
		//GLuint uvBuf, uvImgBuf;
		glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * atlas_coord.size(), atlas_coord.data());
		//glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), atlas_coord.data(), GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * img_coord.size(), img_coord.data());
		//glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), img_coord.data(), GL_STREAM_DRAW);
		//cout << img_coord.size() << "  ,  " << atlas_coord.size() << endl;

		/*uchar* tmp_buf = new uchar[8000 * 8000 * 3];
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height,
		GL_RGB, GL_UNSIGNED_BYTE, tmp_buf);*/
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, tmp_buf);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glUniform1i(atlas_texID, 1);


		///////
		//free(tmp_buf);

		/*GLuint tex;

		glGenTextures(1, &tex);*/
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		/*glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGB, width_img, height_img,
		0, GL_RGB, GL_UNSIGNED_BYTE, &tmpImg[i*width_img*height_img * 3]);*/

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_img, height_img, 0, GL_RGB, GL_UNSIGNED_BYTE, &optimizer->hCImages[i*width_img*height_img * 3]);
		/*glBindBuffer(GL_TEXTURE_BUFFER, tbo);
		glBufferData(GL_TEXTURE_BUFFER, sizeof(float)*width_img*height_img * 3, NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) *width_img*height_img * 3, &optimizer->hCImages[i*width_img*height_img * 3]);
		glBindTexture(GL_TEXTURE_BUFFER, tex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB, tbo);*/
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_img, height_img, GL_RGB, GL_UNSIGNED_BYTE, &optimizer->hCImages[i*width_img*height_img * 3]);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_img, height_img, GL_RGB, GL_UNSIGNED_BYTE, &tmpImg[i*width_img*height_img * 3]);

		sdkStopTimer(&get_timer);
		float get_time = sdkGetAverageTimerValue(&get_timer) / 1000.0f;
		aaa += get_time * 1000;
		printf("%d - time!!!!!!!! : %fms\n", i, get_time * 1000);
		glUniform1i(img_texID, 0);


		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		//glClear(GL_DEPTH_BUFFER_BIT);
		//glDrawArrays(GL_TRIANGLE_FAN, 0, atlas_coord.size() / 6);
		glDrawArrays(GL_TRIANGLES, 0, atlas_coord.size() / 2);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);



		cv::Mat3b ori_img;
		ori_img.create(height_img, width_img);
		memcpy(ori_img.data, &optimizer->hCImages[i*width_img*height_img * 3], sizeof(uchar)*height_img* width_img * 3);
		for (int k = 0; k < img_coord.size(); k += 9) {
			cv::line(ori_img, cv::Point(img_coord[k], img_coord[k + 1]), cv::Point(img_coord[k + 3], img_coord[k + 4]), cv::Vec3b(255, 0, 0), 1);
			cv::line(ori_img, cv::Point(img_coord[k + 3], img_coord[k + 4]), cv::Point(img_coord[k + 6], img_coord[k + 7]), cv::Vec3b(255, 0, 0), 1);
			cv::line(ori_img, cv::Point(img_coord[k + 6], img_coord[k + 7]), cv::Point(img_coord[k], img_coord[k + 1]), cv::Vec3b(255, 0, 0), 1);
		}
		string ori_filename = "D:/3D_data/multiview_opt/atlas/each/test" + to_string(i) + "_ori" + ".bmp";
		//cv::imwrite(ori_filename, ori_img);

		/*glDeleteBuffers(1, &uvBuf);
		glDeleteBuffers(1, &uvImgBuf);
		glDeleteTextures(1, &tex);*/
		atlas_coord.clear();
		img_coord.clear();
		tri_idx.clear();
		//free(tmpImg);
		/*free(atlas_coord);
		free(img_coord);*/
		glDeleteBuffers(1, &tbo);

		vector<uchar> buf(3 * width * height);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height,
			GL_RGB, GL_UNSIGNED_BYTE, buf.data());
		cv::Mat3b ch3Image;
		ch3Image.create(ATLAS_SIDE, ATLAS_SIDE);
		memcpy(ch3Image.data, buf.data(), sizeof(uchar) * ATLAS_SIDE * ATLAS_SIDE * 3);
		string atfilename;
		stringstream paddedNum;
		paddedNum << setw(2) << setfill('0') << i;
		if (draw)
			atfilename = "D:/3D_data/multiview_opt/atlas/each/texel_" + paddedNum.str() + ".png";
		else
			atfilename = "D:/3D_data/multiview_opt/atlas/each/texel_" + paddedNum.str() + "_weighted" + ".png";

		//cv::resize(ch3Image, ch3Image, cv::Size(1024, 1024));
		cv::imwrite(data_root_path + "/unit_test/" + unit_test_path + "/sub_atlas/texel/" + optimizer->opt_mode + "_" + paddedNum.str() + ".png", ch3Image);
		cv::waitKey(10);


		glClearColor(0.f, 0.f, 0.0f, 0.0f);
		//glClearColor(1.f, 1.f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}
	printf("loop time!!!!!!!! : %fms\n", aaa);
	glDeleteBuffers(1, &uvBuf);
	glDeleteBuffers(1, &uvImgBuf);
	//glDeleteTextures(1, &tex);
	/*glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_RGB, GL_UNSIGNED_BYTE, result);*/

	/*vector<uchar> buf(3 * width * height);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_BGR, GL_UNSIGNED_BYTE, buf.data());
	cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, buf.data(), sizeof(uchar) * 8000 * 8000 * 3);
	cv::imshow("111", ch3Image);
	cv::waitKey(0);*/
	/*cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, result, sizeof(uchar) * 8000 * 8000 * 3);
	cv::resize(ch3Image, ch3Image, cv::Size(800, 800));
	cv::imshow("111", ch3Image);
	cv::waitKey(10);*/


	/*if (draw) {
	vector<uchar> buf(3 * width * height);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_RGB, GL_UNSIGNED_BYTE, buf.data());
	cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, buf.data(), sizeof(uchar)*8000*8000 *3);
	cv::imwrite("D:/3D_data/multiview_opt/atlas/test1.bmp", ch3Image);
	cv::resize(ch3Image, ch3Image, cv::Size(800, 800));
	cv::imshow("111", ch3Image);
	cv::waitKey(10);
	}*/

	glGenBuffers(1, &pb_Atlas);
	glReadBuffer(GL_FRONT);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pb_Atlas);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 3, 0, GL_DYNAMIC_COPY);
	//glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);


	/*glDeleteRenderbuffers(GL_RENDERBUFFER, &rb);
	glDeleteFramebuffers(GL_FRAMEBUFFER, &fb);*/
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glUseProgram(program);


	sdkStopTimer(&timer);
	float whole_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
	printf("drawing time: %fms\n", whole_time * 1000);
	//return textureID;
}

void Renderer::GenAtlas_UVAtlas_mask(const char* vshader_filename, const char* fshader_filename, string tex_filename, uchar* result, bool draw, int width, int height) {

	load_obj_tex(tex_filename);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);



	GLuint atlas_prog, atlas_shader_v, atlas_shader_f;
	GLuint atlas_texID, img_texID;

	atlas_prog = glCreateProgram();

	if ((atlas_shader_v = create_shader(vshader_filename, GL_VERTEX_SHADER)) == 0) {
		return;
	}

	if ((atlas_shader_f = create_shader(fshader_filename, GL_FRAGMENT_SHADER)) == 0) {
		return;
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_POLYGON_SMOOTH);

	glAttachShader(atlas_prog, atlas_shader_v);
	glAttachShader(atlas_prog, atlas_shader_f);
	glLinkProgram(atlas_prog);

	glGenFramebuffers(1, &fb);
	glGenRenderbuffers(1, &rb);
	glBindRenderbuffer(GL_RENDERBUFFER, rb);
	glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);
	glFramebufferRenderbuffer(
		GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);


	//glClearColor(0.5f, 0.5f, 1.0f, 1.0f);
	glClearColor(0.f, 0.f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glViewport(0, 0, width, height);
	glUseProgram(atlas_prog);

	atlas_texID = glGetUniformLocation(atlas_prog, "bufImg");
	img_texID = glGetUniformLocation(atlas_prog, "tempImg");

	optimizer->PrepareRend_UVAtlas();

	uint nImage, width_img, height_img;
	optimizer->GetNumber(&nImage, &width_img, &height_img);

	glm::vec2 uniform_imgsize = glm::vec2(width_img, height_img);
	glUniform2fv(glGetUniformLocation(atlas_prog, "img_size"), 1, glm::value_ptr(uniform_imgsize));

	GLuint texture;
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE1);
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);*/
	//glBindTexture(GL_TEXTURE_2D, texture);
	/*glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);*/


	GLuint tex;

	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	/*glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);*/
	//glBindTexture(GL_TEXTURE_2D, tex);
	//glBindTexture(GL_TEXTURE_2D, 0);

	GLuint uvBuf, uvImgBuf;
	glGenBuffers(1, &uvBuf);
	glGenBuffers(1, &uvImgBuf);
	GLuint tbo;
	glGenBuffers(1, &tbo);
	/*glGenBuffers(1, &uvBuf);
	glGenBuffers(1, &uvImgBuf);
	vector<float> atlas_coord;
	vector<float> img_coord;

	optimizer->GetAtlasInfoi(&atlas_coord, &img_coord, 0);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), img_coord.data(), GL_STATIC_DRAW);*/

	//system("pause");
	//Sleep(10);

	//uchar* tmpImg = new uchar[width_img*height_img * 3 * 8];
	/*for (int i = 0; i < 8; i++) {
	cv::Mat colortemp = cv::imread("D:\\3D data\\multiview_opt\\stream\\0\\Color_"+to_string(i)+".png");
	cv::cvtColor(colortemp, colortemp, CV_BGRA2RGB);
	memcpy(&tmpImg[width_img*height_img * 3 * i], colortemp.data, sizeof(uchar)*width_img*height_img * 3);
	}*/

	//memcpy(tmpImg, optimizer->hCImages, sizeof(uchar)*width_img*height_img * 3 * 8);

	float aaa = 0.0;
	vector<float> atlas_coord;
	vector<float> img_coord;
	vector<int> tri_idx;

	//GLuint tbo[8];
	//glGenBuffers(8, tbo);
	//for (uint i = 0; i < 8; i++) {
	//	glBindBuffer(GL_TEXTURE_BUFFER, tbo[i]);
	//	glBufferData(GL_TEXTURE_BUFFER, sizeof(float)*width_img*height_img * 3, &optimizer->hCImages[i*width_img*height_img * 3], GL_STATIC_DRAW);

	//	//glGenTextures(1, &tbo_tex);

	//	glBindBuffer(GL_TEXTURE_BUFFER, 0);
	//
	//}

	for (uint i = 0; i < nImage; i++) {

		StopWatchInterface *get_timer = NULL;
		sdkCreateTimer(&get_timer);
		sdkStartTimer(&get_timer);

		/*float *atlas_coord = new float;
		float *img_coord = new float;*/

		optimizer->GetAtlasInfoi_UVAtlas(&atlas_coord, &img_coord, &tri_idx, i);
		for (int ti = 0; ti < tri_idx.size(); ti++) {
			atlas_coord[ti * 6] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3]].x;
			atlas_coord[ti * 6 + 1] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3]].y;
			atlas_coord[ti * 6 + 2] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 1]].x;
			atlas_coord[ti * 6 + 3] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 1]].y;
			atlas_coord[ti * 6 + 4] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 2]].x;
			atlas_coord[ti * 6 + 5] = main_object_tex.tex_coords[main_object_tex.tex_elements[tri_idx[ti] * 3 + 2]].y;
		}
		/*uchar* tmpImg = new uchar[width_img*height_img * 3];
		memcpy(tmpImg, &optimizer->hCImages[i*width_img*height_img * 3], sizeof(uchar)*width_img*height_img * 3);*/

		//cv::Mat3b ch4Image;
		//ch4Image.create(height_img, width_img);
		//memcpy(ch4Image.data, tmpImg, sizeof(uchar)*width_img*height_img * 3);
		////cv::resize(ch4Image, ch4Image, cv::Size(800, 800));
		//for (int kkk = 0; kkk < atlas_coord.size() / 2; kkk += 9) {
		//	cv::line(ch4Image, cv::Point(img_coord[kkk], img_coord[kkk + 1]), cv::Point(img_coord[kkk + 3], img_coord[kkk + 4]), cv::Scalar(255, 0, 0));
		//}
		//cv::imshow("111", ch4Image);
		//cv::waitKey(0);
		//GLuint uvBuf, uvImgBuf;
		glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * atlas_coord.size(), atlas_coord.data());
		//glBufferData(GL_ARRAY_BUFFER, sizeof(float)*atlas_coord.size(), atlas_coord.data(), GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * img_coord.size(), img_coord.data());
		//glBufferData(GL_ARRAY_BUFFER, sizeof(float)*img_coord.size(), img_coord.data(), GL_STREAM_DRAW);
		//cout << img_coord.size() << "  ,  " << atlas_coord.size() << endl;

		/*uchar* tmp_buf = new uchar[8000 * 8000 * 3];
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height,
		GL_RGB, GL_UNSIGNED_BYTE, tmp_buf);*/
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, tmp_buf);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glUniform1i(atlas_texID, 1);


		///////
		//free(tmp_buf);

		/*GLuint tex;

		glGenTextures(1, &tex);*/
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		/*glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGB, width_img, height_img,
		0, GL_RGB, GL_UNSIGNED_BYTE, &tmpImg[i*width_img*height_img * 3]);*/

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_img, height_img, 0, GL_RGB, GL_UNSIGNED_BYTE, &optimizer->hCImages[i*width_img*height_img * 3]);
		/*glBindBuffer(GL_TEXTURE_BUFFER, tbo);
		glBufferData(GL_TEXTURE_BUFFER, sizeof(float)*width_img*height_img * 3, NULL, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) *width_img*height_img * 3, &optimizer->hCImages[i*width_img*height_img * 3]);
		glBindTexture(GL_TEXTURE_BUFFER, tex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB, tbo);*/
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_img, height_img, GL_RGB, GL_UNSIGNED_BYTE, &optimizer->hCImages[i*width_img*height_img * 3]);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_img, height_img, GL_RGB, GL_UNSIGNED_BYTE, &tmpImg[i*width_img*height_img * 3]);

		sdkStopTimer(&get_timer);
		float get_time = sdkGetAverageTimerValue(&get_timer) / 1000.0f;
		aaa += get_time * 1000;
		printf("%d - time!!!!!!!! : %fms\n", i, get_time * 1000);
		glUniform1i(img_texID, 0);


		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvBuf);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, uvImgBuf);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		//glClear(GL_DEPTH_BUFFER_BIT);
		//glDrawArrays(GL_TRIANGLE_FAN, 0, atlas_coord.size() / 6);
		glDrawArrays(GL_TRIANGLES, 0, atlas_coord.size() / 2);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);



		cv::Mat3b ori_img;
		ori_img.create(height_img, width_img);
		memcpy(ori_img.data, &optimizer->hCImages[i*width_img*height_img * 3], sizeof(uchar)*height_img* width_img * 3);
		for (int k = 0; k < img_coord.size(); k += 9) {
			cv::line(ori_img, cv::Point(img_coord[k], img_coord[k + 1]), cv::Point(img_coord[k + 3], img_coord[k + 4]), cv::Vec3b(255, 0, 0), 1);
			cv::line(ori_img, cv::Point(img_coord[k + 3], img_coord[k + 4]), cv::Point(img_coord[k + 6], img_coord[k + 7]), cv::Vec3b(255, 0, 0), 1);
			cv::line(ori_img, cv::Point(img_coord[k + 6], img_coord[k + 7]), cv::Point(img_coord[k], img_coord[k + 1]), cv::Vec3b(255, 0, 0), 1);
		}
		string ori_filename = "D:/3D_data/multiview_opt/atlas/each/test" + to_string(i) + "_ori" + ".bmp";
		//cv::imwrite(ori_filename, ori_img);

		/*glDeleteBuffers(1, &uvBuf);
		glDeleteBuffers(1, &uvImgBuf);
		glDeleteTextures(1, &tex);*/
		atlas_coord.clear();
		img_coord.clear();
		tri_idx.clear();
		//free(tmpImg);
		/*free(atlas_coord);
		free(img_coord);*/
		glDeleteBuffers(1, &tbo);

		vector<uchar> buf(3 * width * height);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, width, height,
			GL_RGB, GL_UNSIGNED_BYTE, buf.data());
		cv::Mat3b ch3Image;
		ch3Image.create(ATLAS_SIDE, ATLAS_SIDE);
		memcpy(ch3Image.data, buf.data(), sizeof(uchar) * ATLAS_SIDE * ATLAS_SIDE * 3);
		string atfilename;
		stringstream paddedNum;
		paddedNum << setw(2) << setfill('0') << i;
		if (draw)
			atfilename = "D:/3D_data/multiview_opt/atlas/each/mask_" + paddedNum.str() + ".png";
		else
			atfilename = "D:/3D_data/multiview_opt/atlas/each/mask_" + paddedNum.str() + "_weighted" + ".png";

		//cv::resize(ch3Image, ch3Image, cv::Size(1024, 1024));
		cv::imwrite(data_root_path + "/unit_test/" + unit_test_path + "/sub_atlas/mask/" + optimizer->opt_mode + "_" + paddedNum.str() + ".png", ch3Image);
		cv::waitKey(10);


		glClearColor(0.f, 0.f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}
	printf("loop time!!!!!!!! : %fms\n", aaa);
	glDeleteBuffers(1, &uvBuf);
	glDeleteBuffers(1, &uvImgBuf);
	//glDeleteTextures(1, &tex);
	/*glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_RGB, GL_UNSIGNED_BYTE, result);*/

	/*vector<uchar> buf(3 * width * height);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_BGR, GL_UNSIGNED_BYTE, buf.data());
	cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, buf.data(), sizeof(uchar) * 8000 * 8000 * 3);
	cv::imshow("111", ch3Image);
	cv::waitKey(0);*/
	/*cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, result, sizeof(uchar) * 8000 * 8000 * 3);
	cv::resize(ch3Image, ch3Image, cv::Size(800, 800));
	cv::imshow("111", ch3Image);
	cv::waitKey(10);*/


	/*if (draw) {
	vector<uchar> buf(3 * width * height);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height,
	GL_RGB, GL_UNSIGNED_BYTE, buf.data());
	cv::Mat3b ch3Image;
	ch3Image.create(8000, 8000);
	memcpy(ch3Image.data, buf.data(), sizeof(uchar)*8000*8000 *3);
	cv::imwrite("D:/3D_data/multiview_opt/atlas/test1.bmp", ch3Image);
	cv::resize(ch3Image, ch3Image, cv::Size(800, 800));
	cv::imshow("111", ch3Image);
	cv::waitKey(10);
	}*/

	glGenBuffers(1, &pb_Atlas);
	glReadBuffer(GL_FRONT);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pb_Atlas);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 3, 0, GL_DYNAMIC_COPY);
	//glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);


	/*glDeleteRenderbuffers(GL_RENDERBUFFER, &rb);
	glDeleteFramebuffers(GL_FRAMEBUFFER, &fb);*/
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glUseProgram(program);


	sdkStopTimer(&timer);
	float whole_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
	printf("drawing time: %fms\n", whole_time * 1000);
	//return textureID;
}

