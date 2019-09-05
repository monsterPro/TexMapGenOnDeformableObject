#pragma once

#ifndef RENDERER_H
#define RENDERER_H

#include "any.hpp"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include <GL/glew.h>
#include <GL/glut.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include "shader_utils.h"
#include "texture.hpp"
#include "Optimizer.cuh"

#define min(X, Y) ((X) < (Y) ? (X) : (Y))
using namespace std;

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

static GLuint program;
static GLuint text;
static GLuint text_s;
static GLuint text_n;
static GLuint img_text;
static GLint attribute_v_coord = -1;
static GLint attribute_v_normal = -1;
static GLint attribute_v_texcoord = -1;
static GLint attribute_v_img_texcoord = -1;

static GLint uniform_m = -1, uniform_v = -1, uniform_p = -1;
static GLint uniform_m_3x3_inv_transp = -1, uniform_v_inv = -1;
static GLint uniform_flags = -1;
static GLuint TextureID = 0;
static GLuint TextureID_s = 0;
static GLuint TextureID_n = 0;
static GLuint ImgTextureID = 0;

static GLuint fb, rb;
static GLuint pb_Atlas;

static GLuint atlas_program;
static GLint attribute_v_atlascoord = -1;
static GLint attribute_img_coord = -1;


class Mesh4D {
private:
	GLuint vbo_vertices, vbo_normals, vbo_texcoords, vbo_img_texcoords, ibo_elements;
	GLuint vertices_offset;
	GLuint time_stamp = 0;
	GLuint end_stamp = 0;
	TexMap::Optimizer *optimizer_ptr;
	uint cW, cH;
	vector<cv::Mat> images;

public:
	glm::mat4 object2world;
	glm::vec4 _flags;

	Mesh4D() : vbo_vertices(0), vbo_normals(0), vbo_texcoords(0), vbo_img_texcoords(0), ibo_elements(0), object2world(glm::mat4(1)), _flags(glm::vec4(0)) {};
	~Mesh4D() {
		if (vbo_vertices != 0) {
			glDeleteBuffers(1, &vbo_vertices);
		}

		if (vbo_normals != 0) {
			glDeleteBuffers(1, &vbo_normals);
		}

		if (ibo_elements != 0) {
			glDeleteBuffers(1, &ibo_elements);
		}

		if (vbo_texcoords != 0) {
			glDeleteBuffers(1, &vbo_texcoords);
		}
	}

	void upload(string ScanMeshPathAndPrefix, string atlasPath, int startIdx, int endIdx) {
		vector<glm::vec4> vert_vec;
		vector<glm::vec3> norm_vec;
		vector<glm::vec2> tex_vec;
		vector<GLuint> elements;
		vector<GLuint> nb_seen;
		
		end_stamp = endIdx;

		

		//read file
		cout << "Start extracting mesh" << endl;
		for (int frameIdx = startIdx; frameIdx < endIdx; ++frameIdx)
		{
			printProgBar(((float)frameIdx / (endIdx * 2.0)) * 100.0);
			Sleep(1.1);
			string filename;
			filename = ScanMeshPathAndPrefix + "_" + zeroPadding(to_string(frameIdx), 3) + "." + mesh_extension;

			std::ifstream file(filename.c_str());
			if (!file.is_open()) {
				printf("Could not open file : %s", filename.c_str());
				return;
			}
			std::string fileType;
			getline(file, fileType);
			unsigned int numV = 0;
			unsigned int numP = 0;
			unsigned int numE = 0;
			file >> numV >> numP >> numE;
			if (frameIdx == startIdx)
				vertices_offset = numV;

			std::string line, value;

			getline(file, line);
			for (unsigned int i = 0; i < numV; i++) {
				getline(file, line);
				std::stringstream linestream(line);
				glm::vec4 v;
				glm::vec2 vt;
				linestream >> v.x;
				linestream >> v.y;
				linestream >> v.z;
				v.w = 1.0;
				vert_vec.push_back(v);
				if (frameIdx > startIdx)
					continue;
				linestream >> vt.x;
				linestream >> vt.y;
				vt = glm::vec2(vt.x, -vt.y);
				tex_vec.push_back(vt);
			}
			if(frameIdx == startIdx)
				for (unsigned int i = 0; i < numP; i++) {
					unsigned int num_vs;
					file >> num_vs;
					for (unsigned int j = 0; j < num_vs; j++) {
						unsigned int idx;
						file >> idx;
						elements.push_back(idx);
					}
				}
		}
		//initialize normal
		norm_vec.resize(vert_vec.size(), glm::vec3(0.0, 0.0, 0.0));
		glm::vec3 sum = glm::vec3(.0, .0, .0);
		for (int frameIdx = startIdx; frameIdx < endIdx; ++frameIdx) {
			printProgBar((((float)frameIdx + endIdx) / (endIdx * 2.0)) * 100.0);
			Sleep(1.1);
			nb_seen.resize(vertices_offset, 0);
			for (unsigned int i = 0; i < elements.size(); i += 3) {
				GLuint ia = elements[i];
				GLuint ib = elements[i + 1];
				GLuint ic = elements[i + 2];
				glm::vec3 normal = glm::normalize(glm::cross(
					glm::vec3(vert_vec[ib]) - glm::vec3(vert_vec[ia]),
					glm::vec3(vert_vec[ic]) - glm::vec3(vert_vec[ia])));

				int v[3];
				v[0] = ia;
				v[1] = ib;
				v[2] = ic;

				for (int j = 0; j < 3; j++) {
					GLuint cur_v = v[j];
					GLuint cur_v_step = v[j] + frameIdx * vertices_offset;
					nb_seen[cur_v]++;

					if (nb_seen[cur_v] == 1) {
						norm_vec[cur_v_step] = normal;
					}
					else {
						// average
						norm_vec[cur_v_step].x = norm_vec[cur_v_step].x * (1.0 - 1.0 / nb_seen[cur_v]) + normal.x * 1.0 / nb_seen[cur_v];
						norm_vec[cur_v_step].y = norm_vec[cur_v_step].y * (1.0 - 1.0 / nb_seen[cur_v]) + normal.y * 1.0 / nb_seen[cur_v];
						norm_vec[cur_v_step].z = norm_vec[cur_v_step].z * (1.0 - 1.0 / nb_seen[cur_v]) + normal.z * 1.0 / nb_seen[cur_v];
						norm_vec[cur_v_step] = glm::normalize(norm_vec[cur_v_step]);
					}
				}
			}
		}

		for (unsigned int i = 0; i < vert_vec.size(); i++) {
			sum.x += glm::vec3(vert_vec[i]).x;
			sum.y += glm::vec3(vert_vec[i]).y;
			sum.z += glm::vec3(vert_vec[i]).z;

		}
		sum /= vert_vec.size();

		for (unsigned int i = 0; i < vert_vec.size(); i++) {
			vert_vec[i] -= glm::vec4(sum, 0.0);
			//mesh->vertices[i] += glm::vec4(0.0,2.0,0.0, 0.0);
		}

		if (vert_vec.size() > 0) {
			glGenBuffers(1, &this->vbo_vertices);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glBufferData(GL_ARRAY_BUFFER, vert_vec.size() * sizeof(vert_vec[0]),
				vert_vec.data(), GL_STATIC_DRAW);
		}

		if (norm_vec.size() > 0) {
			glGenBuffers(1, &this->vbo_normals);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glBufferData(GL_ARRAY_BUFFER, norm_vec.size() * sizeof(norm_vec[0]),
				norm_vec.data(), GL_STATIC_DRAW);
		}

		if (elements.size() > 0) {
			glGenBuffers(1, &this->ibo_elements);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo_elements);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size() * sizeof(elements[0]),
				elements.data(), GL_STATIC_DRAW);
		}

		if (tex_vec.size() > 0) {
			text = loadPNG_custom(atlasPath.c_str());
			const char* uniform_name;
			uniform_name = "tex";
			TextureID = glGetUniformLocation(program, uniform_name);
			glGenBuffers(1, &this->vbo_texcoords);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_texcoords);
			glBufferData(GL_ARRAY_BUFFER, tex_vec.size() * sizeof(tex_vec[0]),
				tex_vec.data(), GL_STATIC_DRAW);
		}
		printProgBar(100);
		cout << "\n End extracting mesh" << endl;

		vert_vec.clear();
		norm_vec.clear();
		tex_vec.clear();
		elements.clear();
		nb_seen.clear();
		
	}
	glm::vec3 upload_UVAtlas(string ScanMeshPathAndPrefix, string atlasPath, string tex_filename, int startIdx, int endIdx) {
		vector<glm::vec4> vert_vec;
		vector<glm::vec4> tmp_vert_vec;
		vector<glm::vec3> norm_vec;
		vector<glm::vec3> tmp_norm_vec;
		vector<glm::vec2> tex_vec;
		vector<GLuint> elements;
		vector<GLuint> tmp_elements;
		vector<GLuint> nb_seen;

		end_stamp = endIdx;


		ifstream in(tex_filename, ios::in);

		if (!in) {
			cerr << "Cannot open " << tex_filename << endl;
			exit(1);
		}
		string line;
		vector<GLuint> vert_elements;
		vector<GLuint> vert_tex_table;

		while (getline(in, line)) {
			if (line.substr(0, 2) == "v ") {
			}
			else if (line.substr(0, 3) == "vt ") {
				istringstream s(line.substr(3));
				glm::vec2 vt;
				s >> vt.x;
				s >> vt.y;
				tex_vec.push_back(vt);
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
				vert_elements.push_back(a);
				vert_elements.push_back(b);
				vert_elements.push_back(c);
				elements.push_back(at);
				elements.push_back(bt);
				elements.push_back(ct);
			}
			else if (line[0] == '#') {
				/* ignoring this line */
			}
			else {
				/* ignoring this line */
			}
		}

		tmp_vert_vec.clear();
		tmp_elements.clear();
		Sleep(1.1);

		string filename;
		filename = ScanMeshPathAndPrefix + "_" + zeroPadding(to_string(startIdx), 3) + "." + mesh_extension;

		std::ifstream file(filename.c_str());
		if (!file.is_open()) {
			printf("Could not open file : %s", filename);
			return{ 0,0,0 };
		}
		std::string fileType;
		getline(file, fileType);
		unsigned int numV = 0;
		unsigned int numP = 0;
		unsigned int numE = 0;
		file >> numV >> numP >> numE;
		vertices_offset = tex_vec.size();


		getline(file, line);
		for (unsigned int i = 0; i < numV; i++) {
			getline(file, line);
			std::stringstream linestream(line);
			glm::vec4 v;
			linestream >> v.x;
			linestream >> v.y;
			linestream >> v.z;
			v.w = 1.0;
			tmp_vert_vec.push_back(v);
		}
		for (unsigned int i = 0; i < numP; i++) {
			getline(file, line);
			std::stringstream linestream(line);
			GLuint e;
			linestream >> e;
			linestream >> e;
			tmp_elements.push_back(e);
			linestream >> e;
			tmp_elements.push_back(e);
			linestream >> e;
			tmp_elements.push_back(e);
		}

		vert_tex_table.resize(tex_vec.size());
		for (int i = 0; i < elements.size(); i++) {
			vert_tex_table[elements[i]] = tmp_elements[i];
		}

		//read file
		cout << "Start extracting mesh" << endl;
		MyMesh deformed_mesh;
		OpenMesh::IO::Options opt;
		deformed_mesh.request_face_normals();
		deformed_mesh.request_vertex_normals();
		for (int frameIdx = startIdx; frameIdx < endIdx; ++frameIdx) {
			tmp_vert_vec.clear();
			tmp_norm_vec.clear();
			tmp_elements.clear();
			printProgBar(((float)frameIdx / endIdx) * 100.0);
			Sleep(1.1);

			string filename;
			filename = ScanMeshPathAndPrefix + "_" + zeroPadding(to_string(frameIdx), 3) + "." + mesh_extension;

			if (!OpenMesh::IO::read_mesh(deformed_mesh, filename, opt))
			{
				std::cerr << "read error\n";
				exit(1);
			}
			deformed_mesh.update_normals();
			deformed_mesh.update_vertex_normals();
			for (MyMesh::VertexIter v_it = deformed_mesh.vertices_begin(); v_it != deformed_mesh.vertices_end(); ++v_it) {
				glm::vec4 v;
				v.x = deformed_mesh.point(*v_it)[0];
				v.y = deformed_mesh.point(*v_it)[1];
				v.z = deformed_mesh.point(*v_it)[2];
				v.w = 1.0;
				tmp_vert_vec.push_back(v);
			}
			for (auto t_v : vert_tex_table) {
				vert_vec.push_back(tmp_vert_vec[t_v]);
			}
			for (MyMesh::VertexIter v_it = deformed_mesh.vertices_begin(); v_it != deformed_mesh.vertices_end(); ++v_it) {
				glm::vec3 n;
				n.x = deformed_mesh.normal(*v_it)[0];
				n.y = deformed_mesh.normal(*v_it)[1];
				n.z = deformed_mesh.normal(*v_it)[2];
				tmp_norm_vec.push_back(n);
			}
			for (auto t_v : vert_tex_table) {
				norm_vec.push_back(tmp_norm_vec[t_v]);
			}
		}
		
		glm::vec3 sum = glm::vec3(.0, .0, .0);
		for (unsigned int i = 0; i < vert_vec.size(); i++) {
			sum.x += glm::vec3(vert_vec[i]).x;
			sum.y += glm::vec3(vert_vec[i]).y;
			sum.z += glm::vec3(vert_vec[i]).z;

		}
		sum /= vert_vec.size();

		for (unsigned int i = 0; i < vert_vec.size(); i++) {
			vert_vec[i] -= glm::vec4(sum, 0.0);
			//mesh->vertices[i] += glm::vec4(0.0,2.0,0.0, 0.0);
		}

		if (vert_vec.size() > 0) {
			glGenBuffers(1, &this->vbo_vertices);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glBufferData(GL_ARRAY_BUFFER, vert_vec.size() * sizeof(vert_vec[0]),
				vert_vec.data(), GL_STATIC_DRAW);
		}

		if (norm_vec.size() > 0) {
			glGenBuffers(1, &this->vbo_normals);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glBufferData(GL_ARRAY_BUFFER, norm_vec.size() * sizeof(norm_vec[0]),
				norm_vec.data(), GL_STATIC_DRAW);
		}

		if (elements.size() > 0) {
			glGenBuffers(1, &this->ibo_elements);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo_elements);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size() * sizeof(elements[0]),
				elements.data(), GL_STATIC_DRAW);
		}

		if (tex_vec.size() > 0) {
			//text = loadPNG_custom(atlasPath.c_str());
			glActiveTexture(GL_TEXTURE0);
			glEnable(GL_TEXTURE_2D);
			text = loadPNG_edgesmooth((atlasPath + "/multi.png").c_str(), tex_vec, elements);
			const char* uniform_name;
			uniform_name = "tex";
			TextureID = glGetUniformLocation(program, uniform_name);
			glDisable(GL_TEXTURE_2D);
			glGenBuffers(1, &this->vbo_texcoords);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_texcoords);
			glBufferData(GL_ARRAY_BUFFER, tex_vec.size() * sizeof(tex_vec[0]),
				tex_vec.data(), GL_STATIC_DRAW);

			glActiveTexture(GL_TEXTURE1);
			glEnable(GL_TEXTURE_2D);
			text_s = loadPNG_edgesmooth((atlasPath + "/single.png").c_str(), tex_vec, elements);
			const char* uniform_name_s;
			uniform_name_s = "tex_s";
			TextureID_s = glGetUniformLocation(program, uniform_name_s);
			glDisable(GL_TEXTURE_2D);


			glActiveTexture(GL_TEXTURE2);
			glEnable(GL_TEXTURE_2D);
			text_n = loadPNG_edgesmooth((atlasPath + "/naive.png").c_str(), tex_vec, elements);
			const char* uniform_name_n;
			uniform_name_n = "tex_n";
			TextureID_n = glGetUniformLocation(program, uniform_name_n);
			glDisable(GL_TEXTURE_2D);
		}
		printProgBar(100);
		cout << "\n End extracting mesh" << endl;

		vert_vec.clear();
		norm_vec.clear();
		tex_vec.clear();
		elements.clear();
		nb_seen.clear();

		return sum;
	}

	void up_time_stamp() {
		time_stamp++;
		time_stamp %= end_stamp;
	}
	void down_time_stamp() {
		time_stamp--;
		time_stamp = (time_stamp + end_stamp) % end_stamp;
	}
	void set_time_stamp(int stamp) {
		time_stamp = stamp;
	}
	int get_time_stamp() {
		return time_stamp;
	}
	void draw() {
		if (this->vbo_vertices != 0) {
			glEnableVertexAttribArray(attribute_v_coord);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glVertexAttribPointer(
				attribute_v_coord,			 // attribute
				4,							 // number of elements per vertex, here (x,y,z,w)
				GL_FLOAT,					 // the type of each element
				GL_FALSE,					 // take our values as-is
				0,							 // no extra data between each position
				(void*)(vertices_offset * time_stamp * sizeof(glm::vec4))                   // offset of first element
			);
		}

		if (this->vbo_normals != 0) {
			glEnableVertexAttribArray(attribute_v_normal);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glVertexAttribPointer(
				attribute_v_normal, // attribute
				3,                  // number of elements per vertex, here (x,y,z)
				GL_FLOAT,           // the type of each element
				GL_FALSE,           // take our values as-is
				0,                  // no extra data between each position
				(void*)(vertices_offset * time_stamp * sizeof(glm::vec3))                   // offset of first element
			);
		}

		//if (TextureID == -1) {
		//fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		//exit(0);
		//}
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glBindTexture(GL_TEXTURE_2D, text);
		glUniform1i(TextureID, 0);
		glDisable(GL_TEXTURE_2D);

		glActiveTexture(GL_TEXTURE1);
		glEnable(GL_TEXTURE_2D);
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glBindTexture(GL_TEXTURE_2D, text_s);
		glUniform1i(TextureID_s, 1);
		glDisable(GL_TEXTURE_2D);


		glActiveTexture(GL_TEXTURE2);
		glEnable(GL_TEXTURE_2D);
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glBindTexture(GL_TEXTURE_2D, text_n);
		glUniform1i(TextureID_n, 2);
		glDisable(GL_TEXTURE_2D);

		/*glActiveTexture(GL_TEXTURE1);
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glBindTexture(GL_TEXTURE_2D, img_text);*/

		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cW, cH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cW, cH, GL_RGB, GL_UNSIGNED_BYTE, &optimizer_ptr->mapper4D_ptr->colorImages_dummy[time_stamp].data);

		/*glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cW, cH, GL_BGR, GL_UNSIGNED_BYTE, images[time_stamp].data);
		glUniform1i(ImgTextureID, 1);*/

		//GLfloat planeCoefficients[4] = {1, 0, 0, 0 };
		//glTexGenfv(GL_S, GL_OBJECT_PLANE, planeCoefficients);
		//glEnable(GL_TEXTURE_GEN_S);

		if (this->vbo_texcoords != 0) {
			glEnableVertexAttribArray(attribute_v_texcoord);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_texcoords);
			glVertexAttribPointer(
				attribute_v_texcoord,
				2,
				GL_FLOAT,
				GL_FALSE,
				0,
				0
			);
		}

		if (this->vbo_img_texcoords != 0) {
			glEnableVertexAttribArray(attribute_v_img_texcoord);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_img_texcoords);
			glVertexAttribPointer(
				attribute_v_img_texcoord,
				3,
				GL_FLOAT,
				GL_FALSE,
				0,
				(void*)(vertices_offset * time_stamp * sizeof(glm::vec3))
			);
		}

		/* Apply object's transformation matrix */
		glUniformMatrix4fv(uniform_m, 1, GL_FALSE, glm::value_ptr(this->object2world));
		/* Apply flags */
		glUniform4fv(uniform_flags, 1, glm::value_ptr(this->_flags));
		/* Transform normal vectors with transpose of inverse of upper left
		3x3 model matrix (ex-gl_NormalMatrix): */
		glm::mat3 m_3x3_inv_transp = glm::transpose(glm::inverse(glm::mat3(this->object2world)));
		glUniformMatrix3fv(uniform_m_3x3_inv_transp, 1, GL_FALSE, glm::value_ptr(m_3x3_inv_transp));
		glPolygonMode(GL_FRONT, GL_FILL);
		glPolygonMode(GL_BACK, GL_FILL);
		/* Push each element in buffer_vertices to the vertex shader */
		if (this->ibo_elements != 0) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo_elements);
			int size;
			glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
			glDrawElements(GL_TRIANGLES, size / sizeof(GLuint), GL_UNSIGNED_INT, 0);
		}
		else {
			glDrawArrays(GL_TRIANGLES, 0, vertices_offset);
		}

		if (this->vbo_normals != 0) {
			glDisableVertexAttribArray(attribute_v_normal);
		}

		if (this->vbo_vertices != 0) {
			glDisableVertexAttribArray(attribute_v_coord);
		}

		if (this->vbo_texcoords != 0) {
			glDisableVertexAttribArray(attribute_v_texcoord);
		}
		if (this->vbo_img_texcoords != 0) {
			glDisableVertexAttribArray(attribute_v_img_texcoord);
		}
	}
	void drawwireframe() {
		if (this->vbo_vertices != 0) {
			glEnableVertexAttribArray(attribute_v_coord);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glVertexAttribPointer(
				attribute_v_coord,  // attribute
				4,                  // number of elements per vertex, here (x,y,z,w)
				GL_FLOAT,           // the type of each element
				GL_FALSE,           // take our values as-is
				0,                  // no extra data between each position
				(void*)(vertices_offset * time_stamp * sizeof(glm::vec4))                       // offset of first element
			);
		}

		if (this->vbo_normals != 0) {
			glEnableVertexAttribArray(attribute_v_normal);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glVertexAttribPointer(
				attribute_v_normal, // attribute
				3,                  // number of elements per vertex, here (x,y,z)
				GL_FLOAT,           // the type of each element
				GL_FALSE,           // take our values as-is
				0,                  // no extra data between each position
				(void*)(vertices_offset * time_stamp * sizeof(glm::vec3))                       // offset of first element
			);
		}


		/* Apply object's transformation matrix */
		glUniformMatrix4fv(uniform_m, 1, GL_FALSE, glm::value_ptr(this->object2world));
		/* Transform normal vectors with transpose of inverse of upper left
		3x3 model matrix (ex-gl_NormalMatrix): */
		glm::mat3 m_3x3_inv_transp = glm::transpose(glm::inverse(glm::mat3(this->object2world)));
		glUniformMatrix3fv(uniform_m_3x3_inv_transp, 1, GL_FALSE, glm::value_ptr(m_3x3_inv_transp));
		glPolygonMode(GL_FRONT, GL_LINE);
		glPolygonMode(GL_BACK, GL_LINE);
		glLineWidth(1.5);

		/* Push each element in buffer_vertices to the vertex shader */
		glColor4f(0.0, 0.0, 0.0, 1.0);
		glDrawArrays(GL_TRIANGLES, 0, vertices_offset);


		if (this->vbo_normals != 0) {
			glDisableVertexAttribArray(attribute_v_normal);
		}

		if (this->vbo_vertices != 0) {
			glDisableVertexAttribArray(attribute_v_coord);
		}

		if (this->vbo_texcoords != 0) {
			glDisableVertexAttribArray(attribute_v_texcoord);
		}
	}
};

class Mesh {
private:
	GLuint vbo_vertices, vbo_normals, vbo_texcoords, ibo_elements;

public:
	vector<glm::vec4> vertices;
	vector<glm::vec3> normals;
	vector<glm::vec2> tex_coords;
	vector<GLuint> elements;
	vector<GLuint> tex_elements;
	int num_img;
	glm::mat4 object2world;

	Mesh() : vbo_vertices(0), vbo_normals(0), vbo_texcoords(0), ibo_elements(0), object2world(glm::mat4(1)) {};
	~Mesh() {
		if (vbo_vertices != 0) {
			glDeleteBuffers(1, &vbo_vertices);
		}

		if (vbo_normals != 0) {
			glDeleteBuffers(1, &vbo_normals);
		}

		if (ibo_elements != 0) {
			glDeleteBuffers(1, &ibo_elements);
		}

		if (vbo_texcoords != 0) {
			glDeleteBuffers(1, &vbo_texcoords);
		}
	}

	/**
	* Store object vertices, normals and/or elements in graphic card
	* buffers
	*/

	void upload() {
		if (this->vertices.size() > 0) {
			glGenBuffers(1, &this->vbo_vertices);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(this->vertices[0]),
				this->vertices.data(), GL_STATIC_DRAW);
		}

		if (this->normals.size() > 0) {
			glGenBuffers(1, &this->vbo_normals);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glBufferData(GL_ARRAY_BUFFER, this->normals.size() * sizeof(this->normals[0]),
				this->normals.data(), GL_STATIC_DRAW);
		}

		if (this->elements.size() > 0) {
			glGenBuffers(1, &this->ibo_elements);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo_elements);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->elements.size() * sizeof(this->elements[0]),
				this->elements.data(), GL_STATIC_DRAW);
		}

		if (this->tex_coords.size() == 0) {
		for (size_t i = 0; i < this->vertices.size(); i++) {
		float s, t;
		s = ((float)(rand() % 1000)) / 1000;
		t = ((float)(rand() % 1000)) / 1000;
		this->tex_coords.push_back(glm::vec2(s, t));
		}
		}

	}
	void upload(string texturePath) {
		if (this->vertices.size() > 0) {
			glGenBuffers(1, &this->vbo_vertices);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(this->vertices[0]),
				this->vertices.data(), GL_STATIC_DRAW);
		}

		if (this->normals.size() > 0) {
			glGenBuffers(1, &this->vbo_normals);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glBufferData(GL_ARRAY_BUFFER, this->normals.size() * sizeof(this->normals[0]),
				this->normals.data(), GL_STATIC_DRAW);
		}

		if (this->elements.size() > 0) {
			glGenBuffers(1, &this->ibo_elements);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo_elements);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->elements.size() * sizeof(this->elements[0]),
				this->elements.data(), GL_STATIC_DRAW);
		}
		
		if (this->tex_coords.size() > 0) {
			text = loadBMP_custom(texturePath.c_str());
			const char* uniform_name;
			uniform_name = "tex";
			TextureID = glGetUniformLocation(program, uniform_name);
			glGenBuffers(1, &this->vbo_texcoords);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_texcoords);
			glBufferData(GL_ARRAY_BUFFER, this->tex_coords.size() * sizeof(this->tex_coords[0]),
				this->tex_coords.data(), GL_STATIC_DRAW);
		}
	}
	void upload(uchar* atlasdata, uint atlaslenght) {
		if (this->vertices.size() > 0) {
			glGenBuffers(1, &this->vbo_vertices);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(this->vertices[0]),
				this->vertices.data(), GL_STATIC_DRAW);
		}

		if (this->normals.size() > 0) {
			glGenBuffers(1, &this->vbo_normals);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glBufferData(GL_ARRAY_BUFFER, this->normals.size() * sizeof(this->normals[0]),
				this->normals.data(), GL_STATIC_DRAW);
		}

		if (this->elements.size() > 0) {
			glGenBuffers(1, &this->ibo_elements);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo_elements);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->elements.size() * sizeof(this->elements[0]),
				this->elements.data(), GL_STATIC_DRAW);
		}

		if (this->tex_coords.size() > 0) {

			glGenTextures(1, &text);
			glBindTexture(GL_TEXTURE_2D, text);
			//////////////////////////
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pb_Atlas);
			//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			//glReadPixels(0, 0, atlaslenght, atlaslenght, GL_RGB, GL_UNSIGNED_BYTE, 0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, atlaslenght, atlaslenght, 0, GL_BGR, GL_UNSIGNED_BYTE, 0);
			//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			//////////////////////////

			//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, atlaslenght, atlaslenght, 0, GL_BGR, GL_UNSIGNED_BYTE, atlasdata);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

			glBindTexture(GL_TEXTURE_2D, 0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			const char* uniform_name;
			uniform_name = "tex";
			TextureID = glGetUniformLocation(program, uniform_name);
			glGenBuffers(1, &this->vbo_texcoords);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_texcoords);
			glBufferData(GL_ARRAY_BUFFER, this->tex_coords.size() * sizeof(this->tex_coords[0]),
				this->tex_coords.data(), GL_STATIC_DRAW);
		}
	}

	/**
	* Draw the object
	*/
	void draw() {
		if (this->vbo_vertices != 0) {
			glEnableVertexAttribArray(attribute_v_coord);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glVertexAttribPointer(
				attribute_v_coord,  // attribute
				4,                  // number of elements per vertex, here (x,y,z,w)
				GL_FLOAT,           // the type of each element
				GL_FALSE,           // take our values as-is
				0,                  // no extra data between each position
				0                   // offset of first element
			);
		}

		if (this->vbo_normals != 0) {
			glEnableVertexAttribArray(attribute_v_normal);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glVertexAttribPointer(
				attribute_v_normal, // attribute
				3,                  // number of elements per vertex, here (x,y,z)
				GL_FLOAT,           // the type of each element
				GL_FALSE,           // take our values as-is
				0,                  // no extra data between each position
				0                   // offset of first element
			);
		}

		//if (TextureID == -1) {
		//fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
		//exit(0);
		//}
		glActiveTexture(GL_TEXTURE0);
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
		glBindTexture(GL_TEXTURE_2D, text);
		glUniform1i(TextureID, 0);
		//GLfloat planeCoefficients[4] = {1, 0, 0, 0 };
		//glTexGenfv(GL_S, GL_OBJECT_PLANE, planeCoefficients);
		//glEnable(GL_TEXTURE_GEN_S);

		if (this->vbo_texcoords != 0) {
			glEnableVertexAttribArray(attribute_v_texcoord);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_texcoords);
			glVertexAttribPointer(
				attribute_v_texcoord,
				2,
				GL_FLOAT,
				GL_FALSE,
				0,
				0
			);
		}

		/* Apply object's transformation matrix */
		glUniformMatrix4fv(uniform_m, 1, GL_FALSE, glm::value_ptr(this->object2world));
		/* Transform normal vectors with transpose of inverse of upper left
		3x3 model matrix (ex-gl_NormalMatrix): */
		glm::mat3 m_3x3_inv_transp = glm::transpose(glm::inverse(glm::mat3(this->object2world)));
		glUniformMatrix3fv(uniform_m_3x3_inv_transp, 1, GL_FALSE, glm::value_ptr(m_3x3_inv_transp));
		glPolygonMode(GL_FRONT, GL_FILL);
		glPolygonMode(GL_BACK, GL_FILL);
		/* Push each element in buffer_vertices to the vertex shader */
		if (this->ibo_elements != 0) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo_elements);
			int size;
			glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
			glDrawElements(GL_TRIANGLES, size / sizeof(GLuint), GL_UNSIGNED_INT, 0);
		}
		else {
			glDrawArrays(GL_TRIANGLES, 0, this->vertices.size());
		}

		if (this->vbo_normals != 0) {
			glDisableVertexAttribArray(attribute_v_normal);
		}

		if (this->vbo_vertices != 0) {
			glDisableVertexAttribArray(attribute_v_coord);
		}

		if (this->vbo_texcoords != 0) {
			glDisableVertexAttribArray(attribute_v_texcoord);
		}
	}

	/**
	* Draw the atlas
	*/
	void drawatlas() {
		GLuint fbo_handle, fbo_texture_handle;

		GLuint texture_width = 8000;
		GLuint texture_height = 8000;

		// generate texture

		glGenTextures(1, &fbo_texture_handle);
		glBindTexture(GL_TEXTURE_2D, fbo_texture_handle);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

		// generate framebuffer

		glGenFramebuffers(1, &fbo_handle);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo_handle);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture_handle, 0);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo_handle);

		glPushAttrib(GL_VIEWPORT_BIT);
		glViewport(0, 0, texture_width, texture_height);
	}

	void drawwireframe() {
		if (this->vbo_vertices != 0) {
			glEnableVertexAttribArray(attribute_v_coord);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertices);
			glVertexAttribPointer(
				attribute_v_coord,  // attribute
				4,                  // number of elements per vertex, here (x,y,z,w)
				GL_FLOAT,           // the type of each element
				GL_FALSE,           // take our values as-is
				0,                  // no extra data between each position
				0                   // offset of first element
			);
		}

		if (this->vbo_normals != 0) {
			glEnableVertexAttribArray(attribute_v_normal);
			glBindBuffer(GL_ARRAY_BUFFER, this->vbo_normals);
			glVertexAttribPointer(
				attribute_v_normal, // attribute
				3,                  // number of elements per vertex, here (x,y,z)
				GL_FLOAT,           // the type of each element
				GL_FALSE,           // take our values as-is
				0,                  // no extra data between each position
				0                   // offset of first element
			);
		}


		/* Apply object's transformation matrix */
		glUniformMatrix4fv(uniform_m, 1, GL_FALSE, glm::value_ptr(this->object2world));
		/* Transform normal vectors with transpose of inverse of upper left
		3x3 model matrix (ex-gl_NormalMatrix): */
		glm::mat3 m_3x3_inv_transp = glm::transpose(glm::inverse(glm::mat3(this->object2world)));
		glUniformMatrix3fv(uniform_m_3x3_inv_transp, 1, GL_FALSE, glm::value_ptr(m_3x3_inv_transp));
		glPolygonMode(GL_FRONT, GL_LINE);
		glPolygonMode(GL_BACK, GL_LINE);
		glLineWidth(1.5);

		/* Push each element in buffer_vertices to the vertex shader */
		glColor4f(0.0, 0.0, 0.0, 1.0);
		glDrawArrays(GL_TRIANGLES, 0, this->vertices.size());


		if (this->vbo_normals != 0) {
			glDisableVertexAttribArray(attribute_v_normal);
		}

		if (this->vbo_vertices != 0) {
			glDisableVertexAttribArray(attribute_v_coord);
		}

		if (this->vbo_texcoords != 0) {
			glDisableVertexAttribArray(attribute_v_texcoord);
		}
	}

	/**
	* Draw object bounding box
	*/
	void draw_bbox() {
		if (this->vertices.size() == 0) {
			return;
		}

		// Cube 1x1x1, centered on origin
		GLfloat vertices[] = {
			-0.5, -0.5, -0.5, 1.0,
			0.5, -0.5, -0.5, 1.0,
			0.5,  0.5, -0.5, 1.0,
			-0.5,  0.5, -0.5, 1.0,
			-0.5, -0.5,  0.5, 1.0,
			0.5, -0.5,  0.5, 1.0,
			0.5,  0.5,  0.5, 1.0,
			-0.5,  0.5,  0.5, 1.0,
		};
		GLuint vbo_vertices;
		glGenBuffers(1, &vbo_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		GLushort elements[] = {
			0, 1, 2, 3,
			4, 5, 6, 7,
			0, 4, 1, 5, 2, 6, 3, 7
		};
		GLuint ibo_elements;
		glGenBuffers(1, &ibo_elements);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		GLfloat
			min_x, max_x,
			min_y, max_y,
			min_z, max_z;
		min_x = max_x = this->vertices[0].x;
		min_y = max_y = this->vertices[0].y;
		min_z = max_z = this->vertices[0].z;

		for (unsigned int i = 0; i < this->vertices.size(); i++) {
			if (this->vertices[i].x < min_x) {
				min_x = this->vertices[i].x;
			}

			if (this->vertices[i].x > max_x) {
				max_x = this->vertices[i].x;
			}

			if (this->vertices[i].y < min_y) {
				min_y = this->vertices[i].y;
			}

			if (this->vertices[i].y > max_y) {
				max_y = this->vertices[i].y;
			}

			if (this->vertices[i].z < min_z) {
				min_z = this->vertices[i].z;
			}

			if (this->vertices[i].z > max_z) {
				max_z = this->vertices[i].z;
			}
		}

		glm::vec3 size = glm::vec3(max_x - min_x, max_y - min_y, max_z - min_z);
		glm::vec3 center = glm::vec3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2);
		glm::mat4 transform = glm::scale(glm::mat4(1), size) * glm::translate(glm::mat4(1), center);

		/* Apply object's transformation matrix */
		glm::mat4 m = this->object2world * transform;
		glUniformMatrix4fv(uniform_m, 1, GL_FALSE, glm::value_ptr(m));

		glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
		glEnableVertexAttribArray(attribute_v_coord);
		glVertexAttribPointer(
			attribute_v_coord,  // attribute
			4,                  // number of elements per vertex, here (x,y,z,w)
			GL_FLOAT,           // the type of each element
			GL_FALSE,           // take our values as-is
			0,                  // no extra data between each position
			0                   // offset of first element
		);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
		glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, 0);
		glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, (GLvoid*)(4 * sizeof(GLushort)));
		glDrawElements(GL_LINES, 8, GL_UNSIGNED_SHORT, (GLvoid*)(8 * sizeof(GLushort)));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		glDisableVertexAttribArray(attribute_v_coord);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDeleteBuffers(1, &vbo_vertices);
		glDeleteBuffers(1, &ibo_elements);
	}
}static light_bbox;


namespace TexMap {
	class Renderer {
	private:

		Mesh4D main_object4D;
		bool object4D_play = false;
		float fps_play = 20;
		Mesh main_object;
		Mesh main_object_tex;
		Optimizer* optimizer;
		int screen_width = 800, screen_height = 600;
		enum MODES { MODE_OBJECT, MODE_CAMERA, MODE_LIGHT, MODE_LAST } view_mode;
		glm::mat4 transforms[MODE_LAST];
		int rotY_direction = 0, rotX_direction = 0, transZ_direction = 0, strife = 0;
		float speed_factor = 1;
		int last_ticks = 0;
		int last_mx = 0, last_my = 0, cur_mx = 0, cur_my = 0;
		int arcball_on = false;
		unsigned int fps_start = glutGet(GLUT_ELAPSED_TIME);
		unsigned int fps_frames = 0;
		float fps_now = 30.0;

		string viewnum = "0";
		string tmp_s;

		void convertMesh();
		GLuint vbo_imgcoords, vbo_atlascoords;
		vector<glm::vec2> atlas_coords;
		vector<glm::vec3>* img_tex_coords;

		glm::vec3 get_arcball_vector(int x, int y);

		glm::vec3 objmov = { 0,0,0 };
		
	public:
		Renderer();
		Renderer(Optimizer* optimizer);

		void link_optimizer(Optimizer* optimizer) {
			this->optimizer = optimizer;
		}
		int gl_init(int *argc, char** argv) {
			glutInit(argc, argv);
			glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
			glutInitWindowSize(screen_width, screen_height);
			glutCreateWindow("Model Viewer");

			GLenum glew_status = glewInit();

			if (glew_status != GLEW_OK) {
				fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_status));
				return 1;
			}

			if (!GLEW_VERSION_2_0) {
				fprintf(stderr, "Error: your graphic card does not support OpenGL 2.0\n");
				return 1;
			}
			return 0;
		}
		int init_resources(string ScanMeshPathAndPrefix, string bmp_filename, char * vshader_filename, char * fshader_filename, int startIdx, int endIdx);
		void SetMesh4D(string ScanMeshPathAndPrefix, int startIdx, int endIdx);
		int init_resources(string ScanMeshPathAndPrefix, string tex_filename, string bmp_filename, const char * vshader_filename, const char * fshader_filename, int startIdx, int endIdx);
		int init_resources(char* bmp_filename, char* vshader_filename, char* fshader_filename);
		int init_resources(char* vshader_filename, char* fshader_filename);
		int init_resources_UVAtlas(const char * vshader_filename, const char * fshader_filename, string tex_filename);
		int init_resources(uchar * data, char * vshader_filename, char * fshader_filename);
		void load_obj(string filename);
		void load_obj_tex(string filename);
		void init_view();
		void GenAtlas_UVAtlas(const char* vshader_filename, const char* fshader_filename, string tex_filename, uchar* result, bool draw = false, int width = ATLAS_SIDE, int height = ATLAS_SIDE);
		void GenAtlas_UVAtlas_texel(const char* vshader_filename, const char* fshader_filename, string tex_filename, uchar* result, bool draw = false, int width = ATLAS_SIDE, int height = ATLAS_SIDE);
		void GenAtlas_UVAtlas_mask(const char* vshader_filename, const char* fshader_filename, string tex_filename, uchar* result, bool draw = false, int width = ATLAS_SIDE, int height = ATLAS_SIDE);
		void ScreenCapture(const char *strFilePath);
		
		void onSpecial(int key, int x, int y);
		void onKeyboard(unsigned char key, int x, int y);
		void onSpecialUp(int key, int x, int y);
		void onKeyboardUp(unsigned char key, int x, int y);
		void onDisplay();
		void onMouse(int button, int state, int x, int y);
		void onMotion(int x, int y);
		void onReshape(int width, int height);

		
		
		void logic();
		void draw();
		void free_resources();
		void mainloop();
	};
}


#endif RENDERER_H