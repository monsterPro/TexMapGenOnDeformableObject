#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h> 
#include <helper_math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <fstream>
#include "Mapper4D.h"
#include "Optimizer.cuh"
#include "shader_utils.h"
#include "Renderer.h"
#include "Simplifier.h"

#include "any.hpp"
#include <Windows.h>

using namespace TexMap;

static void make_unit_dir(string unitTestDir) {
	CreateDirectory(unitTestDir.c_str(), NULL);
	CreateDirectory((unitTestDir + "/decimated_mesh").c_str(), NULL);
	CreateDirectory((unitTestDir + "/geo_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/ml_color_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/naive_color_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/projection").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sl_color_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sub_atlas").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sub_atlas/mask").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sub_atlas/texel").c_str(), NULL);
	for (int i = 0; i < LAYERNUM; i++) {
		CreateDirectory((unitTestDir + "/decimated_mesh/" + to_string(i)).c_str(), NULL);
		CreateDirectory((unitTestDir + "/projection/" + to_string(i)).c_str(), NULL);
	}
}

int main(int argc, char** argv) {
	if (argc < 2) {
		config.Setconfig("./conf.json", "case_hyomin_taekwon");
	}
	else {
		config.Setconfig(argv[1], argv[2]);
	}
	string streamPath = data_root_path + "/stream";
	string mesh4DPathPrefix = data_root_path + "/mesh/Frame";
	string atlasPath = data_root_path + "/atlas/" + atlas_path;
	string unitpath = data_root_path + "/unit_test/" + unit_test_path;
	string texobjFile = data_root_path + tex_mesh_path;
	CreateDirectory((data_root_path + "/atlas").c_str(), NULL);
	CreateDirectory((data_root_path + "/capture").c_str(), NULL);
	CreateDirectory((data_root_path + "/unit_test").c_str(), NULL);
	CreateDirectory(atlasPath.c_str(), NULL);
	if(RECORD_UNIT)
		make_unit_dir(unitpath);

	string v_shader_filename = "./shaders/" + shader_model + ".v.glsl";
	string f_shader_filename = "./shaders/" + shader_model + ".f.glsl";

	TexMap::Renderer* renderer;
	if (!is_viewer) {
		string template_mesh_name = mesh4DPathPrefix + "_" + zeroPadding(to_string(0), 3) + "." + mesh_extension;
		Mapper4D* mapper4D = new Mapper4D(template_mesh_name, mesh4DPathPrefix, streamPath, start_idx, end_idx);
		mapper4D->ConstructVertree_majorVote();

		TexMap::Optimizer* optimizer = new TexMap::Optimizer("naive");
		optimizer->Model4DLoadandMultiUpdate(mapper4D, streamPath + "/sampled_stream.bin");		
		renderer = new TexMap::Renderer(optimizer);
		if (renderer->gl_init(&argc, argv) > 0)
			return -1;
		renderer->init_resources_UVAtlas(v_shader_filename.c_str(), f_shader_filename.c_str(), texobjFile);
		delete renderer;
		delete optimizer;

		optimizer = new TexMap::Optimizer("single");
		optimizer->Model4DLoadandMultiUpdate(mapper4D, streamPath + "/sampled_stream.bin");
		renderer = new TexMap::Renderer(optimizer);
		renderer->init_resources_UVAtlas(v_shader_filename.c_str(), f_shader_filename.c_str(), texobjFile);
		delete renderer;
		delete optimizer;

		optimizer = new TexMap::Optimizer("multi");
		optimizer->Model4DLoadandMultiUpdate(mapper4D, streamPath + "/sampled_stream.bin");
		renderer = new TexMap::Renderer(optimizer);
		renderer->init_resources_UVAtlas(v_shader_filename.c_str(), f_shader_filename.c_str(), texobjFile);

		renderer->init_resources(mesh4DPathPrefix, texobjFile, atlasPath, v_shader_filename.c_str(), f_shader_filename.c_str(), start_idx, end_idx);
		renderer->init_view();
		renderer->mainloop();
		renderer->free_resources();
	}
	else {
		renderer = new TexMap::Renderer();
		if (renderer->gl_init(&argc, argv) > 0)
			return -1;
		renderer->init_resources(mesh4DPathPrefix, texobjFile, atlasPath, v_shader_filename.c_str(), f_shader_filename.c_str(), start_idx, end_idx);
		renderer->init_view();
		renderer->mainloop();
		renderer->free_resources();
	}
	return 0;
}