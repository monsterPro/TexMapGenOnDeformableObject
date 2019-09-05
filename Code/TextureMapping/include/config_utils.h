#pragma once

#ifndef CONFIG_UTILS
#define CONFIG_UTILS

#include <json\json.h>
#include "any.hpp"

#define MAXTRI 15
#define MAXIMG 80
#define SAMNUM 15

extern std::string data_root_path;
extern std::string tex_mesh_path;
extern std::string atlas_path;
extern std::string unit_test_path;
extern int start_idx;
extern int end_idx;
extern bool is_viewer;

extern int ATLAS_SIDE;
extern int OPT_ITER;
extern bool FROM_FILE;

extern int LAYERNUM;
extern float IRFLX;
extern float IRFLY;
extern float IRCPX;
extern float IRCPY;
extern int IRIMX;
extern int IRIMY;
extern float COFLX;
extern float COFLY;
extern float COCPX;
extern float COCPY;
extern int COIMX;
extern int COIMY;
extern float D_C_EXT[12];
extern float DTEST;

extern float DIVIDER_LAYER;

extern float rendering_spots_c[10][16];
extern float rendering_spots_o[10][16];
extern std::string shader_model;
extern std::string shader_atlas;
extern std::string shader_sub_atlas;
extern std::string shader_mask;

extern bool RECORD_UNIT;
extern std::string mesh_extension;
extern int normal_direction;

using namespace std;

namespace TexMap {
	typedef enum Part {
		MAIN,
		OPTIMIZER,
		RENDERER,
		SIMPLIFIER,
		MAPPER,
		ANY,
		__COUNT__
	};
	static string partName[__COUNT__] = { "main", "optimizer" , "renderer" , "simplifier" , "mapper4D" , "any" };

	class Configurator {
	public:
		void Setconfig(string filename, string case_name);
		void Setcase(string case_name);
		const Json::Value GetValue(Part p);
	private:
		static Json::Value root;
		static string case_now;
		static void updateValues();
	};
	static Configurator config;
}
#endif CONFIG_UTILS