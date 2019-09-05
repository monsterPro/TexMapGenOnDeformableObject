#include <config_utils.h>

using namespace TexMap;
using namespace std;

//const extern int MAXTRI = 10;
//const extern int MAXIMG = 30;
//const extern int SAMNUM = 15;
//__device__ extern float sampling_weights[3][SAMNUM] = { 0.5, 0.25, 0.25, 0.8, 0.1, 0.1, 0.5, 0.0, 0.5, 0.2, 0.6, 0.2, 0.1, 0.5, 0.4,
//0.25, 0.5, 0.25, 0.1, 0.8, 0.1, 0.5, 0.5, 0.0, 0.2, 0.2, 0.6, 0.5, 0.4, 0.1,
//0.25, 0.25, 0.5, 0.1, 0.1, 0.8, 0.0, 0.5, 0.5, 0.6, 0.2, 0.2, 0.4, 0.1, 0.5 };
//__device__ extern float EDGE_WEIGHT = (0.5 * 255);

// case conf
Json::Value Configurator::root = 0;
string Configurator::case_now = "";

// main conf
string data_root_path;
string tex_mesh_path;
string atlas_path;
string unit_test_path;
int start_idx;
int end_idx;
bool is_viewer;

// optimizer conf
int ATLAS_SIDE;
int OPT_ITER;
bool FROM_FILE;

// mapper conf
int LAYERNUM;
float IRFLX;
float IRFLY;
float IRCPX;
float IRCPY;
int IRIMX;
int IRIMY;
float COFLX;
float COFLY;
float COCPX;
float COCPY;
int COIMX;
int COIMY;
float D_C_EXT[12];
float DTEST;

// simplifier conf
float DIVIDER_LAYER;

// renderer conf
float rendering_spots_c[10][16];
float rendering_spots_o[10][16];
string shader_model;
string shader_atlas;
string shader_sub_atlas;
string shader_mask;

// others
bool RECORD_UNIT;
string mesh_extension;
int normal_direction;

void Configurator::Setconfig(string filename, string case_name) {
	ifstream openfile(filename);
	string config_doc;
	if (openfile.is_open()) {
		string line;
		while (getline(openfile, line)) {
			config_doc += line;
		}
		openfile.close();
	}

	Json::Reader reader;
	bool parsingSuccessful = reader.parse(config_doc, root);
	Setcase(case_name);
}
void Configurator::Setcase(string case_name) {
	case_now = case_name;
	updateValues();
}
const Json::Value Configurator::GetValue(Part p) {
	return root[case_now][partName[p]];
}
void Configurator::updateValues() {
	start_idx = config.GetValue(MAIN)["start_idx"].asInt();
	end_idx = config.GetValue(MAIN)["end_idx"].asInt();
	is_viewer = config.GetValue(MAIN)["is_viewer"].asBool();
	data_root_path = config.GetValue(MAIN)["data_root_path"].asString();
	tex_mesh_path = config.GetValue(MAIN)["tex_mesh_path"].asString();
	atlas_path = config.GetValue(MAIN)["atlas_path"].asString();
	unit_test_path = config.GetValue(MAIN)["unit_test_path"].asString();

	ATLAS_SIDE = root[case_now][partName[OPTIMIZER]]["atlas_side_length"].asInt();
	OPT_ITER = root[case_now][partName[OPTIMIZER]]["opt_iter_per_layer"].asInt();

	LAYERNUM = root[case_now][partName[MAPPER]]["layer_number"].asInt();
	IRFLX = root[case_now][partName[MAPPER]]["depth_intrinsic"]["fx"].asFloat();
	IRFLY = root[case_now][partName[MAPPER]]["depth_intrinsic"]["fy"].asFloat();
	IRCPX = root[case_now][partName[MAPPER]]["depth_intrinsic"]["cx"].asFloat();
	IRCPY = root[case_now][partName[MAPPER]]["depth_intrinsic"]["cy"].asFloat();
	IRIMX = root[case_now][partName[MAPPER]]["depth_intrinsic"]["width"].asInt();
	IRIMY = root[case_now][partName[MAPPER]]["depth_intrinsic"]["height"].asInt();
	COFLX = root[case_now][partName[MAPPER]]["color_intrinsic"]["fx"].asFloat();
	COFLY = root[case_now][partName[MAPPER]]["color_intrinsic"]["fy"].asFloat();
	COCPX = root[case_now][partName[MAPPER]]["color_intrinsic"]["cx"].asFloat();
	COCPY = root[case_now][partName[MAPPER]]["color_intrinsic"]["cy"].asFloat();
	COIMX = root[case_now][partName[MAPPER]]["color_intrinsic"]["width"].asInt();
	COIMY = root[case_now][partName[MAPPER]]["color_intrinsic"]["height"].asInt();
	for (int i = 0; i < 12; i++) {
		D_C_EXT[i] = root[case_now][partName[MAPPER]]["c_d_extrinsic"][i].asFloat();
	}
	DTEST = root[case_now][partName[MAPPER]]["depth_test"].asFloat();

	DIVIDER_LAYER = root[case_now][partName[SIMPLIFIER]]["div_per_layer"].asFloat();

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 16; j++) {
			rendering_spots_c[i][j] = root[case_now][partName[RENDERER]]["rendering_spots"][to_string(i) + "c"][j].asFloat();
			rendering_spots_o[i][j] = root[case_now][partName[RENDERER]]["rendering_spots"][to_string(i) + "o"][j].asFloat();
		}
	}
	shader_model = root[case_now][partName[RENDERER]]["shader_model"].asString();
	shader_atlas = root[case_now][partName[RENDERER]]["shader_atlas"].asString();
	shader_sub_atlas = root[case_now][partName[RENDERER]]["shader_sub_atlas"].asString();
	shader_mask = root[case_now][partName[RENDERER]]["shader_mask"].asString();

	RECORD_UNIT = root[case_now][partName[ANY]]["unit_test"].asBool();
	mesh_extension = root[case_now][partName[ANY]]["mesh_extension"].asString();
	normal_direction = root[case_now][partName[ANY]]["face_normal_clockwise"].asBool() ? -1 : 1;
}