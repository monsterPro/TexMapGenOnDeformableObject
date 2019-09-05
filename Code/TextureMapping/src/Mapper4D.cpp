#include "Mapper4D.h"
#include "Simplifier.h"

using namespace TexMap;

static Simplifier* simpleifier_ptr = NULL;

Mapper4D::Mapper4D(string templateMeshPath, string NR_MeshPathAndPrefix, string streamPath, int startIdx, int endIdx)
{
	m_templateMeshPath = templateMeshPath;
	m_NR_MeshPathAndPrefix = NR_MeshPathAndPrefix;
	m_streamPath = streamPath;

	template_mesh.request_vertex_texcoords2D();
	template_mesh.request_face_normals();
	template_mesh.request_vertex_normals();
	opt += OpenMesh::IO::Options::VertexTexCoord;
	if (!OpenMesh::IO::read_mesh(template_mesh, templateMeshPath, opt))
	{
		std::cerr << "read error\n";
		exit(1);
	}
	template_mesh.update_normals();

	_Vertices.resize(template_mesh.n_vertices());
	_Triangles.resize(template_mesh.n_faces());

	numVer = template_mesh.n_vertices();
	numTri = template_mesh.n_faces();

	_CO_IR = cv::Mat::eye(4, 4, CV_32F);
	for (int i = 0; i < 12; i++) {
		_CO_IR.at<float>((int)(i / 4), (int)(i % 4)) = D_C_EXT[i];
	}
	_IR_CO = _CO_IR.inv();

	SpatiotemporalSampleAndCalpos(streamPath, NR_MeshPathAndPrefix, startIdx, endIdx);
}

void Mapper4D::ConstructVertree_majorVote() {

	//// Generate Progressive mesh
	if (!simpleifier_ptr)
		delete simpleifier_ptr;
	simpleifier_ptr = new Simplifier(layer_mesh_vec);
	simpleifier_ptr->simplify_layer(LAYERNUM);

	propAccum.resize(imgNum);
	propValid.resize(imgNum);
	float2 dudvInit;
	dudvInit.x = 0.0;
	dudvInit.y = 0.0;
	for (int i = 0; i < imgNum; i++) {
		propAccum[i].resize(numVer, dudvInit);
		propValid[i].resize(numVer, false);
	}
	layer_Vertices.resize(LAYERNUM);
	layer_Triangles.resize(LAYERNUM);
	layer_numVer.resize(LAYERNUM);
	layer_numTri.resize(LAYERNUM);

	vector<vector<set<uint>>> ref_imgs_set_pre_ver;
	vector<vector<set<uint>>> ref_imgs_set_pre_tri;
	vector<vector<map<uint, float2>>> ref_imgs_set_ver;
	vector<vector<map<uint, float>>> ref_imgs_set_tri_ori;
	vector<vector<map<uint, float>>> ref_imgs_set_tri;

	ref_imgs_set_pre_ver.resize(LAYERNUM - 1);
	ref_imgs_set_pre_tri.resize(LAYERNUM - 1);
	ref_imgs_set_ver.resize(LAYERNUM);
	ref_imgs_set_tri_ori.resize(LAYERNUM);
	ref_imgs_set_tri.resize(LAYERNUM);

	// fine to coarse
	// l -> LAYERNUM (fine) ~ 0 (coarse)
	for (int l = LAYERNUM - 1; l > -1; l--) {
		unsigned int tmp_sim_numVer = simpleifier_ptr->extVer[LAYERNUM - l - 1].size();
		unsigned int tmp_sim_numTri = simpleifier_ptr->extTri[LAYERNUM - l - 1].size();
		if (l < LAYERNUM - 1) {
			ref_imgs_set_pre_ver[l].resize(tmp_sim_numVer);
			ref_imgs_set_pre_tri[l].resize(tmp_sim_numTri);
		}

		ref_imgs_set_ver[l].resize(tmp_sim_numVer);
		ref_imgs_set_tri_ori[l].resize(tmp_sim_numTri);
		ref_imgs_set_tri[l].resize(tmp_sim_numTri);
	}

	//reserve
	for (int l = LAYERNUM - 1; l > -1; l--) {
		unsigned int tmp_sim_numVer = simpleifier_ptr->extVer[LAYERNUM - l - 1].size();
		unsigned int tmp_sim_numTri = simpleifier_ptr->extTri[LAYERNUM - l - 1].size();
		lVertex *tmp_layer_Vertices = simpleifier_ptr->extVer[LAYERNUM - l - 1].data();
		lTriangle *tmp_layer_Triangles = simpleifier_ptr->extTri[LAYERNUM - l - 1].data();

		layer_Vertices[l].resize(tmp_sim_numVer);
		layer_Triangles[l].resize(tmp_sim_numTri);
		layer_numVer[l] = tmp_sim_numVer;
		layer_numTri[l] = tmp_sim_numTri;


		for (int i = 0; i < tmp_sim_numVer; i++) {
			layer_Vertices[l][i]._Triangles.assign(tmp_layer_Vertices[i]._Triangles.begin(), tmp_layer_Vertices[i]._Triangles.end());
		}
		for (int i = 0; i < tmp_sim_numTri; i++) {
			layer_Triangles[l][i]._Vertices[0] = tmp_layer_Triangles[i]._Vertices[0];
			layer_Triangles[l][i]._Vertices[1] = tmp_layer_Triangles[i]._Vertices[1];
			layer_Triangles[l][i]._Vertices[2] = tmp_layer_Triangles[i]._Vertices[2];
		}

		vector<vector<float3>> positions_vec;
		vector<vector<float3>> normals_vec;
		vector<vector<float2>> pixel_vec;
		vector<vector<float>> areas_vec;
		positions_vec.resize(imgNum);
		pixel_vec.resize(imgNum);
		normals_vec.resize(imgNum);
		areas_vec.resize(imgNum);

		for (int i = 0; i < imgNum; i++) {
			positions_vec[i].resize(tmp_sim_numVer);
			pixel_vec[i].resize(tmp_sim_numVer);
			normals_vec[i].resize(tmp_sim_numTri);
			areas_vec[i].resize(tmp_sim_numTri);
			cv::Mat _P_inv_IR = (_Pose.inv());
			cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);

			for (int vi = 0; vi < tmp_sim_numVer; vi++) {
				positions_vec[i][vi] = tmp_layer_Vertices[vi]._Pos[i];

				//////////////////pixel cal
				float3 _Point_CO;
				float3 _Point_IR;
				float2 _Pixel_CO;
				float2 _Pixel_IR;

				Transform(positions_vec[i][vi], _P_inv_CO, _Point_CO);
				PointToPixel_CO(_Point_CO, _Pixel_CO);
				pixel_vec[i][vi] = _Pixel_CO;
			}
			for (int fi = 0; fi < tmp_sim_numTri; fi++) {
				normals_vec[i][fi] = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Normal[i];
			}
			for (int fi = 0; fi < tmp_sim_numTri; fi++) {
				int fvi0 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[0];
				int fvi1 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[1];
				int fvi2 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[2];

				Vec3f v0; // = tmp_mesh_vec[i].point(fv_it++);
				Vec3f v1; // = tmp_mesh_vec[i].point(fv_it++);
				Vec3f v2; // = tmp_mesh_vec[i].point(fv_it);
				v0 = Vec3f(pixel_vec[i][fvi0].x, pixel_vec[i][fvi0].y, 1.0);
				v1 = Vec3f(pixel_vec[i][fvi1].x, pixel_vec[i][fvi1].y, 1.0);
				v2 = Vec3f(pixel_vec[i][fvi2].x, pixel_vec[i][fvi2].y, 1.0);

				Vec3d e1 = OpenMesh::vector_cast<Vec3d, Vec3f>(v1 - v0);
				Vec3d e2 = OpenMesh::vector_cast<Vec3d, Vec3f>(v2 - v0);

				Vec3d fN = OpenMesh::cross(e1, e2);
				double area = fN.norm() / 2.0;

				areas_vec[i][fi] = area;
			}
		}
		for (int vi = 0; vi < tmp_sim_numVer; vi++) {
			std::vector<mDummyImage> candidateImages;
			candidateImages.clear();
			for (int i = 0; i < imgNum; i++)
			{
				float3 _Point_CO;
				float3 _Point_IR;
				float2 _Pixel_CO;
				float2 _Pixel_IR;
				cv::Mat _P_inv_IR = (_Pose.inv());
				cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);
				
				Transform(positions_vec[i][vi], _P_inv_IR, _Point_IR);
				Transform(positions_vec[i][vi], _P_inv_CO, _Point_CO);
				PointToPixel_CO(_Point_CO, _Pixel_CO);
				PointToPixel_IR(_Point_IR, _Pixel_IR);


				if (_Pixel_IR.x < 0 || _Pixel_IR.x >= IRIMX || _Pixel_IR.y < 0 || _Pixel_IR.y >= IRIMY)
					continue;
				if ((_Pixel_IR.x) < 20.0f || (_Pixel_IR.x) > IRIMX - 20.0 || (_Pixel_IR.y) < 20.0f || (_Pixel_IR.y) > IRIMY - 20.0f)
				{

				}
				else
				{

					float _Depth = (float)depthImages[i].at<unsigned short>((int)_Pixel_IR.y, (int)_Pixel_IR.x) / 1000.0f;

					if (abs(_Depth - _Point_IR.z) < DTEST) {
						mDummyImage entry;
						entry._Img = i;
						entry._Img_Tex.x = _Pixel_CO.x;
						entry._Img_Tex.y = _Pixel_CO.y;
						entry._Weight = 1.0f;
						//float weight_max = 0.0;
						//bool is_valid = true;
						float weight = 0.0;
						float3 avgnormal = make_float3(0.0);
						for (auto vf_idx : layer_Vertices[l][vi]._Triangles) {
							avgnormal += normals_vec[i][vf_idx];
							//float _Dot = _Pose.at<float>(0, 2) * normals_vec[i][vf_idx].x + _Pose.at<float>(1, 2) * normals_vec[i][vf_idx].y + _Pose.at<float>(2, 2) * normals_vec[i][vf_idx].z;
							//weight_max = std::max(normal_direction * _Dot, weight_max);
						}
						avgnormal /= layer_Vertices[l][vi]._Triangles.size();
						float _Dot = _Pose.at<float>(0, 2) * avgnormal.x + _Pose.at<float>(1, 2) * avgnormal.y + _Pose.at<float>(2, 2) * avgnormal.z;
						if (normal_direction * _Dot > 0) {
							entry._Weight *= normal_direction * _Dot;
							candidateImages.push_back(entry);
						}

						//entry._Weight *= weight_max;
						//if (is_valid)
							//candidateImages.push_back(entry);
					}
				}
			}
			std::sort(candidateImages.begin(), candidateImages.end());
			for (auto &e : candidateImages) {
				ref_imgs_set_ver[l][vi].insert(make_pair(e._Img, e._Img_Tex));
			}

			if (l < LAYERNUM - 1) {
				for (auto vh : tmp_layer_Vertices[vi].verHistory) {
					int vs = simpleifier_ptr->idx_table_Ver[LAYERNUM - l - 2][vh];
					if (vs < 0)
						continue;
					for (auto v_im : layer_Vertices[l + 1][vs]._Img) {
						ref_imgs_set_pre_ver[l][vi].insert(v_im);
					}
				}
			}
		}

		//tri's visble img initialize
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			set<uint> tmp_v_imgs[3];
			for (int fv_i = 0; fv_i < 3; fv_i++) {
				for (auto im_idx : ref_imgs_set_ver[l][layer_Triangles[l][fi]._Vertices[fv_i]]) {
					tmp_v_imgs[fv_i].insert(im_idx.first);
				}
			}
			set<uint> inter_result, inter_result01;

			for (auto im_idx : tmp_v_imgs[0]) {
				if (tmp_v_imgs[1].find(im_idx) != tmp_v_imgs[1].end())
					inter_result01.insert(im_idx);
			}
			for (auto im_idx : inter_result01) {
				if (tmp_v_imgs[2].find(im_idx) != tmp_v_imgs[2].end())
					inter_result.insert(im_idx);
			}

			for (auto im_idx : inter_result) {
				float _Dot = _Pose.at<float>(0, 2) * normals_vec[im_idx][fi].x + _Pose.at<float>(1, 2) * normals_vec[im_idx][fi].y + _Pose.at<float>(2, 2) * normals_vec[im_idx][fi].z;
				if (_Dot * normal_direction > 0) {
					ref_imgs_set_tri[l][fi].insert(make_pair(im_idx, normal_direction * _Dot));
					ref_imgs_set_tri_ori[l][fi].insert(make_pair(im_idx, normal_direction * _Dot));
				}
			}

			// for coherency (fine to coarse)
			// union of finer level result
			if (l < LAYERNUM - 1) {
				set<uint> tmp_pre_v_imgs[3];
				for (int fv_i = 0; fv_i < 3; fv_i++) {
					for (auto i_idx : ref_imgs_set_pre_ver[l][layer_Triangles[l][fi]._Vertices[fv_i]]) {
						tmp_pre_v_imgs[fv_i].insert(i_idx);
					}
				}
				set<uint> pre_inter_result, pre_inter_result01;

				for (auto im_idx : tmp_pre_v_imgs[0]) {
					if (tmp_pre_v_imgs[1].find(im_idx) != tmp_pre_v_imgs[1].end())
						pre_inter_result01.insert(im_idx);
				}
				for (auto im_idx : pre_inter_result01) {
					if (tmp_pre_v_imgs[2].find(im_idx) != tmp_pre_v_imgs[2].end())
						pre_inter_result.insert(im_idx);
				}
				for (auto im_idx : pre_inter_result) {
					ref_imgs_set_pre_tri[l][fi].insert(im_idx);
				}
			}
		}
		
		if (RECORD_UNIT) {
			for (int tttt = 0; tttt < imgNum; tttt++) {
				cv::Mat tmp;
				cv::cvtColor(colorImages[tttt], tmp, CV_BGRA2BGR);
				for (int i = 0; i < tmp_sim_numTri; i++) {
					for (auto tmp_t_img : ref_imgs_set_tri_ori[l][i]) {
						if (tmp_t_img.first == tttt) {
							float2 uv1;
							float2 uv2;
							float2 uv3;
							if (ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[0]].find(tttt) == ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[0]].end())
								break;
							if (ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[1]].find(tttt) == ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[1]].end())
								break;
							if (ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[2]].find(tttt) == ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[2]].end())
								break;

							uv1 = ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[0]].find(tttt)->second;
							uv2 = ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[1]].find(tttt)->second;
							uv3 = ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[2]].find(tttt)->second;

							cv::line(tmp, cv::Point(uv1.x, uv1.y), cv::Point(uv2.x, uv2.y), cv::Scalar(255, 0, 0), 1);
							cv::line(tmp, cv::Point(uv2.x, uv2.y), cv::Point(uv3.x, uv3.y), cv::Scalar(255, 0, 0), 1);
							cv::line(tmp, cv::Point(uv3.x, uv3.y), cv::Point(uv1.x, uv1.y), cv::Scalar(255, 0, 0), 1);
						}
					}
				}
				cv::imwrite(std::string(data_root_path + "/unit_test/" + unit_test_path + "/projection/" + std::to_string(l)) + "/sampled_Frame_" + std::to_string(tttt) + std::string(".png"), tmp);
			}
		}
		
		//tri's 1 ring neighbor tri initialize
		vector<set<uint>> union_tri;
		union_tri.resize(tmp_sim_numTri);
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			for (int fv_i = 0; fv_i < 3; fv_i++) {
				for (auto fvf_idx : layer_Vertices[l][layer_Triangles[l][fi]._Vertices[fv_i]]._Triangles)
					union_tri[fi].insert(fvf_idx);
			}
		}

		//prune candidate img layer by layer
		bool enough_prune = true;
		uint no_update_count = 0;
		int around_score_thresh = MAXIMG;
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			if (ref_imgs_set_tri[l][fi].size() > MAXIMG) {
				enough_prune = false;
				break;
			}
		}
		if (enough_prune)
			cout << "original mapping is pruned enough" << endl;
		else
			cout << "prune start : " << l << endl;
		
		while (!enough_prune) {
			enough_prune = true;
			map<uint, float> vote_dummy;
			map<uint, float> vote_dummy_out;
			vector<int> prune_target;
			vector<int> recon_target;
			map<float, uint> recon_order;
			prune_target.resize(tmp_sim_numTri, -1);
			recon_target.resize(tmp_sim_numTri, -1);

			//find pruning target per tri
			for (int fi = 0; fi < tmp_sim_numTri; fi++) {
				// enoughly pruned tri
				if (ref_imgs_set_tri[l][fi].size() < MAXIMG)
					continue;

				for (auto tmp_ref_imgs : ref_imgs_set_tri[l][fi]) {
					if (l < LAYERNUM - 1 && ref_imgs_set_pre_tri[l][fi].find(tmp_ref_imgs.first) != ref_imgs_set_pre_tri[l][fi].end())
						vote_dummy.insert(make_pair(tmp_ref_imgs.first, union_tri[fi].size()));
					else
						vote_dummy.insert(make_pair(tmp_ref_imgs.first, 0.0));
				}


				// voting img index (1 ring tri)
				// weight (bucket sorting)
				//			1) pre layer
				//			2) voting
				//			3) img weight
				for (auto ff_idx : union_tri[fi]) {
					float ff_imgNum = ref_imgs_set_tri[l][ff_idx].size();
					for (auto tt_im_idx : ref_imgs_set_tri[l][ff_idx]) {
						auto dummy_iter = vote_dummy.find(tt_im_idx.first);
						if (dummy_iter != vote_dummy.end())
							dummy_iter->second += (1.0 + tt_im_idx.second / union_tri[fi].size()) / union_tri[fi].size();
						else {
							auto dummy_iter_out = vote_dummy_out.find(tt_im_idx.first);
							if (dummy_iter_out != vote_dummy_out.end())
								dummy_iter_out->second += (1.0 + tt_im_idx.second / union_tri[fi].size()) / union_tri[fi].size();
							else
								vote_dummy_out.insert(make_pair(tt_im_idx.first, (1.0 + tt_im_idx.second / union_tri[fi].size()) / union_tri[fi].size()));
						}
					}
				}

				int minElementIndex = -1;
				int maxElementIndex = -1;
				float min_vote_weight = 100000;
				float max_vote_weight = 0;
				for (auto vote_result : vote_dummy) {
					if (vote_result.second < min_vote_weight) {
						minElementIndex = vote_result.first;
						min_vote_weight = vote_result.second;
					}
				}
				prune_target[fi] = minElementIndex;

				//for recon (not use)
				for (auto vote_result : vote_dummy_out) {
					if (vote_result.second > max_vote_weight && ref_imgs_set_tri_ori[l][fi].find(vote_result.first) != ref_imgs_set_tri_ori[l][fi].end()) {
						if (vote_result.second < 1.0 / 3.0)
							continue;
						/*if (vote_result.second < (float)union_tri[fi].size() / 4.0)
						continue;*/
						maxElementIndex = vote_result.first;
						max_vote_weight = vote_result.second;
					}
				}
				recon_target[fi] = maxElementIndex;
				recon_order.insert(make_pair(max_vote_weight, fi));

				vote_dummy.clear();
			}

			//prune
			for (int fi = 0; fi < tmp_sim_numTri; fi++) {
				if(prune_target[fi] > -1)
					ref_imgs_set_tri[l][fi].erase(prune_target[fi]);
			}
			//recon
			/*for (auto t : recon_order) {
			int fi = t.second;
			if (recon_target[fi] < 0)
			continue;

			bool enough_this_tri = true;
			for (auto v : layer_Triangles[l][fi]._Vertices) {
			set<uint> one_ring_img;
			for (int vt_i = 0; vt_i < layer_Vertices[l][v]._Triangles_Num; vt_i++) {
			for (auto vt_im_idx : ref_imgs_set_tri[l][layer_Vertices[l][v]._Triangles[vt_i]])
			one_ring_img.insert(vt_im_idx.first);
			}
			if (one_ring_img.size() > MAXIMG - 1) {
			enough_this_tri = false;
			break;
			}
			}
			if (enough_this_tri) {
			cout << "I'm working on!!" << endl;
			ref_imgs_set_tri[l][fi].insert(make_pair(recon_target[fi], ref_imgs_set_tri_ori[l][fi].find(recon_target[fi])->second));
			}
			}*/
			/*for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			if (recon_target[fi] < 0)
			continue;
			ref_imgs_set_tri[l][fi].insert(make_pair(recon_target[fi], ref_imgs_set_tri[l][fi].find(recon_target[fi])->second));
			}*/
			//for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			//	//if (prune_target[fi] < 0 || ref_imgs_set_tri[l][fi].size() > MAXIMG - MAXTRI)
			//	//if (prune_target[fi] < 0 || ref_imgs_set_tri[l][fi].size() > max(4, MAXIMG - layer_Vertices[l][layer_Triangles[l][fi]._Vertices[0]]._Triangles_Num))
			//	//if (prune_target[fi] < 0 || ref_imgs_set_tri[l][fi].size() > max(4, MAXIMG - union_tri[fi].size()))
			//	//if (recon_target[fi] < 0 || ref_imgs_set_tri[l][fi].size() > 4)
			//	if (recon_target[fi] < 0 || ref_imgs_set_tri[l][fi].size() > max(3, MAXIMG / layer_Vertices[l][layer_Triangles[l][fi]._Vertices[0]]._Triangles_Num))
			//		continue;
			//	ref_imgs_set_tri[l][fi].insert(make_pair(recon_target[fi], ref_imgs_set_tri[l][fi].find(recon_target[fi])->second));
			//}

			//if face's img > MAXIMG, more prune
			uint over_count = 0;
			for (int fi = 0; fi < tmp_sim_numTri; fi++) {
				if (ref_imgs_set_tri[l][fi].size() > MAXIMG) {
					enough_prune = false;
					over_count++;
				}
			}
			printProgBar((1.0 - (float)over_count / (float)tmp_sim_numTri) * 100.0);
			prune_target.clear();
			recon_target.clear();
		}
		cout << endl;

		// assign values
		for (int vi = 0; vi < tmp_sim_numVer; vi++) {
			set<uint> one_ring_img;
			for (auto vf_idx : layer_Vertices[l][vi]._Triangles) {
				for (auto im_idx : ref_imgs_set_tri[l][vf_idx])
					one_ring_img.insert(im_idx.first);
			}
			for (auto im_idx : one_ring_img) {
				auto e = ref_imgs_set_ver[l][vi].find(im_idx);
				layer_Vertices[l][vi]._Img.push_back(e->first);
				layer_Vertices[l][vi]._Img_Tex.push_back(e->second);
			}
		}
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			for (auto im_idx : ref_imgs_set_tri[l][fi]) {
				layer_Triangles[l][fi]._Img.push_back(im_idx.first);
				layer_Triangles[l][fi]._Img_Weight.push_back(im_idx.second);
			}
		}

		// weight normalization
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			float wsum = 0;
			for (auto im_weight : layer_Triangles[l][fi]._Img_Weight) wsum += im_weight;
			for (int wi = 0; wi < layer_Triangles[l][fi]._Img_Weight.size(); wi++)
				layer_Triangles[l][fi]._Img_Weight[wi] /= wsum;
		}

		positions_vec.clear();
		normals_vec.clear();
		pixel_vec.clear();
		areas_vec.clear();
	}
	cout << "Convert tree Done" << endl;
}

void Mapper4D::SetPropVec(vector<vector<float2>> propVec, int layer) {

	vector<int> parentNode(simpleifier_ptr->extVer[LAYERNUM - 1 - layer].size() , -1);

	for (int i = 0; i < simpleifier_ptr->extVer[LAYERNUM - layer].size(); i++) {
		int v_Img_Num = layer_Vertices[layer - 1][i]._Img.size();
		for (auto vh : simpleifier_ptr->extVer[LAYERNUM - layer][i].verHistory) {
			for (int j = 0; j < v_Img_Num; j++) {
				int k = layer_Vertices[layer-1][i]._Img[j];
				propAccum[k][vh] += propVec[k][i];
				propValid[k][vh] = true;
			}

			//link parent node
			int tmp_idx = simpleifier_ptr->idx_table_Ver[LAYERNUM - 1 - layer][vh];
			if (tmp_idx >= 0)
				parentNode[tmp_idx] = i;
		}
	}
	for (int i = 0; i < numVer; i++) {
		int tmp_idx = simpleifier_ptr->idx_table_Ver[LAYERNUM - 1 - layer][i];
		if (tmp_idx < 0)
			continue;
		
		int v_Img_Num = layer_Vertices[layer][tmp_idx]._Img.size();

		set<unsigned int> ringv_idx;
		//(v1 average)->(v2 weighted sum) of parent 1 ring node
		for (int ti = 0; ti < layer_Vertices[layer-1][parentNode[tmp_idx]]._Triangles.size(); ti++) {
			unsigned int ringTri = layer_Vertices[layer-1][parentNode[tmp_idx]]._Triangles[ti];
			for (auto rv : layer_Triangles[layer-1][ringTri]._Vertices) {
				ringv_idx.insert(simpleifier_ptr->idx_table_Ver_inv[LAYERNUM - layer][rv]);
			}
		}
		
		for (int j = 0; j < v_Img_Num; j++) {
			int k = layer_Vertices[layer][tmp_idx]._Img[j];
			float2 meanProp;
			meanProp.x = 0; meanProp.y = 0;
			int n_move = 0;
			float weight_move = 0.0;
			for (auto rv : ringv_idx) {
				int parentLocalrv = simpleifier_ptr->idx_table_Ver[LAYERNUM - layer][rv];
				if (parentLocalrv < 0)
					continue;
				int parentImgIdx = layer_Vertices[layer - 1][parentLocalrv].getImgIdx(k);
				if (propValid[k][rv] && parentImgIdx >= 0) {
					float weight = 1.0 / exp(abs(Distance(layer_Vertices[layer][tmp_idx]._Img_Tex[j], layer_Vertices[layer - 1][parentLocalrv]._Img_Tex[parentImgIdx])));
					meanProp += propAccum[k][rv] * weight;
					weight_move += weight;
					n_move++;
				}
			}
			if (weight_move > 0 && n_move > 0)
				meanProp /= weight_move;
			layer_Vertices[layer][tmp_idx]._Img_Tex[j] += meanProp;
			layer_Vertices[layer][tmp_idx]._Img_Tex[j].x = clamp(layer_Vertices[layer][tmp_idx]._Img_Tex[j].x, (float)0, (float)COIMX - 1);
			layer_Vertices[layer][tmp_idx]._Img_Tex[j].y = clamp(layer_Vertices[layer][tmp_idx]._Img_Tex[j].y, (float)0, (float)COIMY - 1);
		}
	}
}

void Mapper4D::SpatiotemporalSampleAndCalpos(string streamPath, string NR_MeshPathAndPrefix, int startIdx, int endIdx)
{
	vector<int> tempIdx;
	
	vector<cv::Mat> colorImages_temporal;
	vector<cv::Mat> depthImages_temporal;
	vector<float> blurlist_temporal;

	temporalframeSample(streamPath, colorImages_temporal, depthImages_temporal, blurlist_temporal, tempIdx, startIdx, endIdx);


	unsigned int m_DepthImageWidth = depthImages_temporal[0].cols;
	unsigned int m_DepthImageHeight = depthImages_temporal[0].rows;
	unsigned int m_ColorImageWidth = colorImages_temporal[0].cols;
	unsigned int m_ColorImageHeight = colorImages_temporal[0].rows;

	imgNum = (unsigned int)tempIdx.size();

	for (MyMesh::VertexIter v_it = template_mesh.vertices_begin(); v_it != template_mesh.vertices_end(); ++v_it) {
		_Vertices[v_it->idx()]._Pos.x = template_mesh.point(*v_it)[0];
		_Vertices[v_it->idx()]._Pos.y = template_mesh.point(*v_it)[1];
		_Vertices[v_it->idx()]._Pos.z = template_mesh.point(*v_it)[2];

		for (MyMesh::VertexFaceIter vf_it = template_mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
			_Vertices[v_it->idx()]._Triangles.push_back(vf_it->idx());
		}
	}
	for (MyMesh::FaceIter f_it = template_mesh.faces_begin(); f_it != template_mesh.faces_end(); ++f_it) {
		int vi = 0;
		for (MyMesh::FaceVertexIter fv_it = template_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
			if (vi > 2) {
				printf("too many vertex");
				continue;
			}
			_Triangles[f_it->idx()]._Vertices[vi] = fv_it->idx();
			vi++;
		}

		_Triangles[f_it->idx()]._Normal.x = template_mesh.normal(*f_it)[0];
		_Triangles[f_it->idx()]._Normal.y = template_mesh.normal(*f_it)[1];
		_Triangles[f_it->idx()]._Normal.z = template_mesh.normal(*f_it)[2];

	}

	opt -= OpenMesh::IO::Options::VertexTexCoord;
	vector<MyMesh> tmp_mesh_vec;
	vector<vector<float3>> positions_vec;
	vector<vector<float3>> normals_vec;
	vector<vector<float2>> pixel_vec;
	vector<vector<float>> areas_vec;
	tmp_mesh_vec.reserve(imgNum);
	tmp_mesh_vec.resize(imgNum);
	positions_vec.reserve(imgNum);
	positions_vec.resize(imgNum);
	pixel_vec.reserve(imgNum);
	pixel_vec.resize(imgNum);
	normals_vec.reserve(imgNum);
	normals_vec.resize(imgNum);
	areas_vec.reserve(imgNum);
	areas_vec.resize(imgNum);

	for (int i = 0; i < imgNum; i++) {
		tmp_mesh_vec[i].request_face_normals();
		tmp_mesh_vec[i].request_vertex_normals();

		string filename;
		filename = NR_MeshPathAndPrefix + "_" + zeroPadding(to_string(tempIdx[i]), 3) + "." + mesh_extension;
		if (!OpenMesh::IO::read_mesh(tmp_mesh_vec[i], filename, opt))
		{
			std::cerr << "read error\n";
			exit(1);
		}
		tmp_mesh_vec[i].update_normals();
		positions_vec[i].reserve(numVer);
		positions_vec[i].resize(numVer);
		pixel_vec[i].reserve(numVer);
		pixel_vec[i].resize(numVer);
		normals_vec[i].reserve(numTri);
		normals_vec[i].resize(numTri);
		areas_vec[i].reserve(numTri);
		areas_vec[i].resize(numTri);

		cv::Mat _P_inv_IR = (_Pose.inv());
		cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);

		for (MyMesh::VertexIter v_it = tmp_mesh_vec[i].vertices_begin(); v_it != tmp_mesh_vec[i].vertices_end(); ++v_it) {
			positions_vec[i][v_it->idx()].x = tmp_mesh_vec[i].point(*v_it)[0];
			positions_vec[i][v_it->idx()].y = tmp_mesh_vec[i].point(*v_it)[1];
			positions_vec[i][v_it->idx()].z = tmp_mesh_vec[i].point(*v_it)[2];

			//////////////////pixel cal
			float3 _Point_CO;
			float3 _Point_IR;
			float2 _Pixel_CO;
			float2 _Pixel_IR;


			/*Transform(_Vertices[i]._Pos, _P_inv_IR, _Point_IR);
			Transform(_Vertices[i]._Pos, _P_inv_CO, _Point_CO);*/
			

			Transform(positions_vec[i][v_it->idx()], _P_inv_CO, _Point_CO);
			PointToPixel_CO(_Point_CO, _Pixel_CO);


			pixel_vec[i][v_it->idx()] = _Pixel_CO;
		}
		
		for (MyMesh::FaceIter f_it = tmp_mesh_vec[i].faces_begin(); f_it != tmp_mesh_vec[i].faces_end(); ++f_it) {
			normals_vec[i][f_it->idx()].x = tmp_mesh_vec[i].normal(*f_it)[0];
			normals_vec[i][f_it->idx()].y = tmp_mesh_vec[i].normal(*f_it)[1];
			normals_vec[i][f_it->idx()].z = tmp_mesh_vec[i].normal(*f_it)[2];
		}
		for (MyMesh::FaceIter f_it = tmp_mesh_vec[i].faces_begin(); f_it != tmp_mesh_vec[i].faces_end(); ++f_it) {
			MyMesh::FaceVertexIter fv_it = template_mesh.fv_iter(*f_it);

			Vec3f v0; // = tmp_mesh_vec[i].point(fv_it++);
			Vec3f v1; // = tmp_mesh_vec[i].point(fv_it++);
			Vec3f v2; // = tmp_mesh_vec[i].point(fv_it);
			v0 = Vec3f(pixel_vec[i][fv_it->idx()].x, pixel_vec[i][fv_it->idx()].y, 1.0);
			fv_it++;
			v1 = Vec3f(pixel_vec[i][fv_it->idx()].x, pixel_vec[i][fv_it->idx()].y, 1.0);
			fv_it++;
			v2 = Vec3f(pixel_vec[i][fv_it->idx()].x, pixel_vec[i][fv_it->idx()].y, 1.0);

			Vec3d e1 = OpenMesh::vector_cast<Vec3d, Vec3f>(v1 - v0);
			Vec3d e2 = OpenMesh::vector_cast<Vec3d, Vec3f>(v2 - v0);

			Vec3d fN = OpenMesh::cross(e1, e2);
			double area = fN.norm() / 2.0;

			areas_vec[i][f_it->idx()] = area;
		}
	}


	vector<size_t> accumVisTri(numTri, 0);
	vector<size_t> accumsampledTri(numTri, 0);
	vector<map<size_t, float>> visTri(imgNum);
	vector<pair<float, size_t>> uniq_vec;
	vector<size_t> finimgIdx;


	for (int k = 0; k < imgNum; k++)
	{
		cv::Mat _P_inv_IR = (_Pose.inv());
		cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);
		for (MyMesh::FaceIter f_it = template_mesh.faces_begin(); f_it != template_mesh.faces_end(); ++f_it) {
			bool _valid = true;
			float _weight = 0;
			for (MyMesh::FaceVertexIter fv_it = template_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
				float3 _Point_CO;
				float3 _Point_IR;
				float2 _Pixel_CO;
				float2 _Pixel_IR;

				Transform(positions_vec[k][fv_it->idx()], _P_inv_IR, _Point_IR);
				Transform(positions_vec[k][fv_it->idx()], _P_inv_CO, _Point_CO);
				PointToPixel_CO(_Point_CO, _Pixel_CO);
				PointToPixel_IR(_Point_IR, _Pixel_IR);
				if ((_Pixel_IR.x) < 20.0f || (_Pixel_IR.x) > IRIMX - 40 || (_Pixel_IR.y) < 20.0f || (_Pixel_IR.y) > IRIMY - 20.0f)
				{
					_valid = false;
					break;
				}
				else
				{

					float _Depth = (float)depthImages_temporal[k].at<unsigned short>((int)_Pixel_IR.y, (int)_Pixel_IR.x) / 1000.0f;

					if (abs(_Depth - _Point_IR.z) < DTEST) {

						float tmp_Weight = 1.0;
						float weight_max = 0.0;
						float3 avgnormal = make_float3(0.0);
						for (auto fvf_idx : _Vertices[fv_it->idx()]._Triangles) {
							avgnormal += normals_vec[k][fvf_idx];
							/*float _Dot = _Pose.at<float>(0, 2) * normals_vec[k][fvf_idx].x + _Pose.at<float>(1, 2) * normals_vec[k][fvf_idx].y + _Pose.at<float>(2, 2) * normals_vec[k][fvf_idx].z;
							if (normal_direction * _Dot < 0)
								_valid = false;*/
							//weight_max = std::max(normal_direction * _Dot, weight_max);
						}
						avgnormal /= _Vertices[fv_it->idx()]._Triangles.size();
						float _Dot = _Pose.at<float>(0, 2) * avgnormal.x + _Pose.at<float>(1, 2) *avgnormal.y + _Pose.at<float>(2, 2) * avgnormal.z;
						if (normal_direction * _Dot < 0)
							_valid = false;
						weight_max = normal_direction * _Dot;
						tmp_Weight *= weight_max;
						if (_weight < tmp_Weight)
							_weight = tmp_Weight;
					}
					else {
						_valid = false;
						break;
					}
				}
			}

			if (_valid) {
				accumVisTri[f_it->idx()]++;
				visTri[k].insert(make_pair(f_it->idx(), _weight));
			}
		}
	}
	for (int k = 0; k < imgNum; k++)
	{
		float uniqueness = 0;
		for (auto visIter = visTri[k].begin(); visIter != visTri[k].end(); visIter++) {
			uniqueness += visIter->second / accumVisTri[visIter->first];
		}
		uniqueness /= visTri[k].size();
		uniq_vec.push_back(make_pair(uniqueness, k));
	}
	sort(uniq_vec.begin(), uniq_vec.end());
	bool isDel = false;
	for (int k = imgNum - 1; k > -1; k--)
	{
		isDel = false;
		for (auto tmp_check : finimgIdx) {
			if (abs((int)uniq_vec[k].second - (int)tmp_check) < (endIdx - startIdx)/40) {
				isDel = true;
				break;
			}
		}
		for (auto visIter = visTri[uniq_vec[k].second].begin(); visIter != visTri[uniq_vec[k].second].end(); visIter++) {
			if (accumVisTri[visIter->first] < 3 && accumsampledTri[visIter->first] < 3) {
				isDel = false;
				break;
			}
		}
		if (isDel) {
			for (auto visIter = visTri[uniq_vec[k].second].begin(); visIter != visTri[uniq_vec[k].second].end(); visIter++) {
				accumVisTri[visIter->first]--;
			}
		}
		else {
			finimgIdx.push_back(uniq_vec[k].second);
			for (auto visIter = visTri[uniq_vec[k].second].begin(); visIter != visTri[uniq_vec[k].second].end(); visIter++)
				accumsampledTri[visIter->first]++;
		}
	}

	size_t finimgNum = finimgIdx.size();
	sort(finimgIdx.begin(), finimgIdx.end());
	std::cout << finimgNum << " frames extracted in spatial sampling" << std::endl;
	imgNum = finimgNum;
	for (int k = 0; k < finimgNum; k++)
	{
		colorImages.push_back(colorImages_temporal[finimgIdx[k]].clone());
		depthImages.push_back(depthImages_temporal[finimgIdx[k]].clone());
		c_dImages.push_back(c_dImages_temporal[finimgIdx[k]].clone());
		layer_mesh_vec.push_back(tmp_mesh_vec[finimgIdx[k]]);
	}
	TransferColor(colorImages, c_dImages);
	std::cout << "Color transfer done" << std::endl;
	
	// 손볼것
	/*if (RECORD_UNIT) {
		std::ofstream outStream(streamPath + "/sampled_stream.bin", std::ofstream::binary);
		for (int i = 0; i < finimgNum; i++) {
			cv::Mat4b cImg;
			cv::cvtColor(colorImages[i], cImg, CV_RGB2BGRA);
			outStream.write((char*)cImg.data, sizeof(uchar) * 4 * m_ColorImageWidth * m_ColorImageHeight);
			outStream.write((char*)depthImages[i].data, sizeof(ushort) * m_DepthImageWidth * m_DepthImageHeight);
			outStream.write((char*)_Pose.data, sizeof(float) * 16);
		}
		outStream.close();
		std::cout << streamPath + "/sampled_stream.bin" << " Saved" << endl;
	}*/
}

void Mapper4D::temporalframeSample(string streamPath, vector<cv::Mat> &outColor, vector<cv::Mat> &outDepth, vector<float> &outBlur, vector<int> &outIdx, int startIdx, int endIdx) {

	int imgE_S = endIdx - startIdx;

	char frameNameBuffer[512] = { 0 };
	sprintf(frameNameBuffer, "/renderedDepth/Frame_%06d.png", 0);
	//sprintf(frameNameBuffer, "/depth/Frame_%06d.png", 0);
	cv::Mat depthtemp = cv::imread(streamPath + frameNameBuffer, CV_LOAD_IMAGE_UNCHANGED);
	//sprintf(frameNameBuffer, "/color/Frame_%06d.png", 0);
	sprintf(frameNameBuffer, "/filteredColor/Frame_%06d.png", 0);
	cv::Mat colortemp = cv::imread(streamPath + frameNameBuffer);
	unsigned int m_DepthImageWidth = depthtemp.cols;
	unsigned int m_DepthImageHeight = depthtemp.rows;
	unsigned int m_ColorImageWidth = colortemp.cols;
	unsigned int m_ColorImageHeight = colortemp.rows;

	cout << "Reading Stream..." << endl;

	std::vector<float> blurnessList;


	cv::Mat _P_inv_IR = (_Pose.inv());
	cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);

	cv::Scalar colorMean(0, 0, 0);
	cv::Scalar colorStd(0, 0, 0);
	vector<cv::Mat> cc_vec;

	for (int frameIdx = startIdx; frameIdx < endIdx; ++frameIdx)
	{
		printProgBar(((float)frameIdx / (endIdx - 1.0)) * 100);
		//sprintf(frameNameBuffer, "/depth/Frame_%06d.png", 3230 - frameIdx);
		//sprintf(frameNameBuffer, "/depth/Frame_%06d.png", frameIdx);
		sprintf(frameNameBuffer, "/renderedDepth/Frame_%06d.png", frameIdx);
		//sprintf(frameNameBuffer, "/depth/Frame_%06d.png", frameIdx);
		cv::Mat depthImage = cv::imread(streamPath + frameNameBuffer, CV_LOAD_IMAGE_ANYDEPTH);
		depthImages_dummy.push_back(depthImage);
		//sprintf(frameNameBuffer, "/color/Frame_%06d.png", 3230 - frameIdx);
		//sprintf(frameNameBuffer, "/color/Frame_%06d.png", frameIdx);
		sprintf(frameNameBuffer, "/filteredColor/Frame_%06d.png", frameIdx);
		//sprintf(frameNameBuffer, "/color/Frame_%03d.png", frameIdx);
		cv::Mat colorImage = cv::imread(streamPath + frameNameBuffer);
		if (colorImage.channels() == 4)
			cv::cvtColor(colorImage, colorImage, CV_BGRA2BGR);
		colorImages_dummy.push_back(colorImage);

		cv::Mat cc = cv::Mat(depthImage.size(), CV_8UC3, cv::Vec3b::all(0));
		cv::Mat dd;
		cv::inRange(depthImage, 1000, 3000, dd);
		cv::Mat mask = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2), cv::Point(1, 1));
		//cv::erode(src, dst, mask, cv::Point(-1, -1), 3);
		//cv::threshold(dd, dd, 1, 255, cv::THRESH_BINARY);
		cv::dilate(dd, dd, mask, cv::Point(-1, -1), 3);
		//cv::erode(dd, dd, mask, cv::Point(-1, -1), 3);
		/*cv::imshow("111", dd);
		cv::waitKey(0);*/
		//depthImages.push_back(dd);
		int count = 0;
		for (int y = 0; y < m_DepthImageHeight; y += 1)
			for (int x = 0; x < m_DepthImageWidth; x += 1) {
				if (dd.at<uchar>(y, x) == 0)
					continue;

				float3 ir_pixel = { x, y, (float)depthImage.at<unsigned short>(y, x) / 1000.0f };
				float3 ir_point;
				float3 _Point_CO;
				float2 _Pixel_CO;
				PixelToPoint_IR(ir_pixel, ir_point);
				Transform(ir_point, _P_inv_CO, _Point_CO);
				PointToPixel_CO(_Point_CO, _Pixel_CO);

				int clr_u, clr_v;
				clr_u = (int)(_Pixel_CO.x + 0.5);
				clr_v = (int)(_Pixel_CO.y + 0.5);

				if (clr_u < 0 || clr_u > colorImage.cols - 1 || clr_v < 0 || clr_v > colorImage.rows - 1)
					continue;
				//cout << clr_v << "            " << clr_u << endl;
				cc.at<cv::Vec3b>(y, x) = colorImage.at<cv::Vec3b>(clr_v, clr_u);
				count++;
			}

		/*cv::imshow("11", cc);
		cv::waitKey(1);*/
		float blurness = CalcBlurMetric(cc, count);
		//float blurness = CalcBlurMetric(colorImage);
		blurnessList.push_back(blurness);

		cc_vec.push_back(cc);
		/*cv::Mat3f cImg_lab;
		cc.convertTo(cImg_lab, CV_32FC3, 1.0 / 255.0f);
		cv::cvtColor(cImg_lab, cImg_lab, CV_BGR2Lab);
		cv::Scalar cImg_mean, cImg_std;
		cv::meanStdDev(cImg_lab, cImg_mean, cImg_std);
		cc_vec.push_back(cImg_lab);
		colorMean += cImg_mean;
		colorStd += cImg_std;*/
	}
	//////////////////////colortransfer
	/*colorMean[0] /= colorImages_dummy.size();
	colorMean[1] /= colorImages_dummy.size();
	colorMean[2] /= colorImages_dummy.size();
	colorStd[0] /= colorImages_dummy.size();
	colorStd[1] /= colorImages_dummy.size();
	colorStd[2] /= colorImages_dummy.size();
	for (int i = 0; i < colorImages_dummy.size(); i++) {
		TransferColor(colorImages_dummy[i], cc_vec[i],colorMean, colorStd);
	}*/
	//////////////////////colortransfer
	/*cv::Scalar colorMean(0, 0, 0);
	cv::Scalar colorStd(0, 0, 0);
	for (int i = 0; i < colorImages_dummy.size(); i++) {
		cv::Mat cImg = colorImages_dummy[i];
		cv::Mat3f cImgf;
		cImg.convertTo(cImgf, CV_32FC3, 1.0 / 255.0f);
		cv::Mat3f cImg_lab;
		cv::cvtColor(cImgf, cImg_lab, CV_BGR2Lab);
		cv::Scalar cImg_mean, cImg_std;
		cv::meanStdDev(cImg_lab, cImg_mean, cImg_std);
		colorMean += cImg_mean;
		colorStd += cImg_std;
	}
	colorMean[0] /= colorImages_dummy.size();
	colorMean[1] /= colorImages_dummy.size();
	colorMean[2] /= colorImages_dummy.size();
	colorStd[0] /= colorImages_dummy.size();
	colorStd[1] /= colorImages_dummy.size();
	colorStd[2] /= colorImages_dummy.size();
	for (int i = 0; i < colorImages_dummy.size(); i++) {
		TransferColor(colorImages_dummy[i], colorMean, colorStd);
	}*/
	
	std::cout << "\nCalculation finished" << std::endl;

	int searchIdx = 0;
	int maxRange = 0;
	//int maxRange = 20;
	std::cout << std::endl << "Start extracting keyframes & graph optimization" << std::endl;


	while (searchIdx + maxRange < imgE_S)
	{
		float minBlurness = 1.0f;
		int minIdx = 0;
		for (int i = searchIdx; i < searchIdx + maxRange; i++)
		{
			if (minBlurness > blurnessList[i])
			{
				minIdx = i;
				minBlurness = blurnessList[i];
			}
		}
		minIdx = searchIdx;
		outColor.push_back(colorImages_dummy[minIdx].clone());
		outDepth.push_back(depthImages_dummy[minIdx].clone());
		outBlur.push_back(blurnessList[minIdx]);
		c_dImages_temporal.push_back(cc_vec[minIdx].clone());
		outIdx.push_back(minIdx);
		searchIdx = minIdx + 1;
	}


	/*for (int frameIdx = startIdx; frameIdx < endIdx; frameIdx+=30)
	{
	outColor.push_back(colorImages_dummy[frameIdx].clone());
	outDepth.push_back(depthImages_dummy[frameIdx].clone());
	outIdx.push_back(frameIdx);

	}*/

	std::cout << outColor.size() << " frames extracted in temporal sampling" << std::endl;
}

void Mapper4D::Get_VT_layer(vector<hostVertex> &hVs, vector<hostTriangle> &hTs, int layer) {
	hVs.clear();
	hTs.clear();
	hVs.assign(layer_Vertices[layer].begin(), layer_Vertices[layer].end());
	hTs.assign(layer_Triangles[layer].begin(), layer_Triangles[layer].end());
}

void Mapper4D::GetNumInfo(int *nV, int *nT, int *nI)
{
	*nV = template_mesh.n_vertices();
	*nT = template_mesh.n_faces();
	*nI = imgNum;
}

void Mapper4D::GetNumInfo_layer(int *nV, int *nT, int layer)
{
	*nV = layer_numVer[layer];
	*nT = layer_numTri[layer];
}

//손볼것
/*void Mapper4D::SaveResult(string outDir, string FileName)
{
	int _Size_V = template_mesh.n_vertices();
	int _Size_F = template_mesh.n_faces();
	//For Optimization
	//std::ofstream fout(outDir + "/" + FileName + "_opt.vat", std::ofstream::binary);
	//fout.write((char*)&_Size_V, sizeof(int));
	//fout.write((char*)&_Size_F, sizeof(int));
	//fout.write((char*)&imgNum, sizeof(int));
	//fout.write((char*)_Vertices, _Size_V * sizeof(mVertex));
	//fout.write((char*)_Triangles, _Size_F * sizeof(mTriangle));
	//fout.close();

	//For Rendering
	std::ofstream fout_s(outDir + "/" + FileName + "_nopt.vat", std::ofstream::binary);
	fout_s.write((char*)&_Size_V, sizeof(int));
	fout_s.write((char*)&_Size_F, sizeof(int));
	fout_s.write((char*)&imgNum, sizeof(int));

	for (int i = 0; i < _Size_V; i++) mWriteVertex(_Vertices[i], fout_s);
	for (int i = 0; i < _Size_F; i++) mWriteTriangle(_Triangles[i], fout_s);

	fout_s.close();
}
void Mapper4D::ShowResult() {
	for (int tttt = 0; tttt < imgNum; tttt++) {
		for (int i = 0; i < numTri; i++) {
			for (int j = 0; j < _Triangles[i]._Img_Num; j++) {
				if (_Triangles[i]._Img[j] == tttt) {
					float2 uv1;
					float2 uv2;
					float2 uv3;

					for (int k = 0; k < _Vertices[_Triangles[i]._Vertices[0]]._Img_Num; k++)
						if (_Vertices[_Triangles[i]._Vertices[0]]._Img[k] == tttt) {
							uv1 = _Vertices[_Triangles[i]._Vertices[0]]._Img_Tex[k]; break;
						}
					for (int k = 0; k < _Vertices[_Triangles[i]._Vertices[1]]._Img_Num; k++)
						if (_Vertices[_Triangles[i]._Vertices[1]]._Img[k] == tttt) {
							uv2 = _Vertices[_Triangles[i]._Vertices[1]]._Img_Tex[k]; break;
						}
					for (int k = 0; k < _Vertices[_Triangles[i]._Vertices[2]]._Img_Num; k++)
						if (_Vertices[_Triangles[i]._Vertices[2]]._Img[k] == tttt) {
							uv3 = _Vertices[_Triangles[i]._Vertices[2]]._Img_Tex[k]; break;
						}

					cv::line(colorImages[tttt], cv::Point(uv1.x, uv1.y), cv::Point(uv2.x, uv2.y), cv::Scalar(0, 0, 255), 1);
					cv::line(colorImages[tttt], cv::Point(uv2.x, uv2.y), cv::Point(uv3.x, uv3.y), cv::Scalar(0, 0, 255), 1);
					cv::line(colorImages[tttt], cv::Point(uv3.x, uv3.y), cv::Point(uv1.x, uv1.y), cv::Scalar(0, 0, 255), 1);
				}
			}
		}
		cv::Mat tmp;
		cv::cvtColor(colorImages[tttt], tmp, CV_BGRA2BGR);
		//cv::imwrite(std::string("./imgs/subtexture_") + std::to_string(tttt) + std::string("11.jpg"), tmp);
		cv::imwrite(std::string("D:/3D_data/multiview_opt/capture/selected/subtexture_") + std::to_string(tttt) + std::string(".jpg"), tmp);
		cv::imshow("sub-textuers", colorImages[tttt]);
		cv::waitKey(0);
	}
}*/