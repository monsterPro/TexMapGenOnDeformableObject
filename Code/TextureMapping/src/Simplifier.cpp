#include "Simplifier.h"

using namespace TexMap;

Simplifier::Simplifier(vector<MyMesh> tmp_mesh_vec) {
	//initialize
	progTri.resize(tmp_mesh_vec[0].n_faces());
	progVer.resize(tmp_mesh_vec[0].n_vertices());
	frameNum = tmp_mesh_vec.size();
	//memory init
	for (int i = 0; i < progTri.size(); i++) {
		progTri[i]._Normal.resize(tmp_mesh_vec.size());
		progTri[i]._Area.resize(tmp_mesh_vec.size());
		progTri[i]._Q.resize(tmp_mesh_vec.size());
	}
	for (int i = 0; i < progVer.size(); i++) {
		progVer[i]._Pos.resize(tmp_mesh_vec.size());
	}
	//4D elements init
	for (int i = 0; i < frameNum; i++) {
		//Vertex4D init
		for (MyMesh::VertexIter v_it = tmp_mesh_vec[i].vertices_begin(); v_it != tmp_mesh_vec[i].vertices_end(); ++v_it) {
			//position
			progVer[v_it->idx()]._Pos[i] = Vector3f(tmp_mesh_vec[i].point(*v_it)[0], tmp_mesh_vec[i].point(*v_it)[1], tmp_mesh_vec[i].point(*v_it)[2]);
		}
		//Triangle4D init
		for (MyMesh::FaceIter f_it = tmp_mesh_vec[i].faces_begin(); f_it != tmp_mesh_vec[i].faces_end(); ++f_it) {
			//vertex index
			Vector3f f_p[3];
			int k = 0;
			for (MyMesh::FaceVertexIter fv_it = tmp_mesh_vec[i].fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
				f_p[k] = Vector3f(tmp_mesh_vec[i].point(*fv_it)[0], tmp_mesh_vec[i].point(*fv_it)[1], tmp_mesh_vec[i].point(*fv_it)[2]);
				k++;
			}
			//normal
			progTri[f_it->idx()]._Normal[i] = Vector3f(tmp_mesh_vec[i].normal(*f_it)[0], tmp_mesh_vec[i].normal(*f_it)[1], tmp_mesh_vec[i].normal(*f_it)[2]);
			//area
			progTri[f_it->idx()]._Area[i] = ((f_p[0] - f_p[1]).cross(f_p[0] - f_p[2])).norm() / 2.0;
			//Q
			float d = -progTri[f_it->idx()]._Normal[i].dot(f_p[0]);
			Vector4f q = Vector4f(progTri[f_it->idx()]._Normal[i].x(), progTri[f_it->idx()]._Normal[i].y(), progTri[f_it->idx()]._Normal[i].z(), d);
			progTri[f_it->idx()]._Q[i] = q*(q.transpose());
		}
	}

	//3D elements init
	for (MyMesh::VertexIter v_it = tmp_mesh_vec[0].vertices_begin(); v_it != tmp_mesh_vec[0].vertices_end(); ++v_it) {
		//connected face
		for (MyMesh::VertexFaceIter vf_it = tmp_mesh_vec[0].vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
			progVer[v_it->idx()]._Triangles.push_back(vf_it->idx());
		}
		//history init
		progVer[v_it->idx()].verHistory.push_back(v_it->idx());
	}
	for (MyMesh::FaceIter f_it = tmp_mesh_vec[0].faces_begin(); f_it != tmp_mesh_vec[0].faces_end(); ++f_it) {
		//vertex index
		int k = 0;
		for (MyMesh::FaceVertexIter fv_it = tmp_mesh_vec[0].fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
			progTri[f_it->idx()]._Vertices[k] = fv_it->idx();
			k++;
		}
	}
	for (MyMesh::EdgeIter e_it = tmp_mesh_vec[0].edges_begin(); e_it != tmp_mesh_vec[0].edges_end(); ++e_it) {
		sEdge *edge = new sEdge;
		unsigned int v0_idx = tmp_mesh_vec[0].to_vertex_handle(tmp_mesh_vec[0].halfedge_handle(*e_it, 0)).idx();
		unsigned int v1_idx = tmp_mesh_vec[0].from_vertex_handle(tmp_mesh_vec[0].halfedge_handle(*e_it, 0)).idx();
		edge->_Endpoints[0] = v0_idx;
		edge->_Endpoints[1] = v1_idx;
		updateEdge(*edge);

		progVer[v0_idx]._Edges.push_back(edge);
		progVer[v1_idx]._Edges.push_back(edge);
		progEdge.insert(edge);
	}

}

void Simplifier::simplify(unsigned int tar_Num) {
	unsigned int count = 0;
	for (auto t : progTri) {
		if (t.valid)
			count++;
	}
	unsigned int preNum = count;

	cout << "Mesh simplifying to " << tar_Num << " Faces..." << endl;
	while (count > tar_Num) {
		//cout << "Mesh has " << count << "Faces..." << endl;
		printProgBar(((float)(preNum - count) / (float)(preNum - tar_Num - 2)) * 100);

		sEdge *min_edge = NULL;

		if (progEdge.begin() != progEdge.end())
			min_edge = *progEdge.begin();
		else
			return;

		count -= collapseEdge(*min_edge);
	}
	cout << endl;

	savetmpOBJ("D:/3D_data/test/simple.obj");

}
void Simplifier::simplify_layer(int layer) {
	layer_now = layer - 1;
	//original mesh
	stamp_now();
	while (layer > 1) {
		layer--;
		unsigned int count = 0;
		for (auto t : progTri) {
			if (t.valid)
				count++;
		}
		unsigned int preNum = count;
		unsigned int tar_Num = preNum / DIVIDER_LAYER;
		cout << "Mesh simplifying to " << tar_Num << " Faces..." << endl;
		while (count > tar_Num) {
			//cout << "Mesh has " << count << "Faces..." << endl;
			printProgBar(((float)(preNum - count) / (float)(preNum - tar_Num - 2)) * 100);

			sEdge *min_edge = NULL;

			if (progEdge.begin() != progEdge.end())
				min_edge = *progEdge.begin();
			else
				return;

			count -= collapseEdge(*min_edge);
			/*if (count == tar_Num + 2) {
			savetmpOBJ("D:/3D_data/test/layer_" + to_string(count) + "_0.obj");
			savetmpOBJ("D:/3D_data/test/layer_" + to_string(count) + "_10.obj", 10);
			savetmpOBJ("D:/3D_data/test/layer_" + to_string(count) + "_20.obj", 20);
			}*/

		}
		//simple mesh
		/*savetmpOBJ("D:/3D_data/test/layer_"+ to_string(tar_Num) + "_0.obj");
		savetmpOBJ("D:/3D_data/test/layer_" + to_string(tar_Num) + "_10.obj", 10);
		savetmpOBJ("D:/3D_data/test/layer_" + to_string(tar_Num) + "_20.obj", 20);*/
		layer_now = layer - 1;
		stamp_now();
		cout << " Done..." << endl;
	}
	cout << endl;
}
void Simplifier::savetmpOBJ(string fileName) {
	//save .obj
	FILE* OutFileHandle = fopen(fileName.c_str(), "w");
	unsigned int numTri = 0;
	unsigned int numVer = 0;
	vector<int> idx_table(progVer.size(), -1);
	for (auto t : progTri) {
		if (t.valid)
			numTri++;
	}
	unsigned int count = 0;
	for (int i = 0; i < progVer.size(); i++) {
		if (progVer[i].valid) {
			numVer++;
			idx_table[count] = numVer;
		}
		count++;
	}
	fprintf(OutFileHandle, "####\n");
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "# OBJ File Generated by Meshlab\n");
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "####\n");
	fprintf(OutFileHandle, "# Object solidtexture.obj\n");
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "# Vertices: %d\n", numVer);
	fprintf(OutFileHandle, "# Faces : %d\n", numTri);
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "####\n");

	for (int i = 0; i < progVer.size(); i++) {
		if (progVer[i].valid)
			fprintf(OutFileHandle, "v %f %f %f\n", progVer[i]._Pos[0].x(), progVer[i]._Pos[0].y(), progVer[i]._Pos[0].z());
	}
	fprintf(OutFileHandle, "\n\n");
	for (auto t : progTri) {
		if (t.valid)
			fprintf(OutFileHandle, "f %d %d %d\n", idx_table[t._Vertices[0]], idx_table[t._Vertices[1]], idx_table[t._Vertices[2]]);
	}
	fclose(OutFileHandle);
}
void Simplifier::savetmpOBJ(string fileName, int frame_idx) {
	//save .obj
	FILE* OutFileHandle = fopen(fileName.c_str(), "w");
	unsigned int numTri = 0;
	unsigned int numVer = 0;
	vector<int> idx_table(progVer.size(), -1);
	for (auto t : progTri) {
		if (t.valid)
			numTri++;
	}
	unsigned int count = 0;
	for (int i = 0; i < progVer.size(); i++) {
		if (progVer[i].valid) {
			numVer++;
			idx_table[count] = numVer;
		}
		count++;
	}
	fprintf(OutFileHandle, "####\n");
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "# OBJ File Generated by Meshlab\n");
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "####\n");
	fprintf(OutFileHandle, "# Object solidtexture.obj\n");
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "# Vertices: %d\n", numVer);
	fprintf(OutFileHandle, "# Faces : %d\n", numTri);
	fprintf(OutFileHandle, "#\n");
	fprintf(OutFileHandle, "####\n");

	for (int i = 0; i < progVer.size(); i++) {
		if (progVer[i].valid)
			fprintf(OutFileHandle, "v %f %f %f\n", progVer[i]._Pos[frame_idx].x(), progVer[i]._Pos[frame_idx].y(), progVer[i]._Pos[frame_idx].z());
	}
	fprintf(OutFileHandle, "\n\n");
	for (auto t : progTri) {
		if (t.valid)
			fprintf(OutFileHandle, "f %d %d %d\n", idx_table[t._Vertices[0]], idx_table[t._Vertices[1]], idx_table[t._Vertices[2]]);
	}
	fclose(OutFileHandle);
}
void Simplifier::stamp_now() {
			vector<lTriangle> tmpTri_vec;
			vector<lVertex> tmpVer_vec;
			vector<int> tmp_table_Tri;
			vector<int> tmp_table_Ver;
			vector<int> tmp_table_Tri_inv;
			vector<int> tmp_table_Ver_inv;
			unsigned int numTri = 0;
			unsigned int numVer = 0;
			unsigned int count = 0;

			//table init
			tmp_table_Tri.resize(progTri.size(), -1);
			tmp_table_Ver.resize(progVer.size(), -1);
			for (int i = 0; i < progTri.size(); i++) {
				if (progTri[i].valid) {
					tmp_table_Tri[i] = count;
					tmp_table_Tri_inv.push_back(i);
					count++;
				}
			}
			count = 0;
			for (int i = 0; i < progVer.size(); i++) {
				if (progVer[i].valid) {
					tmp_table_Ver[i] = count;
					tmp_table_Ver_inv.push_back(i);
					count++;
				}
			}

			for (auto t : progTri) {
				if (t.valid) {
					numTri++;
					lTriangle tmpTri;
					tmpTri._Vertices[0] = tmp_table_Ver[t._Vertices[0]];
					tmpTri._Vertices[1] = tmp_table_Ver[t._Vertices[1]];
					tmpTri._Vertices[2] = tmp_table_Ver[t._Vertices[2]];
					float3 tmp_normal;
					for (auto nn : t._Normal) {
						tmp_normal.x = nn(0);
						tmp_normal.y = nn(1);
						tmp_normal.z = nn(2);
						tmpTri._Normal.push_back(tmp_normal);
					}
					for (auto a : t._Area) {
						tmpTri._Area.push_back(a);
					}
					tmpTri_vec.push_back(tmpTri);
				}
			}
			for (int i = 0; i < progVer.size(); i++) {
				if (progVer[i].valid) {
					numVer++;
					lVertex tmpVer;
					float3 tmp_pos;
					for (auto p : progVer[i]._Pos) {
						tmp_pos.x = p(0);
						tmp_pos.y = p(1);
						tmp_pos.z = p(2);
						tmpVer._Pos.push_back(tmp_pos);
					}
					for (list<unsigned int>::iterator l_it = progVer[i]._Triangles.begin(); l_it != progVer[i]._Triangles.end(); l_it++) {
						tmpVer._Triangles.push_back(tmp_table_Tri[*l_it]);
					}
					for (auto h : progVer[i].verHistory) {
						tmpVer.verHistory.push_back(h);
					}
					tmpVer_vec.push_back(tmpVer);
				}
				/*progVer[i].verHistory.clear();
				progVer[i].verHistory.push_back(i);*/
			}

			extTri.push_back(tmpTri_vec);
			extVer.push_back(tmpVer_vec);
			idx_table_Tri.push_back(tmp_table_Tri);
			idx_table_Ver.push_back(tmp_table_Ver);
			idx_table_Tri_inv.push_back(tmp_table_Tri_inv);
			idx_table_Ver_inv.push_back(tmp_table_Ver_inv);
			if (RECORD_UNIT) {
				for (int i = 0; i < frameNum; i++) {
					savetmpOBJ(data_root_path + "/unit_test/" + unit_test_path + "/decimated_mesh/" + to_string(layer_now) + "/sampled_Frame_" + to_string(i) + ".obj", i);
				}
			}
		}
void Simplifier::updateEdge(sEdge &e) {
			if (e._v.empty())
				e._v.resize(frameNum);
			sVertex v0 = progVer[e._Endpoints[0]];
			sVertex v1 = progVer[e._Endpoints[1]];

			float DSD = 0;
			float avg_Area = 0;
			//DSD & Area
			for (int i = 0; i < frameNum; i++) {
				Matrix4f quadric = Matrix4f::Zero();
				Matrix4f quadric0 = Matrix4f::Zero();
				Matrix4f quadric1 = Matrix4f::Zero();

				vector<unsigned int> tri_vec;  ///////////////////////////
											   //vector<unsigned int> edge_tri; //for finding edge's face//

				float area = 0;
				for (auto t : v0._Triangles) {
					quadric0 = quadric0 + progTri[t]._Q[i];
					area += progTri[t]._Area[i];
					tri_vec.push_back(t);
				}
				quadric0 = quadric0*area;

				area = 0;
				for (auto t : v1._Triangles) {
					quadric1 = quadric1 + progTri[t]._Q[i];
					area += progTri[t]._Area[i];
					if (find(tri_vec.begin(), tri_vec.end(), t) != tri_vec.end())
						avg_Area += progTri[t]._Area[i];
				}
				quadric1 = quadric1*area;

				quadric = quadric0 + quadric1;

				Matrix4f w = Matrix4f::Identity();
				w(0, 0) = quadric(0, 0); w(0, 1) = quadric(0, 1); w(0, 2) = quadric(0, 2); w(0, 3) = quadric(0, 3);
				w(1, 0) = quadric(0, 1); w(1, 1) = quadric(1, 1); w(1, 2) = quadric(1, 2); w(1, 3) = quadric(1, 3);
				w(2, 0) = quadric(0, 2); w(2, 1) = quadric(1, 2); w(2, 2) = quadric(2, 2); w(2, 3) = quadric(2, 3);

				//w.topRows(3) = quadric.topRows(3);
				if (w.determinant() > 1e-3) {
					e._v[i] = (w.inverse()*Vector4f(0, 0, 0, 1));
				}
				else {
					e._v[i] = ((v0._Pos[i] + v1._Pos[i]) / 2.0).homogeneous();
				}
				//QEM
				DSD += e._v[i].dot(quadric * e._v[i]);
			}
			avg_Area /= frameNum;

			float avg_len_diff = 0;
			float sqr_len_diff = 0;
			float max_len_diff = 0;
			vector<float> len_vec;
			//len_diff
			for (int i = 0; i < frameNum - 1; i++) {
				//eta
				float len_diff = abs((v0._Pos[i] - v1._Pos[i]).norm() - (v0._Pos[i + 1] - v1._Pos[i + 1]).norm());
				avg_len_diff += len_diff;
				len_vec.push_back(len_diff);
				if (max_len_diff < len_diff)
					max_len_diff = len_diff;
				//sqr_len_diff += pow(len_diff,2);
			}
			avg_len_diff /= frameNum - 1;
			for (auto ld : len_vec) {
				sqr_len_diff += pow(ld - avg_len_diff, 2);
			}
			len_vec.clear();
			//frameNum * (frameNum - 1)*(frameNum - 1)*
			float w = (max_len_diff - avg_len_diff) / pow(sqr_len_diff / (frameNum - 1), 0.5);

			e.DOD = DSD + (w * avg_Area * pow(avg_len_diff, 2));
		}
void Simplifier::remove_from_heap(std::multiset<sEdge*, sEdgeComp>& eh, sEdge* e)
		{
			auto pp = eh.equal_range(e);
			std::multiset<sEdge*, sEdgeComp>::iterator tod;
			for (auto it = pp.first; it != pp.second; it++)
				if (*it == e)
					tod = it;
			eh.erase(tod);
		}
void Simplifier::remove_from_list(std::list<sEdge*>& el, sEdge* e)
		{
			el.remove(e);
			/*list<sEdge*>::iterator e_iter = find(el.begin(), el.end(), e);
			el.erase(e_iter);*/
		}
void Simplifier::updateTri(sTriangle &t) {
			for (int i = 0; i < frameNum; i++) {
				Vector3f f_p[3];
				for (int k = 0; k < 3; k++) {
					f_p[k] = progVer[t._Vertices[k]]._Pos[i];
				}
				Vector3f nor_n_unit = (f_p[0] - f_p[1]).cross(f_p[0] - f_p[2]);
				//area
				t._Area[i] = nor_n_unit.norm() / 2.0;
				//normal
				nor_n_unit.normalize();
				t._Normal[i] = nor_n_unit;
				//Q
				float d = -nor_n_unit.dot(f_p[0]);
				Vector4f q = Vector4f(nor_n_unit.x(), nor_n_unit.y(), nor_n_unit.z(), d);
				t._Q[i] = q*(q.transpose());
			}
		}
int Simplifier::collapseEdge(sEdge &e) {
			unsigned int v0_idx, v1_idx;
			set<unsigned int> v_cross_index;

			v0_idx = e._Endpoints[0];
			v1_idx = e._Endpoints[1];

			sVertex* v0 = &progVer[e._Endpoints[0]];
			sVertex* v1 = &progVer[e._Endpoints[1]];

			set<unsigned int> intersec_tri;
			set<unsigned int> union_inter_tri;
			//set<unsigned int> recon_tri;
			set<unsigned int> diffset_tri;

			vector<sEdge*> del_edge;

			//__eeeeeeeee__do not use union_inter_tri
			/*for (auto t : v1->_Triangles) {
			if(find(v0->_Triangles.begin(), v0->_Triangles.end(), t) != v0->_Triangles.end()) {
			intersec_tri.insert(t);
			for (auto t_v : progTri[t]._Vertices)
			if (t_v != v0_idx && t_v != v1_idx)
			v_cross_index.insert(t_v);
			}
			else {
			diffset_tri.insert(t);
			}
			}*/

			//init union_inter_tri
			for (auto t : v0->_Triangles) {
				union_inter_tri.insert(t);
			}
			//find intersect tri and delete tri from v_cross_index
			for (auto t : v1->_Triangles) {
				if (union_inter_tri.find(t) != union_inter_tri.end()) {
					intersec_tri.insert(t);
					for (auto t_v : progTri[t]._Vertices)
						if (t_v != v0_idx && t_v != v1_idx) {
							v_cross_index.insert(t_v);
							list<unsigned int>::iterator t_iter = find(progVer[t_v]._Triangles.begin(), progVer[t_v]._Triangles.end(), t);
							progVer[t_v]._Triangles.erase(t_iter);
						}
				}
				else {
					union_inter_tri.insert(t);
					diffset_tri.insert(t);
				}
			}
			//construct union_inter_tri
			for (auto t : intersec_tri) {
				if (union_inter_tri.find(t) != union_inter_tri.end())
					union_inter_tri.erase(union_inter_tri.find(t));
			}
			//assert(v_cross_index.size() == 2);
			/*if (v_cross_index.size() != 2) {
			cout << "!!!!!!!!!!! : "<< v_cross_index.size() << endl << endl;

			}*/
			//delete target edge from v_cross_index
			for (auto e : v1->_Edges) {
				if (e->hasVertex(v0_idx))
					del_edge.push_back(e);
				else {
					for (auto c_v : v_cross_index) {
						if (e->hasVertex(c_v)) {
							del_edge.push_back(e);
							remove_from_list(progVer[c_v]._Edges, e);
						}
					}
				}
			}
			//add history
			//v0->verHistory.push_back(v1_idx);
			for (auto h : v1->verHistory) {
				v0->verHistory.push_back(h);
			}
			//reconnect Tri & edge with v0 and delete target edge
			v0->_Triangles.sort();
			v1->_Triangles.sort();
			v0->_Triangles.merge(v1->_Triangles);
			v0->_Triangles.sort();
			v0->_Triangles.unique();
			for (auto t : intersec_tri) {
				v0->_Triangles.remove(t);
			}
			for (auto t : diffset_tri) {
				//v0->_Triangles.push_back(t);
				for (int i = 0; i < 3; i++)
					if (progTri[t]._Vertices[i] == v1_idx)
						progTri[t]._Vertices[i] = v0_idx;
			}
			v0->_Edges.sort();
			v1->_Edges.sort();
			v0->_Edges.merge(v1->_Edges);
			v0->_Edges.sort();
			v0->_Edges.unique();
			for (auto e : del_edge) {
				remove_from_list(v0->_Edges, e);
				remove_from_list(v1->_Edges, e);
				remove_from_heap(progEdge, e);
				//delete e;
			}
			del_edge.clear();
			//delete edge's triangle
			for (auto t : intersec_tri) {
				progTri[t].valid = false;
			}
			//update Q & area & normal
			for (auto t : union_inter_tri) {
				updateTri(progTri[t]);
			}
			for (int i = 0; i < frameNum; i++) {
				v0->_Pos[i] = e._v[i].head(3);
			}
			//update edges
			for (auto e : v0->_Edges) {
				remove_from_heap(progEdge, e);
				for (int i = 0; i < 2; i++) {
					if (e->_Endpoints[i] == v1_idx)
						e->_Endpoints[i] = v0_idx;
				}
				updateEdge(*e);
				progEdge.insert(e);
			}
			v1->valid = false;

			return intersec_tri.size();
		}