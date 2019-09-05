#pragma once

#ifndef SIMPLIFIER_H
#define SIMPLIFIER_H

#if !defined(__CUDACC__)
#include<Eigen\Eigen>
#endif

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include "TriangleAndVertex.h"
#include "any.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

using namespace std;
using namespace Eigen;


class sTriangle {
public:
	unsigned int _Vertices[3];
	vector<Vector3f> _Normal;
	vector<float> _Area;
	vector<Matrix4f> _Q;
	bool valid = true;
};
class sEdge {
public:
	vector<Vector4f> _v;
	unsigned int _Endpoints[2];
	double DOD;
	bool operator==(const sEdge* e1) const {
		return ((this->_Endpoints[0] == e1->_Endpoints[0] || this->_Endpoints[0] == e1->_Endpoints[1]) && (this->_Endpoints[1] == e1->_Endpoints[0] || this->_Endpoints[1] == e1->_Endpoints[1]));
	}
	bool hasVertex(unsigned int v_idx) {
		return (this->_Endpoints[0] == v_idx || this->_Endpoints[1] == v_idx);
	}
};
class sVertex {
public:
	vector<Vector3f> _Pos;
	list<unsigned int> _Triangles;
	list<sEdge*> _Edges;
	vector<unsigned int> verHistory;
	bool valid = true;
};
struct sEdgeComp
{
	bool operator()(const sEdge* lhs, const sEdge* rhs) const
	{
		return lhs->DOD < rhs->DOD;
	}
};

class lTriangle {
public:
	int		_Vertices[3];
	vector<float3> _Normal;
	vector<float> _Area;
};
class lVertex {
public:
	vector<float3> _Pos;
	vector<int> _Triangles;
	vector<unsigned int> verHistory;
};

namespace TexMap {
	class Simplifier {
	public:
		Simplifier(vector<MyMesh> tmp_mesh_vec);

		void simplify(unsigned int tar_Num);
		void simplify_layer(int layer);

		void savetmpOBJ(string fileName);
		void savetmpOBJ(string fileName, int frame_idx);

		vector<vector<lTriangle>> extTri;
		vector<vector<lVertex>> extVer;
		vector<vector<int>> idx_table_Tri;
		vector<vector<int>> idx_table_Ver;
		vector<vector<int>> idx_table_Tri_inv;
		vector<vector<int>> idx_table_Ver_inv;


	private:
		vector<sTriangle> progTri;
		vector<sVertex> progVer;
		multiset<sEdge *, sEdgeComp> progEdge;
		unsigned int frameNum;
		unsigned int layer_now;

		void stamp_now();
		void updateEdge(sEdge &e);
		void remove_from_heap(std::multiset<sEdge*, sEdgeComp>& eh, sEdge* e);
		void remove_from_list(std::list<sEdge*>& el, sEdge* e);
		void updateTri(sTriangle &t);
		int collapseEdge(sEdge &e);
		};
}




#endif