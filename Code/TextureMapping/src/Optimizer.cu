#include "Optimizer.cuh"

using namespace TexMap;

__device__ float sampling_weights[3][SAMNUM] = { 0.5, 0.25, 0.25, 0.8, 0.1, 0.1, 0.5, 0.0, 0.5, 0.2, 0.6, 0.2, 0.1, 0.5, 0.4,
0.25, 0.5, 0.25, 0.1, 0.8, 0.1, 0.5, 0.5, 0.0, 0.2, 0.2, 0.6, 0.5, 0.4, 0.1,
0.25, 0.25, 0.5, 0.1, 0.1, 0.8, 0.0, 0.5, 0.5, 0.6, 0.2, 0.2, 0.4, 0.1, 0.5 };
__device__ float EDGE_WEIGHT = (0.5 * 255);
	
void deviceMemHolderT::push_back(hostTriangle &hT) {
	for (auto imIdx : hT._Img) _Img_t.push_back(imIdx);
	_SamCol_t.resize(_SamCol_t.size() + SAMNUM);
}

void deviceMemHolderV::push_back(hostVertex &hV) {
	for (auto triIdx : hV._Triangles) _Triangles_t.push_back((int)triIdx);
	for (auto imIdx : hV._Img) _Img_t.push_back((int)imIdx);
	for (auto imTex : hV._Img_Tex) _Img_Tex_t.push_back((float2)imTex);
	_Edge_Init_t.resize(_Edge_Init_t.size() + hV._Triangles.size() * hV._Img.size());
}

inline __device__ __host__ float2 sampling(float2 a, float2 b, float2 c, float alpha, float beta, float gamma)
{
	return alpha*a + beta*b + gamma*c;
}

__device__ __host__ void calcJTJ(float* out, float *J, float rows)
{
	/*
	M is samples x 2 (jacobian matrix)
	M^T * M
	*/
	out[0] = 0;
	out[1] = 0;
	out[2] = 0;
	out[3] = 0;
	for (int i = 0; i < rows * 2; i += 2) {
		out[0] += J[i] * J[i];
		out[1] += J[i] * J[i + 1];
		out[3] += J[i + 1] * J[i + 1];
	}
	out[2] = out[1];
}

__device__ __host__ void calcJTF(float* out, float *J, float *F, float rows)
{
	out[0] = 0;
	out[1] = 0;
	for (int i = 0; i < rows; i++) {
		out[0] += J[i * 2] * F[i];
		out[1] += J[i * 2 + 1] * F[i];
	}
}

__device__ __host__ void calcInv2x2(float *out, float *in)
{
	float mult = 1.0f / (in[0] * in[3] - in[1] * in[2]);
	out[0] = in[3] * mult;
	out[1] = -in[1] * mult;
	out[2] = -in[2] * mult;
	out[3] = in[0] * mult;
}

__device__ __host__ uchar linearInterpolate(uchar a, uchar b, uchar c, uchar d, float ddx, float ddy) {
	return (uchar)(a * (1 - ddx)*(1 - ddy) + b * ddx*(1 - ddy) + c*(1 - ddx)*ddy + d*ddx*ddy);
}

__device__ __host__ float gaussianInterpolate(float a, float b, float c, float d, float ddx, float ddy) {
	return expf(-(a*ddx*ddx + (b + c)*ddx*ddy + d*ddy*ddy));
}

__global__ void update_initial_edge(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, float2 *dmh_Edge_Init, int* dmh_ImgT, int *dmh_Tri, unsigned char *images, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	deviceVertex v1, v2, v3;
	int tmpImgNum = vCurrent._Img_Num;
	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];
		int tmpTriNum = vCurrent._Triangles_Num;
		for (int j = 0; j < tmpTriNum; j++) {
			deviceTriangle tCurrent = triangles[dmh_Tri[vCurrent._triOffset + j]];
			if (tCurrent.isCoveredBy(imgidx, &dmh_ImgT[tCurrent._imgOffset])) {
				int imgidx1, imgidx2, imgidx3;
				float2 uv1, uv2, uv3;

				// get other vertices
				if (tCurrent._Vertices[0] == vidx) {
					v1 = vCurrent;
					v2 = vertices[tCurrent._Vertices[1]];
					v3 = vertices[tCurrent._Vertices[2]];
					imgidx1 = v1.getImgIdx(imgidx, &dmh_ImgV[v1._imgOffset]);
					imgidx2 = v2.getImgIdx(imgidx, &dmh_ImgV[v2._imgOffset]);
					imgidx3 = v3.getImgIdx(imgidx, &dmh_ImgV[v3._imgOffset]);
					uv1 = dmh_Img_Tex[v1._imgOffset + imgidx1];
					uv2 = dmh_Img_Tex[v2._imgOffset + imgidx2];
					uv3 = dmh_Img_Tex[v3._imgOffset + imgidx3];
					float len1 = sqrt(dot(uv1 - uv2, uv1 - uv2));
					float len2 = sqrt(dot(uv1 - uv3, uv1 - uv3));
					dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j] = make_float2(len1, len2);
				}
				else if (tCurrent._Vertices[1] == vidx) {
					v1 = vertices[tCurrent._Vertices[0]];
					v2 = vCurrent;
					v3 = vertices[tCurrent._Vertices[2]];
					imgidx1 = v1.getImgIdx(imgidx, &dmh_ImgV[v1._imgOffset]);
					imgidx2 = v2.getImgIdx(imgidx, &dmh_ImgV[v2._imgOffset]);
					imgidx3 = v3.getImgIdx(imgidx, &dmh_ImgV[v3._imgOffset]);
					uv1 = dmh_Img_Tex[v1._imgOffset + imgidx1];
					uv2 = dmh_Img_Tex[v2._imgOffset + imgidx2];
					uv3 = dmh_Img_Tex[v3._imgOffset + imgidx3];
					float len1 = sqrt(dot(uv2 - uv1, uv2 - uv1));
					float len2 = sqrt(dot(uv2 - uv3, uv2 - uv3));
					dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j] = make_float2(len1, len2);
				}
				else if (tCurrent._Vertices[2] == vidx) {
					v1 = vertices[tCurrent._Vertices[0]];
					v2 = vertices[tCurrent._Vertices[1]];
					v3 = vCurrent;
					imgidx1 = v1.getImgIdx(imgidx, &dmh_ImgV[v1._imgOffset]);
					imgidx2 = v2.getImgIdx(imgidx, &dmh_ImgV[v2._imgOffset]);
					imgidx3 = v3.getImgIdx(imgidx, &dmh_ImgV[v3._imgOffset]);
					uv1 = dmh_Img_Tex[v1._imgOffset + imgidx1];
					uv2 = dmh_Img_Tex[v2._imgOffset + imgidx2];
					uv3 = dmh_Img_Tex[v3._imgOffset + imgidx3];
					float len1 = sqrt(dot(uv3 - uv1, uv3 - uv1));
					float len2 = sqrt(dot(uv3 - uv2, uv3 - uv2));
					dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j] = make_float2(len1, len2);
				}
				else {
					return;
				}
			}
		}
	}
}

__global__ void update_texture_coordinate(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, float2 *dmh_Edge_Init, int* dmh_ImgT, int *dmh_Tri, unsigned char *dmh_samcol, unsigned char *images, short *ug, short * vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	float F[(SAMNUM + 2)*MAXTRI]; // sampling points, edge regularize, position regularize
	float J[(SAMNUM * 2 + 4)*MAXTRI];
	int2 samples[SAMNUM];

	int size = w*h;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];

		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];

			float *sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {
				float2 uv1, uv2, uv3;
				float2 edge_length;
				edge_length = dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j];
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])]; 
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])]; 
					float2 diff1 = uv1 - uv2;
					float2 diff2 = uv1 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv2 - uv1;
					float2 diff2 = uv2 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
				}
				else {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
					float2 diff1 = uv3 - uv1;
					float2 diff2 = uv3 - uv2;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
				}

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = make_int2(sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]));
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					// get intensity from the image
					float intensity = images[imgidx*size + samples[s].y*w + samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy);
					J[currentSample * 2] = (float)ug[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];
					J[currentSample * 2 + 1] = (float)vg[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			float JTJ[4]; float JTF[2];
			float JTJinv[4];
			calcJTJ(JTJ, J, currentSample);
			calcJTF(JTF, J, F, currentSample);
			if (abs(JTJ[0] * JTJ[3] - JTJ[1] * JTJ[2]) < 0.0001) {
				JTJ[0] = JTJ[0] + 0.0001;
				JTJ[3] = JTJ[3] + 0.0001;
			}
			calcInv2x2(JTJinv, JTJ);
			float2 step = make_float2(JTJinv[0] * JTF[0] + JTJinv[1] * JTF[1], JTJinv[2] * JTF[0] + JTJinv[3] * JTF[1]);
			if (isnan(step.x) || isnan(step.y))
				step = make_float2(0, 0);
			step = clamp(step, -5.0, 5.0);
			dmh_Img_Tex[vertices[vidx]._imgOffset + i] = make_float2(clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].x - step.x, 0.0, w - 1.0), clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].y - step.y, 0.0, h - 1.0));
		}
	}
}

__global__ void calc_energy(float *energy, deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, int* dmh_ImgT, int *dmh_Tri, unsigned char *dmh_samcol, unsigned char *images, short *ug, short * vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	//float *F = (float*)malloc(SAMNUM * tmpTriNum);
	//float *F = new float[(SAMNUM) * tmpTriNum];
	float F[SAMNUM*MAXTRI]; // sampling points, edge regularize, position regularize
	int2 samples[SAMNUM];

	float vertex_energy = 0;
	int size = w*h;
	int imgcounted = 0;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum ; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];
		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];

			float *sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {
				int imgidx1, imgidx2, imgidx3;
				float2 uv1, uv2, uv3;
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[2] == vidx) {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
				}
				else {
					continue;
				}

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = make_int2(sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]));
					samples[s].x = min(w - 1, max(0, samples[s].x));
					samples[s].y = min(h - 1, max(0, samples[s].y));
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					// get intensity from the image
					float intensity = images[imgidx*size + samples[s].y*w + samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy) / 255.0f;
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			for (int k = 0; k < currentSample; k++)
				vertex_energy += F[k] * F[k];
			imgcounted++;
		}
	}

	energy[vidx] = vertex_energy;
}

__global__ void update_proxy_color(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, int* dmh_ImgT, unsigned char *dmh_samcol, unsigned char *images, int w, int h, int nTriangle)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= nTriangle) return;
	deviceTriangle tCurrent = triangles[tidx];
	int tmpImgNum = tCurrent._Img_Num;
	deviceVertex *v1 = &vertices[tCurrent._Vertices[0]];
	deviceVertex *v2 = &vertices[tCurrent._Vertices[1]];
	deviceVertex *v3 = &vertices[tCurrent._Vertices[2]];
	float samcol[SAMNUM];
	for (int i = 0; i < SAMNUM; i++) { samcol[i] = 0; }
	for (int i = 0; i < tmpImgNum; i++) { // for each image projects to this triangle
		int imgidx = dmh_ImgT[tCurrent._imgOffset + i];

		int imgidx1 = v1->getImgIdx(imgidx, &dmh_ImgV[v1->_imgOffset]);
		int imgidx2 = v2->getImgIdx(imgidx, &dmh_ImgV[v2->_imgOffset]);
		int imgidx3 = v3->getImgIdx(imgidx, &dmh_ImgV[v3->_imgOffset]);
		float2 uv1 = dmh_Img_Tex[v1->_imgOffset + imgidx1];
		float2 uv2 = dmh_Img_Tex[v2->_imgOffset + imgidx2];
		float2 uv3 = dmh_Img_Tex[v3->_imgOffset + imgidx3];

		int2 samples[SAMNUM];
		// sampling the uv coordinates
		for (int s = 0; s < SAMNUM; s++)
			samples[s] = make_int2(sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]));
		for (int s = 0; s < SAMNUM; s++) {
			// get intensity from the image
			samcol[s] += images[imgidx*w*h + samples[s].y*w + samples[s].x];// *tCurrent._Img_Weight[i];
		}
	}
	for (int s = 0; s < SAMNUM; s++)
		dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmpImgNum, 0.f, 255.f));
}

StopWatchInterface* Optimizer::sample_host(int width, int height, int nVertex, int nTriangle, float *hEnergy, int currentIter)
{
	StopWatchInterface *t = NULL;
	sdkCreateTimer(&t);
	sdkStartTimer(&t);
	checkCudaErrors(cudaDeviceSynchronize());

	dim3 gridSize_proxy((nTriangle + 512 - 1) / 512);
	dim3 blockSize_proxy(512);

	//printf("update_proxy_color\n");
	update_proxy_color << <gridSize_proxy, blockSize_proxy >> >(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._SamCol, dImages_original, width, height, nTriangle);
	cudaCheckError();

	dim3 gridSize((nVertex + 256 - 1) / 256);
	dim3 blockSize(256);
	//printf("update_texture_coordinate\n");
	update_texture_coordinate << < gridSize, blockSize >> >(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhVertices._Triangles, dmhTriangles._SamCol, dImages, dUg, dVg, width, height, nVertex);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaCheckError();
	update_texture_coordinate << < gridSize, blockSize >> >(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhVertices._Triangles, dmhTriangles._SamCol, dImages, dUg, dVg, width, height, nVertex);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaCheckError();


	update_proxy_color << <gridSize, blockSize >> >(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._SamCol, dImages_original, width, height, nTriangle);
	cudaMemset(dEnergy, 0, sizeof(float)*nVertex);
	calc_energy << < gridSize, blockSize >> >(dEnergy, ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhVertices._Triangles, dmhTriangles._SamCol, dImages_original, dUg, dVg, width, height, nVertex);

	checkCudaErrors(cudaMemcpy(hEnergy, dEnergy, sizeof(float)*nVertex, cudaMemcpyDeviceToHost));
	float sum_energy = 0;
	for (int i = 0; i < nVertex; i++) {
		sum_energy += hEnergy[i];
		//  printf("%f ", hEnergy[i]);
	}
	//sum_energy /= nVertex;
	printf("energy: %f, ", sum_energy);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&t);
	return t;
}

Optimizer::~Optimizer() {
	delete[] atlas_coord;
	delete[] img_coord;
	delete[] All_img_coord;
	freeCuda();
}
void Optimizer::Initialize(int layer) {
	if (layer == 0) {
		hCImages = new uchar[nImage*COIMX*COIMY * 3];
		hCImages4 = new uchar4[nImage*COIMX*COIMY];
		hImages = new uchar[nImage*COIMX*COIMY];
		hUg = new short[nImage*COIMX*COIMY];
		hVg = new short[nImage*COIMX*COIMY];
		for (int i = 0; i < nImage; i++) {
			cv::Mat1b grayImage;
			cv::Mat3b ch3Image;
			cv::cvtColor(rgbImages[i], grayImage, CV_BGRA2GRAY);
			cv::cvtColor(rgbImages[i], ch3Image, CV_BGRA2BGR);
			memcpy(&hImages[i*COIMX*COIMY], grayImage.data, sizeof(uchar)*COIMY*COIMX);
			memcpy(&hCImages[i*COIMX*COIMY * 3], ch3Image.data, sizeof(uchar)*COIMY*COIMX * 3);
			memcpy(&hCImages4[i*COIMX*COIMY], rgbImages[i].data, sizeof(uchar4)*COIMY*COIMX);
		}
		meshImages = new cv::Mat4b[nImage];
	}
	else {
		delete[] hEnergy;
		freeCuda();
	}
	hEnergy = new float[nVertex];

	initCuda(COIMX, COIMY, nVertex, nTriangle, nImage);
	dmhVertices.clear();
	dmhTriangles.clear();

	StopWatchInterface *t = NULL;
	sdkCreateTimer(&t);
	sdkStartTimer(&t);

	size_t imgOffsetT = 0;
	for (int i = 0; i < nTriangle; i++) {
		ddTriangles[i].init(hTriangles[i], imgOffsetT);
		dmhTriangles.push_back(hTriangles[i]);
		imgOffsetT += hTriangles[i]._Img.size();
	}
	dmhTriangles.ready();
	sdkStopTimer(&t);
	float layer_time = sdkGetAverageTimerValue(&t) / 1000.0f;
	printf("//////////////////////////////////////////separate f time: %fms\n", layer_time * 1000);
	t = NULL;
	sdkCreateTimer(&t);
	sdkStartTimer(&t);

	size_t imgOffsetV = 0;
	size_t triOffsetV = 0;
	size_t edgeOffsetV = 0;
	for (int i = 0; i < nVertex; i++) {
		ddVertices[i].init(hVertices[i], imgOffsetV, triOffsetV, edgeOffsetV);
		dmhVertices.push_back(hVertices[i]);
		imgOffsetV += hVertices[i]._Img.size();
		triOffsetV += hVertices[i]._Triangles.size();
		edgeOffsetV += hVertices[i]._Triangles.size() * hVertices[i]._Img.size();
	}
	dmhVertices.ready();
	sdkStopTimer(&t);
	layer_time = sdkGetAverageTimerValue(&t) / 1000.0f;
	printf("//////////////////////////////////////////separate v time: %fms\n", layer_time * 1000);

	checkCudaErrors(cudaMemcpy(dImages_original, hImages, sizeof(uchar)*COIMY*COIMX*nImage, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dCImages, hCImages, sizeof(uchar)*COIMY*COIMX*nImage * 3, cudaMemcpyHostToDevice));

#ifdef _DEBUG
	gridSize = dim3((nVertex + 512 - 1) / 512);
	blockSize = dim3(512);
#else
	gridSize = dim3((nVertex + 1024 - 1) / 1024);
	blockSize = dim3(1024);
#endif
	update_initial_edge << <gridSize, blockSize >> >(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhVertices._Triangles, dImages, COIMX, COIMY, nVertex);
	cudaCheckError();
	printf("Initializing done...\n");
}
int Optimizer::Update(int iteration, int layer) {
	bool layerup = true;
	int layeriter = 0;
	int till_iter = 0;
	for (int i = 0; i < iteration; i++) {
		till_iter++;
		if (layerup) {
			for (int j = 0; j < nImage; j++) {
				memcpy(&hImages[j*COIMX*COIMY], blur_vec[LAYERNUM - 1 - layer][j].data, sizeof(uchar)*COIMY*COIMX);
				memcpy(&hUg[j*COIMX*COIMY], hUg_vec[LAYERNUM - 1 - layer][j].data, sizeof(short)*COIMY*COIMX);
				memcpy(&hVg[j*COIMX*COIMY], hVg_vec[LAYERNUM - 1 - layer][j].data, sizeof(short)*COIMY*COIMX);
			}
			checkCudaErrors(cudaMemcpy(dImages, hImages, sizeof(uchar)*COIMY*COIMX*nImage, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dUg, hUg, sizeof(short)*COIMY*COIMX*nImage, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dVg, hVg, sizeof(short)*COIMY*COIMX*nImage, cudaMemcpyHostToDevice));
			layerup = false;
		}

		float preEnerge = 0;
		for (int i = 0; i < nVertex; i++) {
			preEnerge += hEnergy[i];
		}
		printf("layer %d, iteration %d - ", layer, i + 1);
		timer = sample_host(COIMX, COIMY, nVertex, nTriangle, hEnergy, i);
		float layer_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
		printf("elapsed time: %fms\n", layer_time * 1000);

		float sum_energy = 0;
		for (int i = 0; i < nVertex; i++) {
			sum_energy += hEnergy[i];
		}
		if (abs(sum_energy - preEnerge) / sum_energy < 1e-5) {
			break;
		}

	}
	preIter += iteration;
	return iteration - till_iter;
}

//// ¼Õº¼°Í
/*void Optimizer::WriteModel(string outFilePath) {
	checkCudaErrors(cudaMemcpy(hVertices, dVertices, sizeof(Vertex)*nVertex, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hTriangles, dTriangles, sizeof(Triangle)*nTriangle, cudaMemcpyDeviceToHost));
	Vertex_Save *hVertices_optimized = new Vertex_Save[nVertex];
	Triangle_Save *hTriangles_optimized = new Triangle_Save[nTriangle];
	for (int i = 0; i < nVertex; i++) {
		hVertices_optimized[i]._Col = hVertices_dummy[i]._Col;
		hVertices_optimized[i]._Img_Num = hVertices[i]._Img_Num;
		hVertices_optimized[i]._Pos = hVertices_dummy[i]._Pos;
		hVertices_optimized[i]._Triangles_Num = hVertices[i]._Triangles_Num;
		memcpy(hVertices_optimized[i]._Img, hVertices[i]._Img, sizeof(int)*MAXIMG);
		memcpy(hVertices_optimized[i]._Img_Tex, hVertices[i]._Img_Tex, sizeof(float2)*MAXIMG);
		memcpy(hVertices_optimized[i]._Triangles, hVertices[i]._Triangles, sizeof(int)*MAXTRI);
	}
	for (int i = 0; i < nTriangle; i++) {
		hTriangles_optimized[i]._Img_Num = hTriangles[i]._Img_Num;
		hTriangles_optimized[i]._Normal = hTriangles_dummy[i]._Normal;
		memcpy(hTriangles_optimized[i]._Img, hTriangles[i]._Img, sizeof(int)*hTriangles_optimized[i]._Img_Num);
		memcpy(hTriangles_optimized[i]._Img_Weight, hTriangles[i]._Img_Weight, sizeof(float)*MAXIMG);
		memcpy(hTriangles_optimized[i]._Vertices, hTriangles[i]._Vertices, sizeof(int) * 3);
		memcpy(hTriangles_optimized[i]._Tex_BIGPIC, hTriangles_dummy[i]._Tex_BIGPIC, sizeof(float2) * 3);
	}
	std::ofstream fout_s(outFilePath, std::ofstream::binary);
	fout_s.write((char*)&nVertex, sizeof(int));
	fout_s.write((char*)&nTriangle, sizeof(int));
	fout_s.write((char*)&nImage, sizeof(int));

	for (int i = 0; i < nVertex; i++) WriteVertex(hVertices_optimized[i], fout_s);
	for (int i = 0; i < nTriangle; i++) WriteTriangle(hTriangles_optimized[i], fout_s);

	fout_s.close();
	delete[] hVertices_optimized;
	delete[] hTriangles_optimized;
	delete[] hVertices_dummy;
	delete[] hTriangles_dummy;
}
void Optimizer::GetModel(Triangle_Load *tempT, Vertex_Load *tempV) {

	for (int i = 0; i < nVertex; i++) {
		tempV[i]._Img_Num = hVertices[i]._Img_Num;
		tempV[i]._Pos = hVertices_dummy[i]._Pos;
		tempV[i]._Triangles_Num = hVertices[i]._Triangles_Num;
		memcpy(tempV[i]._Img, hVertices[i]._Img, sizeof(int)*MAXIMG);
		memcpy(tempV[i]._Img_Tex, hVertices[i]._Img_Tex, sizeof(float2)*MAXIMG);
		memcpy(tempV[i]._Triangles, hVertices[i]._Triangles, sizeof(int)*MAXTRI);
	}
	for (int i = 0; i < nTriangle; i++) {
		tempT[i]._Img_Num = hTriangles[i]._Img_Num;
		tempT[i]._Normal = hTriangles_dummy[i]._Normal;
		memcpy(tempT[i]._Img, hTriangles[i]._Img, sizeof(int)*tempT[i]._Img_Num);
		memcpy(tempT[i]._Img_Weight, hTriangles[i]._Img_Weight, sizeof(float)*MAXIMG);
		memcpy(tempT[i]._Vertices, hTriangles[i]._Vertices, sizeof(int) * 3);
		memcpy(tempT[i]._Tex_BIGPIC, hTriangles_dummy[i]._Tex_BIGPIC, sizeof(float2) * 3);
	}
}*/
void Optimizer::PrepareRend_UVAtlas() {

	atlas_coord = new vector<float2>[nImage];
	atlas_tri_idx = new vector<int>[nImage];
	img_coord = new vector<float3>[nImage];

	int f_idx = 0;
	for (int f_idx = 0; f_idx < nTriangle; f_idx++) {
		hostVertex *hV[3];
		float2 tmp_at_coord[3] = { 0 };
		for (int i = 0; i < 3; i++)
			hV[i] = &hVertices[hTriangles[f_idx]._Vertices[i]];
		for (int i = 0; i < hTriangles[f_idx]._Img.size(); i++) {
			int vIm[3] = { -1,-1,-1 };
			int imgidx = hTriangles[f_idx]._Img[i];
			float weight = hTriangles[f_idx]._Img_Weight[i];
			for (int j = 0; j < 3; j++) {
				vIm[j] = hV[j]->getImgIdx(imgidx);
				atlas_coord[imgidx].push_back(tmp_at_coord[j]);
				img_coord[imgidx].push_back(make_float3(hV[j]->_Img_Tex[vIm[j]].x, hV[j]->_Img_Tex[vIm[j]].y, weight));
			}
			atlas_tri_idx[imgidx].push_back(f_idx);
			if (vIm[0] < 0 || vIm[1] < 0 || vIm[2] < 0)	cout << "Mapping wrong!!!";
		}
	}
}
void Optimizer::GetAtlasInfoi_UVAtlas(vector<float> *_uv, vector<float> *_uvImg, vector<int> *_triIdx, int idx) {
	_uv->resize(2 * atlas_coord[idx].size());
	_uvImg->resize(3 * img_coord[idx].size());
	_triIdx->resize(atlas_tri_idx[idx].size());
	memcpy(_uv->data(), atlas_coord[idx].data(), sizeof(float2)*atlas_coord[idx].size());
	memcpy(_uvImg->data(), img_coord[idx].data(), sizeof(float3)*img_coord[idx].size());
	memcpy(_triIdx->data(), atlas_tri_idx[idx].data(), sizeof(int)*atlas_tri_idx[idx].size());
}
void Optimizer::GetNumber(uint *nT, uint *nV) {
	*nT = nTriangle;
	*nV = nVertex;
}
void Optimizer::GetNumber(uint *nI, uint *w, uint *h) {
	*nI = nImage;
	*w = COIMX;
	*h = COIMY;
}

void Optimizer::Model4DLoadandMultiUpdate(Mapper4D * mapper4D, string streamPath) {
	mapper4D_ptr = mapper4D;
	All_img_coord = new vector<float3>[mapper4D_ptr->colorImages_dummy.size()];
	all_imgNum = mapper4D_ptr->colorImages_dummy.size();
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	rgbImages = new cv::Mat4b[nImage];
	depthImages = new cv::Mat1w[nImage];
	float * poses = new float[16 * nImage];

	if (FROM_FILE) {
		std::ifstream imageIn(streamPath, std::ios::in | std::ios::binary);
		for (int i = 0; i < nImage; i++) {
			rgbImages[i].create(COIMY, COIMX);
			depthImages[i].create(IRIMY, IRIMX);
			imageIn.read((char*)rgbImages[i].data, sizeof(uchar) * 4 * COIMX*COIMY);
			cv::cvtColor(rgbImages[i], rgbImages[i], CV_RGBA2BGRA);
			imageIn.read((char*)depthImages[i].data, sizeof(ushort) * IRIMX*IRIMY);
			imageIn.read((char*)&poses[i * 16], sizeof(float) * 16);
		}
		imageIn.close();
	}
	else {

		for (int i = 0; i < nImage; i++) {
			rgbImages[i].create(COIMY, COIMX);
			depthImages[i].create(IRIMY, IRIMX);
			cv::cvtColor(mapper4D_ptr->colorImages[i], rgbImages[i], CV_BGR2BGRA);
			depthImages[i] = mapper4D_ptr->depthImages[i].clone();
			memcpy(&poses[i * 16], mapper4D_ptr->_Pose.data, sizeof(float) * 16);
		}

	}

	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat4b tmpImage = rgbImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGRA2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	for (int i = 0; i < mapper4D_ptr->colorImages_dummy.size(); i++) {
		All_img_coord[i].resize(nTriangle * 3, { 0,0,0 });
	}

	vector<vector<float2>> propVec;
	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			/*hVertices.clear();
			hTriangles.clear();*/
			mapper4D_ptr->SetPropVec(propVec, layer);
		}
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}

		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, layer);

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		if (layer < LAYERNUM - 1) {
			if(opt_mode == "multi")
				remain_pre_layer = Update(iter_layer, layer);
		}
		else {
			if (opt_mode == "naive") {}
			else if (opt_mode == "single")
				remain_pre_layer = Update(iter_layer * LAYERNUM, layer);
			else
				remain_pre_layer = Update(iter_layer, layer);
			// remain_pre_layer = Update(iter_layer + remain_pre_layer, layer);
		}
		
		//checkCudaErrors(cudaMemcpy(hVertices, dVertices, sizeof(Vertex)*nVertex, cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(hTriangles, dTriangles, sizeof(Triangle)*nTriangle, cudaMemcpyDeviceToHost));
		//checkCudaErrors();

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2)*hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}

	}
}