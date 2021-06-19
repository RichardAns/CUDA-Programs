// cxtextures.h
#pragma once
// Copyright Richard Ansorge 2020. 
// Part of cx utility suite devevoped for CUP Publication:
//         Programming in Parallel with CUDA 
//
// This vesion only for cudaArrayDefault
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//==========================================================================
// Helper functions create 1, 2 and 3D textures, the user must supply a 
// cudaArray or C array for the object.  If this pointer is null the object
// will be allocated. Data for the texture is in the template type *data and 
// will be copied to the texture if data not null. The available options are:
//
//  address modes            filter modes          read modes               
//
// cudaAddressModeWrap    cudaFilterModePoint   cudaReadModeElementType     
// cudaAddressModeClamp   cudaFilterModeLinear  cudaReadModeNormalizedFloat 
// cudaAddressModeMirror													
// cudaAddressModeBorder    												
//  
// coordinate modes          arrayTypes
//
//   1 => normalized       cudaArrayDefault
//   0 => natural	       cudaArrayLayered
//         or              cudaArraySurfaceLoadStore
// cudaCoordNormalized     cudaArrayCubemap
// cudaCoordNatural        cudaArrayTextureGather
//===========================================================================
// The classes create a 1, 2 or 3D texture and optional surface sharing same 
// cudaArray. Member functions copyTo and copyFrom allow trqansfer of data
// between host and device memories. The destructors tidy up on exit.
// Either individual public member varaiables, e.g. tex, can be passed as 
// kernel arguments or, if several are required, the object itself is safe 
// to pass as a kernel argument.
//===========================================================================

// Since CUDA does not define names for the texture coordinate modes we will
#define cudaCoordNormalized 1
#define cudaCoordNatural    0

namespace cx {
	// class txs1D 1D-texture
template <typename T> class txs1D {
	private:
		cudaArray * carray;
	public:
		int1 n;
		cudaTextureObject_t tex;

	txs1D(){ n ={0}; carray = nullptr; tex = 0; }

	txs1D(int1 m, T *data, cudaTextureFilterMode filtermode,
		cudaTextureAddressMode addressmode, cudaTextureReadMode readmode,
		int normmode, int arrayType=cudaArrayDefault)
	{
		n = m; tex = 0; carray = nullptr;

		cudaChannelFormatDesc cd = cudaCreateChannelDesc<T>();
		cx::ok(cudaMallocArray(&carray,&cd,n.x,1,arrayType));

		if(data != nullptr){
			cx::ok(cudaMemcpyToArray(carray,0,0,data,n.x*sizeof(T),
				cudaMemcpyHostToDevice));
		}

		cudaResourceDesc rd ={};  // make ResourceDesc
		rd.resType = cudaResourceTypeArray;
		rd.res.array.array = carray;

		
		cudaTextureDesc td ={};  // make TextureDesc
		td.addressMode[0] = addressmode;
		td.filterMode     = filtermode;
		td.readMode       = readmode; 
		td.normalizedCoords = normmode;
		cx::ok(cudaCreateTextureObject(&tex,&rd,&td,nullptr));
	}

	txs1D(const txs1D &txs2){ n.x = txs2.n.x; carray = nullptr; tex = txs2.tex; }

	void copyTo(T *data) {
		if(data != nullptr && carray != nullptr) cx::ok(
			cudaMemcpyToArray(carray,0,0,data,n.x*sizeof(T),
				cudaMemcpyHostToDevice));
		return;
	}

	void copyFrom(T *data) {
		if(data != nullptr && carray != nullptr) cx::ok(
			cudaMemcpyFromArray(data,carray,0,0,n.x*sizeof(T),
				cudaMemcpyDeviceToHost));
		return;
	}

	~txs1D() {
		if(carray != nullptr){
			if(tex != 0) cx::ok(cudaDestroyTextureObject(tex));
			cx::ok(cudaFreeArray(carray));
		}
	}
};  // end class txs1D

// class txs2D 2D-texture
template <typename T> class txs2D {
	private:
		cudaArray * carray;
	public:
		int2 n;
		cudaTextureObject_t tex;

	txs2D(){ n ={0,0}; carray = nullptr; tex = 0; }

	txs2D(int2 m, T *data, cudaTextureFilterMode filtermode, 
		cudaTextureAddressMode addressmode, cudaTextureReadMode readmode, 
		int normmode, int arrayType=cudaArrayDefault)
	{
		n = m; tex = 0; carray = nullptr;
		cudaChannelFormatDesc cd = cudaCreateChannelDesc<T>();
		cx::ok(cudaMallocArray(&carray,&cd,n.x,n.y,arrayType));

		if(data != nullptr){
			cx::ok(cudaMemcpyToArray(carray,0,0,data,n.x*n.y*sizeof(T),cudaMemcpyHostToDevice));
		}

		cudaResourceDesc rd ={};  // make ResourceDesc
		rd.resType = cudaResourceTypeArray;
		rd.res.array.array = carray;

		// make TextureDesc
		cudaTextureDesc td ={};  // make TextureDesc
		td.addressMode[0] = addressmode;
		td.addressMode[1] = addressmode; 
		td.filterMode     = filtermode;
		td.readMode       =  readmode;
		td.normalizedCoords = normmode;
		cx::ok(cudaCreateTextureObject(&tex,&rd,&td,nullptr));
	}

	txs2D(const txs2D &txs2){ n = txs2.n; carray = nullptr; tex = txs2.tex; }

	void copyTo(T *data) {
		if(data != nullptr && carray != nullptr) cx::ok(cudaMemcpyToArray(carray,0,0,data,n.x*n.y*sizeof(T),cudaMemcpyHostToDevice));
		return;
	}

	void copyFrom(T *data) {
		if(data != nullptr && carray != nullptr) cx::ok(cudaMemcpyFromArray(data,carray,0,0,n.x*n.y*sizeof(T),cudaMemcpyDeviceToHost));
		return;
	}

	~txs2D() {
		if(carray != nullptr){
			if(tex != 0) cx::ok(cudaDestroyTextureObject(tex));
			cx::ok(cudaFreeArray(carray));
		}
	}
};  // end txs2D

// class txs3D 3D-texture
template <typename T> class txs3D {
	private:
		cudaArray * carray;
	public:
		int3 n;
		cudaTextureObject_t tex;

	txs3D(){ n ={0,0,0}; carray = nullptr; tex = 0; }

	void copy3D(T* data,cudaMemcpyKind  copykind)
	{
		cudaMemcpy3DParms cp ={0};
		cp.srcPtr   = make_cudaPitchedPtr(data,n.x*sizeof(T),n.x,n.y); 
		cp.dstArray = carray;
		cp.extent   = make_cudaExtent(n.x,n.y,n.z);
		cp.kind     = copykind;
		cx::ok(cudaMemcpy3D(&cp));
	}

	txs3D(int3 m, T *data, cudaTextureFilterMode filtermode, 
		cudaTextureAddressMode addressmode, cudaTextureReadMode readmode, 
		int normmode, int arrayType=cudaArrayDefault)
	{
		n = m; tex = 0; carray = nullptr;

		cudaChannelFormatDesc cd = cudaCreateChannelDesc<T>();
		cudaExtent cx ={(size_t)n.x,(size_t)n.y,(size_t)n.z}; 
		cx::ok(cudaMalloc3DArray(&carray,&cd,cx,arrayType));
		if(data != nullptr) copy3D(data,cudaMemcpyHostToDevice);

		cudaResourceDesc rd ={};  // make ResourceDesc
		rd.resType = cudaResourceTypeArray;
		rd.res.array.array = carray;

		// make TextureDesc
		cudaTextureDesc td ={};  // make TextureDesc
		td.addressMode[0] = addressmode;
		td.addressMode[1] = addressmode;
		td.addressMode[2] = addressmode; 
		td.filterMode     = filtermode;
		td.readMode       =  readmode;
		td.normalizedCoords = normmode;
		cx::ok(cudaCreateTextureObject(&tex,&rd,&td,nullptr));
	}

	txs3D(const txs3D &txs2){ n = txs2.n; carray = nullptr;  tex = txs2.tex; }

	void copyTo(T *data) {
		if(data != nullptr && carray != nullptr) copy3D(data,cudaMemcpyHostToDevice);
		return;
	}

	void copyFrom(T *data) {
		if(data != nullptr && carray != nullptr) copy3D(data,cudaMemcpyDeviceToHost);
		return;
	}

	~txs3D() {
		if(carray != nullptr) {
			if(tex != 0) cx::ok(cudaDestroyTextureObject(tex));
			cx::ok(cudaFreeArray(carray));
		}
	}
};  // end txs3D

}  // end namespace cx
// end file cxtextures.h


