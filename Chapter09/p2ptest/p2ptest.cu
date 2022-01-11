// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 9.2 p2ptest  NB this example has not been tested
//
// example closely based on simpleP2P example from NVIDIA CUDA SDK.
//
#include "cx.h"
#include <vector>

// very simple test kernel dst = 2*src
__global__ void p2ptest(float *src,float *dst,int size) {
	for(int id = blockIdx.x*blockDim.x+threadIdx.x; id<size; 
		               id += blockDim.x*gridDim.x ) dst[id] = 2.0f*src[id];
}

int main(int argc,char *argv[])
{
	int blocks = (argc > 1) ? atoi(argv[1]) : 256;
	int threads =(argc > 2) ? atoi(argv[2]) : 256;
	uint dsize = (argc > 3) ? 1 << atoi(argv[3]) : 1 << 28;
	uint bsize = dsize*sizeof(float);  // buffer size in bytes 

	int ngpu = 0; cudaGetDeviceCount(&ngpu);
	if(ngpu <2) { printf("Need 2 GPUs to run p2p, found %d\n",ngpu); return 1; }

	int gpu1 = 0;  // Just use first 2 devices
	int gpu2 = 1;

	// check for p2p access
	int p2p_1to2; cx::ok( cudaDeviceCanAccessPeer(&p2p_1to2,gpu1,gpu2) ); 
	int p2p_2to1; cx::ok( cudaDeviceCanAccessPeer(&p2p_2to1,gpu2,gpu1) );
	if(p2p_1to2 == 0 || p2p_2to1 == 0) return 1;

	cudaSetDevice(gpu1); cudaDeviceEnablePeerAccess(gpu2,0); // gpu2 can p2p with gpu1
	cudaSetDevice(gpu2); cudaDeviceEnablePeerAccess(gpu1,0); // gpu1 can p2p with gpu2

	// allocate buffer
	float *host_buf; cudaMallocHost(&host_buf,bsize); // Automatically portable with UVA

	cudaSetDevice(gpu1);
	float *gpu1_buf; cudaMalloc(&gpu1_buf,bsize);   // allocate on gpu1
	cudaSetDevice(gpu2);
	float *gpu2_buf; cudaMalloc(&gpu2_buf,bsize);   // allocate on gpu2

	for(uint k=0;k<dsize;k++) host_buf[k] = k;       // fillhost buffer

	cudaSetDevice(gpu1);
	cudaMemcpy(gpu1_buf,host_buf,bsize,cudaMemcpyDefault);   // host => gpu1
	p2ptest<<<blocks,threads>>>(gpu1_buf, gpu2_buf, dsize);  // gpu1 => gpu2 using best route

	//
	// use results and/or do other host work here
	//

	// tidy up
	// disable p2p access
	cudaSetDevice(gpu1); cudaDeviceDisablePeerAccess(gpu1);
	cudaSetDevice(gpu2); cudaDeviceDisablePeerAccess(gpu2);
	// free memory
	cudaFreeHost(host_buf);
	cudaSetDevice(gpu1); cudaFree(gpu1_buf);
	cudaSetDevice(gpu2); cudaFree(gpu2_buf);


	return 0;
}
