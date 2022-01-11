// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 9.1 multiGPU 

#include "cx.h"
#include <vector>

__global__ void copydata(float *a,float *b,int n,int gpu)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	while(id < n){
		b[id] = a[id];  // copy a to b
		id += blockDim.x*gridDim.x;
	}
	if(threadIdx.x==0 && blockIdx.x ==0) printf("copydata gpu = %d\n",gpu);
}

int main(int argc,char *argv[])
{
	int blocks = (argc > 1) ? atoi(argv[1]) : 256;
	int threads =(argc > 2) ? atoi(argv[2]) : 256;
	uint dsize = (argc > 3) ? 1 << atoi(argv[3]) : 1 << 28;
	uint bsize = dsize*sizeof(float);  // buffer size in bytes 

	int ngpu = 0;
	cudaGetDeviceCount(&ngpu);
	printf("Number of GPUs on this PC is %d\n",ngpu);

	// these vectors store a set of seperate pointers for each GPU
	std::vector<float *> host_buf;
	std::vector<float *> dev_a;
	std::vector<float *> dev_b;

	for(int gpu=0; gpu<ngpu; gpu++){
		cudaSetDevice(gpu);   // must select a gpu before use
		float *a = (float *)malloc(bsize); host_buf.push_back(a);
		cudaMalloc(&a,bsize); dev_a.push_back(a);
		cudaMalloc(&a,bsize); dev_b.push_back(a);
		for(uint k=0;k<dsize;k++) host_buf[gpu][k] = (float)sqrt((double)k);
		//cudaMemcpy(dev_a[gpu],host_buf[gpu],bsize,cudaMemcpyHostToDevice);
		cudaMemcpy(dev_a[gpu],host_buf[gpu],bsize,cudaMemcpyDefault);
		copydata<<<blocks,threads>>>(dev_a[gpu],dev_b[gpu],dsize,gpu);
	}

	// do host concurrent work here ...
	for(int gpu=0; gpu<ngpu; gpu++){
		cudaSetDevice(gpu);   // select gpu before use
		cudaMemcpy(host_buf[gpu],dev_b[gpu],bsize,cudaMemcpyDefault);
	}
	// do host work on kernel results here ...
	printf("check host %.3f %.3f %.3f %.3f\n",host_buf[0][0],host_buf[0][1],host_buf[0][2],host_buf[0][3]);

	for(int gpu=0; gpu<ngpu; gpu++){ // tidy up
		cudaSetDevice(gpu);          // select gpu before use
		cudaDeviceSynchronize();     // Just in case there is pending work
		free(host_buf[gpu]);
		cudaFree(dev_a[gpu]);
		cudaFree(dev_b[gpu]);
		cudaDeviceReset();    // reset current device. NB no thrust
	}

	return 0;
}
