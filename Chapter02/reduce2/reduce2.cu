// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.7 reduce2
// 
// RTX 2070
// C:\bin\reduce2.exe
// sum of 16777216 numbers: host 8388314.9 14.107 ms GPU 8388314.5 0.166 ms
// 
// RTX 3080
// C:\bin\reduce2.exe 24 256 256
// sum of 16777216 numbers: host 8388314.9 15.662 ms GPU 8388314.5 0.113 ms

#include "cx.h"
#include "cxtimers.h"
#include <random>

__global__ void reduce2(float *y,float *x,int N)
{
	extern __shared__ float tsum[]; // Dynamically Allocated Shared Mem
	int id = threadIdx.x;
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	tsum[id] = 0.0f;
	for(int k=tid;k<N;k+=stride) tsum[id] += x[k];
	__syncthreads();
	for(int k=blockDim.x/2; k>0; k /= 2){ // power of 2 reduction loop
		if(id<k) tsum[id] += tsum[id+k];
		__syncthreads();
	}
	if(id==0) y[blockIdx.x] = tsum[0]; // store one value per block
}

int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 256;  // power of 2
	int threads = (argc > 3) ? atoi(argv[3]) : 256;
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000; // set this to 1 for correct answer or >> 1 for timing tests
	thrust::host_vector<float>    x(N);
	thrust::device_vector<float>  dx(N);
	thrust::device_vector<float>  dy(blocks);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<N; k++) x[k] = fran(gen);
	dx = x;  // H2D copy (N words)
	cx::timer tim;
	double host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!
	double t1 = tim.lap_ms();

	// simple GPU reduce for any value of N
	tim.reset();
	double gpu_sum = 0.0;
	for(int rep=0;rep<nreps;rep++){
		reduce2<<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
		reduce2<<<     1, blocks, blocks*sizeof(float)>>>(dx.data().get(),dy.data().get(),blocks);
		if(rep==0)  gpu_sum = dx[0];  
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms()/nreps;  // time for one pass here
	//double gpu_sum = dx[0]/nreps;          // D2H copy (1 word) 
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",N,host_sum,t1,gpu_sum,t2);
	return 0;
}
