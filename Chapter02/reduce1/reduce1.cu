// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.6 reduce1
// 
// RTX 2070
// C:\bin\reduce1.exe
// reduce1 config 288 256: sum of 16777216 numbers: host 8388314.9 14.261 ms GPU 8388315.5 0.197 ms
// 
// RTX 3080
// C:\bin\reduce1.exe 24 272 256 1000
// reduce1 config 272 256: sum of 16777216 numbers: host 8388314.9 15.546 ms GPU 8388315.0 0.130 ms

#include "cx.h"
#include "cxtimers.h"
#include <random>

__global__ void reduce1(float *x,int N)
{
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	float tsum = 0.0f;
	int stride = gridDim.x*blockDim.x;
	for(int k=tid; k<N; k += stride) tsum += x[k];
	x[tid] = tsum;
}

int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 288;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000;    // set this to 1 for correct answer
	thrust::host_vector<float>    x(N);
	thrust::device_vector<float> dev_x(N);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<N; k++) x[k] = fran(gen);
	//for(int k = 0; k<N; k++) x[k] = 0.5f; // debug accuracy

	dev_x = x;  // H2D copy (N words)

	cx::timer tim;
	double host_sum = 0.0;
	//float host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!
	double t1 = tim.lap_ms();

	// simple GPU reduce for N = power of 2  
	tim.reset();
	double gpu_sum = 0.0;
	for(int rep=0;rep<nreps;rep++){
		reduce1<<< blocks,threads >>>(dev_x.data().get(),N);
		reduce1<<< 1,threads >>>(dev_x.data().get(),blocks*threads);
		reduce1<<< 1,1 >>>(dev_x.data().get(),threads);
		if (rep==0) gpu_sum = dev_x[0];
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms()/nreps;  // time for one pass to compare with host
	//double gpu_sum = dev_x[0];  //D2H copy (1 word) but wrong here for nreps > 1

	printf("reduce1 config %d %d: sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",blocks,threads,N,host_sum,t1,gpu_sum,t2);
	return 0;
}
