// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program gpusaxpy  example D.4
//
// RTX 3080 Linux
// ../../Linux/gpusaxpy 21 3000
// gpusaxpy: size 536870912, time 268.861378 ms check 1000187.81250 GFlops 23961.980

#include "cx.h"
#include "cxtimers.h"

__global__ void gpusaxpy(r_Ptr<float> x,cr_Ptr<float> y,float a,int size,int reps)
{
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	while(tid < size){
		for(int k=0;k<reps;k++) x[tid] = a*x[tid]+y[tid];
		tid += gridDim.x*blockDim.x;
	}
}

int main(int argc,char* argv[])
{
	signed int size = (argc > 1) ? 2 << atoi(argv[1]) : 1 << 21;
	int blocks      = (argc > 2) ? atoi(argv[2]) : 288;
	int threads     = (argc > 3) ? atoi(argv[3]) : 256;
	int reps        = (argc > 4) ? atoi(argv[4]) : 100;
	float a = 1.002305238f;  // 1000th root of 10 

	thrustDvec<float> dev_x(size,1.0f);
	thrustDvec<float> dev_y(size,0.0f);

	cx::timer tim;
	gpusaxpy<<<blocks,threads>>>(dev_x.data().get(),dev_y.data().get(),a,size,reps);
	cx::ok(cudaDeviceSynchronize());
	double t1 = tim.lap_ms();
	float gpu_check = dev_x[128];

	double gflops = 2.0*(double)(size)*(double)reps/(t1*1000000);
	printf("gpusaxpy: size %d, time %.6f ms check %10.5f GFlops %.3f\n",size,t1,gpu_check,gflops);

	return 0;

}
