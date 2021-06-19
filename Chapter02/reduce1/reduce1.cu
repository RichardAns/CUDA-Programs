// example 2.6 reduce1

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

	dev_x = x;  // H2D copy (N words)

	cx::timer tim;
	//double host_sum = 0.0;
	float host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!
	double t1 = tim.lap_ms();

	// simple GPU reduce for N = power of 2  
	tim.reset();
	for(int rep=0;rep<nreps;rep++){
		reduce1<<< blocks,threads >>>(dev_x.data().get(),N);
		reduce1<<< 1,threads >>>(dev_x.data().get(),blocks*threads);
		reduce1<<< 1,1 >>>(dev_x.data().get(),threads);
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms();
	double gpu_sum = dev_x[0];  //D2H copy (1 word)

	printf("reduce1 config %d %d: sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",blocks,threads,N,host_sum,t1,gpu_sum,t2);
	return 0;
}
