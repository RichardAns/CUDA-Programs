// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 6.6 piH6 cuRand Host API cudaMemcpyAsync

#include "cx.h"  
#include "cxtimers.h"
#include "curand.h"
#include <random>

void sum_part(cr_Ptr<float> rnum,int tries,long long &pisum)
{
	unsigned int sum = 0;
	for(int i=0;i<tries;i++){
		float x = rnum[i*2];
		float y = rnum[i*2+1];
		if(x*x + y*y < 1.0f) sum++;
	}
	pisum += sum;
}

int main(int argc,char *argv[])
{
	std::random_device rd;
	int points = 1000000;
	int passes =        (argc > 1) ? atoi(argv[1]) : 1;
	unsigned int seed = (argc > 2) ? atoi(argv[2]) : rd();

	long long pisum = 0;

	int bsize = points*2*sizeof(float);
	float *a;       cudaMallocHost(&a,bsize);   // host buffers a and b
	float *b;       cudaMallocHost(&b,bsize);   // in pinned memory
	float *dev_rdm; cudaMalloc(&dev_rdm,bsize); // single device buffer

	cudaEvent_t copydone; cudaEventCreate(&copydone);  // CUDA event

	cx::timer tim;   // overall time

	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);

	//curandSetGeneratorOrdering(gen,CURAND_ORDERING_PSEUDO_SEEDED);
	curandSetPseudoRandomGeneratorSeed(gen,seed);
	curandGenerateUniform(gen,dev_rdm,points*2);
	cudaMemcpy(a,dev_rdm,bsize,cudaMemcpyDeviceToHost);  // get 1st block in a 

	for(int k = 0; k < passes; k++) {
		curandGenerateUniform(gen,dev_rdm,points*2);
		cudaMemcpyAsync(b,dev_rdm,bsize,cudaMemcpyDeviceToHost); // async copy to b
		cudaEventRecord(copydone,0);
		cudaEventQuery(copydone);  // WHY DO I NEED THIS event with streams???????
		sum_part(a,points,pisum);    //  process a while b downloading
		std::swap(a,b);
		cudaStreamWaitEvent(0,copydone,0);
	}
	double t1 = tim.lap_ms();

	double pi = 4.0*(double)pisum / ((double)points*(double)passes);
	long long ntot = passes*points;
	double frac_error = 1000000.0*(pi - cx::pi<double>)/cx::pi<double>; // error ppm
	printf("pi = %10.8f err %.1f, ntot %lld, time %.3f ms\n",pi,frac_error,ntot,t1);

	// tidy up
	cudaFreeHost(a); cudaFreeHost(b); cudaFree(dev_rdm);
	curandDestroyGenerator(gen);
	return 0;
}
