// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 6.5 piH5 cuRand Host API with pinned memory
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
	std::random_device rd;  // truely random but slow to generate
	int points = 1000000;   // points per inner loop iteration
	int passes =        (argc > 1) ? atoi(argv[1]) : 1;
	unsigned int seed = (argc > 2) ? atoi(argv[2]) : rd();

	thrustHvecPin<float> rdm(points*2);  // host pinned memory 
	thrustDvec<float> dev_rdm(points*2); // Device RN buffers

	long long pisum = 0;
	cx::timer tim;
	curandGenerator_t gen;  // setup Host API cuRand generator
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,seed);

	for(int k=0; k<passes; k++) {
		curandGenerateUniform(gen,dev_rdm.data().get(),points*2); // Generate on GPU
		rdm = dev_rdm;                    // Copy GPU => Host
		sum_part(rdm.data(),points,pisum); // and use on Host
	}

	double gen_time = tim.lap_ms();
	double pi = 4.0*(double)pisum /((double)points*(double)passes);
	double frac_error = 1000000.0*(pi- cx::pi<double>)/cx::pi<double>;
	long long ntot = (long long)passes*(long long)points;
	printf("pi = %10.8f err %.1f, ntot %lld, time %.3f ms (float gen)\n",
		pi,frac_error,ntot,gen_time);

	curandDestroyGenerator(gen); // tidy up
	return 0;
}
