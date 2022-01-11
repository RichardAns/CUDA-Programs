// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 6.3 piOMP
// NB this example requires OMP support (in VS properties C++ -> Language _> Open MP Support-> yes)
// 
// RTX 2070
// C:\bin\piOMP.exe 1000 123456  8
// pi = 3.14160701 err 4.6, ntot 1000000000, time 2348.383 ms
// 
// RTX 3080
// C:\bin\piOMP.exe 1000 123456 20
// pi = 3.14164101 err 15.4, ntot 1000000000, time 872.135 ms

#include "cx.h"
#include "cxtimers.h"
#include <random>
#include "omp.h"

// initialise generator etc on a per thread basis
long long int sum_part(uint seed,int points,int passes)
{
	int thread = omp_get_thread_num();
	// NB seeds different for each thread depends 
	std::default_random_engine gen(seed+113*thread);
	std::uniform_int_distribution<int> idist(0,2147483647);
	double idist_scale = 1.0/2147483647.0;

	long long int pisum = 0;
	for(int n = 0; n<passes; n++){
		int subtot = 0;
		for(int k = 0; k < points; k++) {
			float x = idist_scale*idist(gen);
			float y = idist_scale*idist(gen);
			if(x*x + y*y < 1.0f) subtot++;  // inside circle?
		}
		pisum += subtot;
	}
	return pisum;
}

int main(int argc,char *argv[])
{
	std::random_device rd;  // truely random but slow to generate
	int points = 1000000;
	int passes =        (argc > 1) ? atoi(argv[1]) : 1;
	unsigned int seed = (argc > 2) ? atoi(argv[2]) : rd();
	int omp_threads =   (argc > 3) ? atoi(argv[3]) : 4;

	omp_set_num_threads(omp_threads);
	long long int pisum = 0;

	cx::timer tim;
#pragma omp parallel for reduction (+:pisum)
	for(int k = 0; k < omp_threads; k++) {
		pisum += sum_part(seed,points,passes/omp_threads);
	}
	double gen_time = tim.lap_ms();

	double pi = 4.0*(double)pisum /((double)(passes)*(double)points);
	double frac_error = 1000000.0*(pi-cx::pi<double>)/cx::pi<double>;
	long long ntot = (long long)(passes)*(long long)points;
	printf("pi = %10.8f err %.1f, ntot %lld, time %.3f ms\n",
		pi,frac_error,ntot,gen_time);
	return 0;
}
