// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// Example 6.1 piH Host Monte Carlo estimate of pi using faster int RNG
// 
// RTX 2070
// C:\bin\piH.exe 1000 123456
// pi = 3.14152040 err -23.0, ntot 1000000000, time 72138.720 ms (float gen)
// 
// RTX 3080
// C:\bin\piH.exe 1000 123456
// pi = 3.14152040 err -23.0, ntot 1000000000, time 64159.500 ms (float gen)

#include "cx.h"  
#include "cxtimers.h"
#include <random>

int main(int argc,char *argv[])
{
	std::random_device rd;  // truly random but slow to generate
	int points = 1000000;   // inner loop 10^6 generations
	int passes =        (argc > 1) ? atoi(argv[1]) : 1;    // outer loop
	unsigned int seed = (argc > 2) ? atoi(argv[2]) : rd(); // seed

	std::default_random_engine gen(seed);
	std::uniform_real_distribution<float>  fdist(0.0,1.0); //in [0.0,1.0) 

	long long pisum = 0;
	cx::timer tim;

	for(int n = 0; n<passes; n++){
		int subtot = 0;
		for(int k = 0; k < points; k++) {
			float x = fdist(gen);  // generate point
			float y = fdist(gen);  // in square
			if(x*x + y*y < 1.0f) subtot++;  // inside circle?
		}
		pisum += subtot; // accumulate int subtotals in long long
	}

	double gen_time = tim.lap_ms();
	double pi = 4.0*(double)pisum /((double)points *(double)passes);
	double frac_error = 1000000.0*(pi- cx::pi<double>)/cx::pi<double>;
	long long ntot = (long long)passes*(long long)points;
	printf("pi = %10.8f err %.1f, ntot %lld, time %.3f ms (float gen)\n",
		pi,frac_error,ntot,gen_time);
	return 0;
}
