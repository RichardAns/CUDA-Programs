// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// Example 6.2 piH2 Host Monte Carlo estimate of pi using faster int RNG
// 
// RTX 2070
// C:\bin\piH2.exe 1000 123456
// pi = 3.14158036 err -3.9, ntot 1000000000, time 9995.125 ms (float gen)
// 
// RTX 3080
// C:\bin\piH2.exe 1000 123456
// pi = 3.14158036 err -3.9, ntot 1000000000, time 7647.633 ms (float gen)

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "cx.h" 
#include "cxtimers.h"

int main(int argc,char *argv[])
{
	std::random_device rd;  // truely random but slow to generate
	int points = 1000000;
	int passes =        (argc > 1) ? atoi(argv[1]) : 1;
	unsigned int seed = (argc > 2) ? atoi(argv[2]) : rd();

	std::default_random_engine gen(seed);
	std::uniform_int_distribution<int>  idist(0,2147483647); // uniform ints
	double idist_scale = 1.0/2147483648.0;

	long long pisum = 0;
	cx::timer tim;

	for(int n = 0; n<passes; n++){
		int subtot = 0;  // ++ faster using int 
		for(int k = 0; k < points; k++) {  // generate
			float x = idist_scale*idist(gen); // uniform floats in [0,1.0]
			float y = idist_scale*idist(gen);
			if(x*x + y*y < 1.0f) subtot++;  // inside circle?
		}
		pisum += subtot; // accumulate int subtotals in long long
	}

	double gen_time = tim.lap_ms();
	double pi = 4.0*(double)pisum /((double)points*(double)passes);
	double frac_error = 1000000.0*(pi- cx::pi<double>)/cx::pi<double>;
	long long ntot = (long long)passes*(long long)points;
	printf("pi = %10.8f err %.1f, ntot %lld, time %.3f ms (float gen)\n",
		pi,frac_error,ntot,gen_time);
	return 0;
}