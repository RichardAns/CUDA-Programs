// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 this code is licensed under CC BY-NC 4.0 for non-commercial use
// The code may be freely changed but please retain an acknowledgement

// 1.1 cpusum
// 
// RTX 2070
// C:\bin\cpusum.exe 1000000 1000
// cpu sum = 1.9999999978,steps 1000000 terms 1000 time 1966.554 ms
// 
// RTX 3080
// C:\bin\cpusum.exe 1000000 1000
// cpu sum = 1.9999999978,steps 1000000 terms 1000 time 1085.465 ms

#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"

inline float sinsum(float x,int terms) // sin(x) = x - x^3/3! + x^5/5! ...
{
	float term = x;    // first term of series
	float sum  = term; // sum of terms so far
	float x2   = x*x;
	for(int n = 1; n < terms; n++){
		term *= -x2 / (float)(2*n*(2*n+1)); // get next term from previous
		sum += term;                        // e.g. x^5/5! = (x^3/3!)*(x^2)/(4*5)
	}
	return sum;
}
int main(int argc,char *argv[])
{
	int steps = (argc > 1) ? atoi(argv[1]) : 10000000; // get command
	int terms = (argc > 2) ? atoi(argv[2]) : 1000;     // line arguments
	double pi = 3.14159265358979323;
	double step_size = pi / (steps-1); // NB n-1 steps between n points
	cx::timer tim;
	double cpu_sum = 0.0;
	for(int step = 0; step < steps; step++){
		float x = (float)( step_size*step );
		cpu_sum += sinsum(x,terms);   // get sum of Taylor series
	}
	double cpu_time = tim.lap_ms(); // get elapsed time
	// Trapezoidal Rule correction for end points
	cpu_sum -= 0.5f*(sinsum(0.0,terms)+sinsum(pi,terms));
	cpu_sum *= step_size;
	printf("cpu sum = %.10f,steps %d terms %d time %.3f ms\n",
		cpu_sum,steps,terms,cpu_time);
	return 0;
}