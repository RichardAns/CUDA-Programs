// program ompsum, example 1.2
// 1.2 ompsum
//
// RTX 2070 
// C:\bin\ompsum.exe 1000000 1000 4
// omp sum = 1.9999999978,steps 1000000 terms 1000 time 648.783 ms
// C:\bin\ompsum.exe 1000000 1000 8
// omp sum = 1.9999999978,steps 1000000 terms 1000 time 491.938 ms
// 
// RTX 3080
// C:\bin\ompsum.exe 1000000 1000 4
// omp sum = 1.9999999978,steps 1000000 terms 1000 time 272.854 ms
// C:\bin\ompsum.exe 1000000 1000 8
// omp sum = 1.9999999978,steps 1000000 terms 1000 time 148.224 ms
// C:\bin\ompsum.exe 1000000 1000 10
// omp sum = 1.9999999978,steps 1000000 terms 1000 time 140.529 ms
// C:\bin\ompsum.exe 1000000 1000 20
// omp sum = 1.9999999978,steps 1000000 terms 1000 time 90.348 ms

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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
	int steps   = (argc > 1) ? atoi(argv[1]) : 10000000; // get command
	int terms   = (argc > 2) ? atoi(argv[2]) : 1000;     // line arguments
	int threads = (argc > 3) ? atoi(argv[3]) : 4;        // omp threads
	double pi = 3.14159265358979323;
	double step_size = pi / (steps-1); // NB n-1 steps between n points
	cx::timer tim;
	double omp_sum = 0.0;

	omp_set_num_threads(threads);                   // OpenMP 
    #pragma omp parallel for reduction (+:omp_sum)  // OpenMP
	for(int step = 0; step < steps; step++){
		float x = step_size*step;
		omp_sum += sinsum(x,terms);   // get sum of Taylor series
	}
	double cpu_time = tim.lap_ms(); // get elapsed time
	// Trapezoidal Rule correction for end points
	omp_sum -= 0.5*(sinsum(0.0,terms)+sinsum(pi,terms));
	omp_sum *= step_size;
	printf("omp sum = %.10f,steps %d terms %d time %.3f ms\n",
		omp_sum,steps,terms,cpu_time);
	return 0;
}
