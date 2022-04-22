// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// ompsaxpy program example D.3
//
//  avxsaxpy: a*x + y for vectors x and y
//
// This example requires Intel C++ with omp support enabled
//
// if not in the default path required dll files are in
//  C:\Program Files (x86)\
//             IntelSWTools\
//                  parallel_studio_xe_2020.2.899\
//                       compilers_and_libraries_2020\
//                                 windows\redist\intel64\compiler
// RTX 3080 Linux (very fast)
// ../../Linux/ompsaxpy 21 3000
// ompsaxpy: size 4194304, time 106.611439 ms check 1000.09351 GFlops 236.052

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage avxsaxpy <size as power of 2|20> reps|1000\n");
		return 0;
	}

	int size = (argc > 1) ? 2 << atoi(argv[1]) : 1 << 10;
	int reps = (argc > 2) ? atoi(argv[2]) : 100;

	__m256  ma = _mm256_set1_ps(1.002305238f); // 1000th root of 10 

	__m256 *mx = (__m256 *)malloc(sizeof(__m256)*size/8);
	__m256 *my = (__m256 *)malloc(sizeof(__m256)*size/8);

	for(int k=0;k<size/8;k++) mx[k] = _mm256_set1_ps(1.0f);
	for(int k=0;k<size/8;k++) my[k] = _mm256_set1_ps(0.0f);

	cx::timer tim;
#pragma omp parallel for // OpenMP
	for(int k=0;k<size/8;k++){
		for(int i=0;i<reps;i++){
			mx[k] = _mm256_fmadd_ps(ma,mx[k],my[k]); // x = a*x+y
		}
	}
	double t1 = tim.lap_ms();

	float check[8];  _mm256_storeu_ps(check,mx[7]); // get 8 elements u in case check no aligned
	double gflops = 2.0*(double)(size)*(double)reps/(t1*1000000);
	printf("ompsaxpy: size %d, time %.6f ms check %10.5f GFlops %.3f\n",size,t1,check[7],gflops);

	free(mx); free(my);	//tidy up
	return 0;
}