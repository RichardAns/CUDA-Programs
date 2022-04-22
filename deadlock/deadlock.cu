// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// deadlock example 3.8

#include "cx.h"


__global__ void deadlock(int gsync,int dolock)
{
	__shared__ int lock;
	if(threadIdx.x == 0) lock = 0;
	__syncthreads();  // normal sync after setting shared memory
	if(threadIdx.x < gsync) {  // group A 
		__syncthreads(); // sync A
		if(threadIdx.x == 0) lock = 1;   // deadlock unless we do this
	}
	else if(threadIdx.x < 2*gsync) { // group B
		__syncthreads(); // sync B
	}
	if(dolock) while(lock != 1); // group C may cause deadlock here
	if(threadIdx.x == 0 && blockIdx.x==0) printf("deadlock OK\n");
}

int main(int argc,char* argv[])
{
	int warps =  (argc > 1) ? atoi(argv[1]) : 3; // 3 warps
	int blocks = (argc > 2) ? atoi(argv[2]) : 1; // 1 block (this does not matter)
	int gsync =  (argc > 3) ? atoi(argv[3]) : 32; // one warp
	int dolock = (argc > 4) ? atoi(argv[4]) : 1; // use lock?               
	printf("about to call\n");
	deadlock<<<blocks,warps*32>>>(gsync,dolock);
	printf("done\n");
	return 0;
}
