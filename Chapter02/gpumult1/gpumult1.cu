// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.14 gpumult1 GPU simple matrix multiply one thread per output element with restrict
// 
// RTX 2070
// C:\bin\gpumult1.exe 1024 1024 1024 32 32
// A 1024 x 1024 B 1024 x 32 gpu time 0.115 ms GFlops 582.046 GBytes 3492.278
// 
// RTX 3080
// C:\bin\gpumult1.exe 1024 1024 1024 32 8
// A 1024 x 1024 B 1024 x 32 gpu time 0.067 ms GFlops 1008.538 GBytes 6051.230

#include "cx.h"
#include "cxtimers.h"
#include <random>

// standard C++ declaration
//__global__ void gpumult1(float * __restrict C, const float * __restrict A,
//	                 const float * __restrict B, int Ay, int Ax, int Bx)

// or use cx defined r_Ptr and cr_Ptr to reduce verbosity
__global__ void gpumult1(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B, int Ay, int Ax, int Bx)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;  // col index j
	int ty = blockIdx.y*blockDim.y + threadIdx.y;  // row index i
	if(ty >= Ay || tx >= Bx) return;

	C[ty*Bx+tx] = 0.0;
	for(int k=0;k<Ax;k++) C[ty*Bx+tx] += A[ty*Bx+k]*B[k*Bx+tx];
}

int main(int argc,char *argv[])
{

	int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // default 2^10
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;

	uint tilex = (argc > 4) ? atoi(argv[4]) : 32;  // thread-block x
	uint tiley = (argc > 5) ? atoi(argv[5]) : 8;   // thread-block y
	int nacc = (argc > 6) ? atoi(argv[6]) : 100;   // for timing

	thrust::host_vector<float>       A(Arow*Acol);
	thrust::host_vector<float>       B(Brow*Bcol);
	thrust::host_vector<float>       C(Crow*Ccol);
	thrust::device_vector<float> dev_C(Crow*Ccol);
	thrust::device_vector<float> dev_A(Arow*Acol);
	thrust::device_vector<float> dev_B(Brow*Bcol);

	// initialise A and B with random numbers
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);

	dev_A = A;  // H2D copy
	dev_B = B;  // H2D copy
	dim3 threads ={tilex,tiley,1};
	dim3 blocks ={(Bcol+threads.x-1)/threads.x,(Arow+threads.y-1)/threads.y,1};

	cx::timer tim;
	for(int k=0;k<nacc;k++){
		gpumult1<<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
	}
	cudaDeviceSynchronize();  // wait for kernel
	double t2 = tim.lap_ms()/(double)(nacc);

	C = dev_C;               // D2H copy
	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops = flops/(t2*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term
	printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.3f GBytes %.3f\n",
		Arow,Acol,Brow,Bcol,t2,gflops,gbytes);
	return 0;
}
