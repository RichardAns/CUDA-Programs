// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// gpublas example 2.18 
// the final argument argv[5] controls the use of Tensor Cores.
// 
// RTX 2070
// C:\bin\blasmult.exe 1024 1024 1024 0/1
// A A 1024 x 1024 B 1024 x 1024 gpu time 0.418 ms GFlops 5133.6 GBytes 30801.3 (no TC)
// A 1024 x 1024 B 1024 x 1024 gpu time 0.220 ms GFlops 9741.4 GBytes 58448.3 (with TC)
// 
// RTX 3080
// C:\bin\blasmult.exe 1024 1024 1024 0/1
// A 1024 x 1024 B 1024 x 1024 gpu time 0.168 ms GFlops 12777.6 GBytes 76665.7 (no TC)
// A 1024 x 1024 B 1024 x 1024 gpu time 0.092 ms GFlops 23221.6 GBytes 139329.4 (with TC)

#include "cx.h"
#include "cxtimers.h"
#include <random>
#include "cublas_v2.h"

int main(int argc,char *argv[])
{
	int Arow = (argc > 1) ? atoi(argv[1]) : 1 << 10; // default 2^10
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;
	int useTC = (argc > 4) ? atoi(argv[4]) : 1;   // use TC by default
	int nacc = (argc > 5) ? atoi(argv[5]) : 100;  // for timing

	//printf("params %d %d %d to %d nacc %d\n", Arow, Acol, Bcol, useTC, nacc);

	thrust::host_vector<float>       A(Arow*Acol);
	thrust::host_vector<float>       B(Brow*Bcol);
	thrust::host_vector<float>       C(Crow*Ccol);
	thrust::device_vector<float> dev_A(Arow*Acol);
	thrust::device_vector<float> dev_B(Brow*Bcol);
	thrust::device_vector<float> dev_C(Crow*Ccol);
	thrust::device_vector<float> dev_D(Crow*Ccol);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);
	for(int k = 0; k<Crow*Ccol; k++) C[k] = 0.0f;

	dev_A = A;  // H2D copy
	dev_B = B;  // H2D copy
	dev_C = C;  // clear

	float alpha = 1.0f; // 128th root of 10
	float beta  = 1.0f;
	cublasHandle_t handle; 	cublasCreate(&handle);
	if(useTC != 0) cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH); // optional enable tensor cores
	cx::timer tim;
	for(int k=0;k<nacc;k++){  // C = alpha*(A*B) + beta*C
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,Crow,Ccol,Acol,&alpha,dev_A.data().get(),Arow,dev_B.data().get(),Brow,&beta,dev_C.data().get(),Crow);
	}
	beta = 0.0f;  // D = transpose(C)
	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_T,Crow,Ccol,&alpha,dev_C.data().get(),Crow,&beta,dev_C.data().get(),Crow,dev_D.data().get(),Ccol);
	cudaDeviceSynchronize();
	double t3 = tim.lap_ms()/(double)(nacc);
	C = dev_D; // D2H copy

	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops = flops/(t3*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term
	if(useTC==0) printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.1f GBytes %.1f (no TC)\n",Arow,Acol,Brow,Bcol,t3,gflops,gbytes);
	else         printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.1f GBytes %.1f (with TC)\n",Arow,Acol,Brow,Bcol,t3,gflops,gbytes);
	return 0;
}
