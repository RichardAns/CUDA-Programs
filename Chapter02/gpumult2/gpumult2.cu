// example 2.15 gpumult1 GPU simple matrix multiply one thread per output element with restrict
// introduces lambda function for 2D arressing

#include "cx.h"
#include "cxtimers.h"
#include <random>

__global__ void gpumult2(r_Ptr<float> C,cr_Ptr<float> A,cr_Ptr<float> B,int Ay,int Ax,int Bx)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;  // col index j
	int i = blockIdx.y*blockDim.y + threadIdx.y;  // row index i
	if(j >= Ay || i >= Bx) return;

	auto idx = [&Bx](int i,int j){ return i*Bx+j; }; // lambda function

	// this the the closest we can get to using convential mathmatical notation
	// while retaining the speed advantage of restrict.
	C[idx(i,j)] = 0.0;
	for(int k=0;k<Ax;k++) C[idx(i,j)] += A[idx(i,k)]*B[idx(k,j)];
}

int main(int argc,char *argv[])
{
	int nacc = (argc > 1) ? atoi(argv[1]) : 100;
	int Arow = (argc > 2) ? atoi(argv[2]) : 1024; // default 2^10
	int Acol = (argc > 3) ? atoi(argv[3]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 4) ? atoi(argv[4]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;

	uint tilex = (argc > 5) ? atoi(argv[5]) : 32;  // thread-block x
	uint tiley = (argc > 6) ? atoi(argv[6]) : 8;   // thread-block y

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
		gpumult2<<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
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
