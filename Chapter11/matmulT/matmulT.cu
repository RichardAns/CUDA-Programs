// program matmulT example 11.1 

// This complete program also includes host and our original tiled gpu matrix multiplation 
// for timing comparisons
//
// // RTX 2070
// C:\bin\matmulT.exe 100 1024 1024 1024
// blocks 512 threads 256
// A 1024 x 1024 B 1024 x 1024 host 0.000 gpu time 264.978 TC time 56.578 ms GFlops 810.438 3795.609 speedup 4.68
// 
// RTX 3080
// C:\bin\matmulT.exe 100 1024 1024 1024
// blocks 512 threads 256
// A 1024 x 1024 B 1024 x 1024 host 0.000 gpu time 91.002 TC time 18.075 ms GFlops 2359.812 11881.158 speedup 5.03

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "helper_math.h"
#include <random>
#include "mma.h"   // this for TC support

namespace cg = cooperative_groups;
using namespace nvcuda;

// This is simple host version
template <typename T> int host_mult(r_Ptr<float> C,cr_Ptr<T> A,cr_Ptr<T> B,int ay,int ax,int bx)
{
	for(int i=0;i<ay;i++) for(int j=0;j<bx;j++){
		C[i*bx+j] = 0.0f;
		for(int k=0;k<ax;k++) C[i*bx+j] += (float)A[i*ax+k]*(float)B[bx*k+j];
	}
	return 0;
}

// print 2D matrix (if ax < stride only corner of matrix printed)
template <typename T> int show_mat(cchar *name,T *a,int ay,int ax,int stride)
{
	printf("%s\n",name);
	for(int i=0; i<ay; i++){
		for(int j=0;j<ax;j++) printf(" %10.3f",(float)a[stride*i+j]);
		printf("\n");
	}
	return 0;
}

// this is example 2.16
template <typename T> __global__ void gputiled(r_Ptr<float> C,cr_Ptr<T> A,cr_Ptr<T> B,int Ay,int Ax,int Bx)
{
	__shared__ float Atile[16][16];  // tile in A eg [16][16]
	__shared__ float Btile[16][16];  // tile in B eg [16][16]

	int tx = threadIdx.x;             // tile col index j
	int ty = threadIdx.y;             // tile row index i
	int ocx = blockDim.x*blockIdx.x;  // tile x origin in C 
	int ocy = blockDim.y*blockIdx.y;  // tile y origin in C 

	int ax = tx;      // j in first tile on A
	int ay = ocy+ty;  // i in first tile on A
	int bx = ocx+tx;  // j in first tile on B
	int by = ty;	  // i in first tile on B

	float csum = 0.0f;
#pragma unroll 16
	for(int t=0;t<gridDim.x;t++){     // step A tiles along rows of A
		Atile[ty][tx] = A[ay*Ax+ax];  // step B tiles down  cols of B
		Btile[ty][tx] = B[by*Bx+bx];
		__syncthreads();
		for(int k=0;k<16;k++) csum += Atile[ty][k]*Btile[k][tx];
		__syncthreads();
		ax += 16;  // step A tiles along rows of A
		by += 16;  // step B tiles down  cols of B
	}
	C[ay*Bx+bx] = csum; // store complete result
}

// matmulT example 11.1
__global__ void matmulT(r_Ptr<float> C,cr_Ptr<half> A,cr_Ptr<half> B,int Ay,int Ax,int Bx)
{
	int warp = (blockDim.x*blockIdx.x+threadIdx.x)/warpSize; // warp rank in grid

	int cx = warp%(Bx/16);  // (x,y) location of active tile
	int cy = warp/(Bx/16);  // for current warp in C matrix

	int Atile_pos = cy*16*Bx; // start x (row) for first A tile
	int Btile_pos=  cx*16;    // start y (col) for first B tile

	// Declare the fragments as 16 x 16 tiles
	wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;  // A 
	wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;  // B  
	wmma::fragment<wmma::accumulator,16,16,16,float>             c_frag;  // C 
	wmma::fill_fragment(c_frag,0.0f);       // set C = 0
	
	for(int k=0;k<Ax/16;k++){ // accumulate su, of row*column for C tile
		wmma::load_matrix_sync(a_frag,&A[Atile_pos],Ax);  // load A as 16x16 tile
		wmma::load_matrix_sync(b_frag,&B[Btile_pos],Bx);  // load B as 16x16 tile		
		wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);      // C = A*B + C
		Atile_pos += 16;     // step along row of A
		Btile_pos += 16*Bx;  // step down column of B
	}
	wmma::store_matrix_sync(&C[(cy*Bx+cx)*16],c_frag,Bx,wmma::mem_row_major);
}


int main(int argc,char *argv[])
{
	if(argc <2){
		printf("usage matmulT reps|100 size|24 Arow|1024 Acol|1024 Bcol|1024 show|0 host ver|0 gpu ver|1 TC ver|1\n");
		return 0;
	}
	int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // default 
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int show = (argc > 4) ? atoi(argv[4]) : 0;
	int host = (argc > 5) ? atoi(argv[5]) : 0;
	int gpu  = (argc > 6) ? atoi(argv[6]) : 1;   // this for exmple 2.16
	int TC   = (argc > 7) ? atoi(argv[7]) : 1;
	int reps = (argc > 8) ? atoi(argv[8]) : 100;  // for timing
	int Crow = Arow;
	int Ccol = Bcol;

	thrust::host_vector<half>       A(Arow*Acol);
	thrust::host_vector<half>       B(Brow*Bcol);
	thrust::host_vector<float>      C(Crow*Ccol);
	thrust::device_vector<half>  dev_A(Arow*Acol);
	thrust::device_vector<half>  dev_B(Brow*Bcol);
	thrust::device_vector<float> dev_C(Crow*Ccol);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = (half)fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = (half)fran(gen);
	dev_A = A;  // H2D copy
	dev_B = B;  // H2D copy


	double t1=0.0;
	double t2=0.0;
	double t3=0.0;
	//---------------------------- Host version--------------------------
	cx::timer tim;
	if(host){  // just once is enough 
		host_mult(C.data(),A.data(),B.data(),Arow,Acol,Bcol);
		t1 = tim.lap_ms();
		if(show)show_mat("host",C.data(),16,16,Ccol);
	}

	//----------------------- CUDA tiled matrix multiply ----------------------------------
	tim.reset();
	if(gpu){
		dim3 threads ={16,16,1}; // force square
		dim3 blocks ={(Bcol+threads.x-1)/threads.x,(Arow+threads.y-1)/threads.y,1};

		for(int k=0;k<reps;k++){
			gputiled<<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		}
		cudaDeviceSynchronize();
		cx::ok(cudaGetLastError());
		t2 = tim.lap_ms();
		C = dev_C; // D2H copy
		if(show)show_mat("old GPU ",C.data(),16,16,Ccol);
		//cx::write_raw("guptiled.raw",C.data(),Crow*Ccol);
	}
	//---------------------- Tensor core version -------------------------------------------

	int threadsT = 256; // fixed
	int blocksT = Arow*Bcol/(8*threadsT);
	tim.reset();
	if(TC){
		for(int k=0;k<reps;k++){
			matmulT<<<blocksT,threadsT>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		}
		cudaDeviceSynchronize();
		cx::ok(cudaGetLastError());
		t3 = tim.lap_ms();
		C = dev_C; // D2H copy
		if(show)show_mat("new GPU ",C.data(),16,16,Ccol);
	}
	//------------------------------------------------------------------------------

	printf("blocks %d threads %d\n",blocksT,threadsT);

	double flops = 2.0*(double)(reps)*(double)Arow*(double)Acol*(double)Bcol;
	double gflops2 = flops/(t2*1000000.0);
	double gflops3 = flops/(t3*1000000.0);
	double speedup = t2/t3;
	printf("A %d x %d B %d x %d host %.3f gpu time %.3f TC time %.3f ms GFlops %.3f %.3f speedup %.2f\n",Arow,Acol,Brow,Bcol,t1,t2,t3,gflops2,gflops3,speedup);

	return 0;
}
