// program matmulTS example 11.2 

// This complete program also includes host and our original tiled gpu matrix multiplation 
// for timing comparisons

// RTX 2070
// C:\bin\matmulTS.exe 100 1024 1024 1024
// blocks 512 threads 256
// A 1024 x 1024 B 1024 x 1024 host 0.000 gpu time 250.169 TCS time 36.931 ms GFlops 858.415 5814.901 speedup 6.77
//
// RTX 3080
// c:\matmulTS.exe 100 1024 1024 1024
// blocks 512 threads 256
// A 1024 x 1024 B 1024 x 1024 host 0.000 gpu time 201.368 TCS time 29.580 ms GFlops 1066.446 7259.991 speedup 6.81

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "helper_math.h"
#include <random>
#include "mma.h"   // this for TC support

namespace cg = cooperative_groups;
using namespace nvcuda;  // this for mma functions

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

// matmulTS example 11.2 (all other code identical to example 11.1)
__global__ void matmulTS(r_Ptr<float> C,cr_Ptr<half> A,cr_Ptr<half> B,int Ay,int Ax,int Bx)
{
	__shared__ half as[256];
	__shared__ half bs[8][256];

	if(blockDim.x != 256) return;  // force 256 threads per block

	// Find row tile and 8 col tiles for this thread block
	int warp = (blockDim.x*blockIdx.x+threadIdx.x)/warpSize;
	int wy = warp/(Bx/16);
	int Atile_pos = wy*16*Ax; // A starts 1 left row at wy 
	int wx = warp%(Bx/16);
	int Btile_pos=  wx*16;    // B starts 8 top cols at wx 

	int wb =  threadIdx.x/32;  // warp rank in block  in [0,255]
	int trw = threadIdx.x%32;  // thread rank in warp 
	int txw = trw%16;          // thread x in warp    in [0,15]
	int tyw = trw/16;          // thread y in warp    in [0, 1]

	int idx = threadIdx.x%16;  // assign 256 threads to cover
	int idy = threadIdx.x/16;  // 16 x 16 x-y values in tile

	// Declare the fragments
	wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;  // A 
	wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;  // B  
	wmma::fragment<wmma::accumulator,16,16,16,float>             c_frag;  // C 
	wmma::fill_fragment(c_frag,0.0f);       // set C = 0

	for(int k=0;k<Ax/16;k++){
		as[idy*16+idx] =  A[Atile_pos+idy*Ax+idx];  // 256 threads used here
		__syncthreads();   // 32 threads fill tile in 8 passes
		for(int p=0;p<8;p++)  bs[wb][p*32+tyw*16+txw] = B[p*2*Bx+Btile_pos+tyw*Bx+txw];
		__syncwarp();
		wmma::load_matrix_sync(a_frag,&as[0],16);      // load A as 16x16 tile
		wmma::load_matrix_sync(b_frag,&bs[wb][0],16);  // load B as 16x16 tile	
		wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);   // C = A*B + C
		Atile_pos += 16;     // move along A row
		Btile_pos += 16*Bx;  // move down B cols
	}
	wmma::store_matrix_sync(&C[(wy*Bx+wx)*16],c_frag,Bx,wmma::mem_row_major);
}

// This is a templated version of matmulTS allowing variable thread block sizes.
// The template parameter T is the number of warps per thread block.
template <int T> __global__ void matmulTTS(r_Ptr<float> C,cr_Ptr<half> A,cr_Ptr<half> B,int Ay,int Ax,int Bx)
{
	__shared__ half as[256];
	__shared__ half bs[T][256];

	if(blockDim.x/32 != T) return;  

	int warp = (blockDim.x*blockIdx.x+threadIdx.x)/warpSize;
	int wx = warp%(Bx/16);  // tile in C x [0,T-1] (for Bx >= 16*T)
	int wy = warp/(Bx/16);  // tile in C y same for all

	int wb = threadIdx.x/32;     // warp rank in block  in [0,T-1]
	int trw = threadIdx.x%32;    // thread rank in warp 
	int txw = trw%16;            // thread x in warp    in [0,15]
	int tyw = trw/16;            // thread y in warp    in [0, 1]

	int Atile_pos = wy*16*Ax;   // start left col at wy (same for all warps)
	int Btile_pos=  wx*16;      // start top row at  wx (different for all warps)

	int idx = threadIdx.x%16;  //
	int idy = threadIdx.x/16;

	// Declare the fragments
	wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;  // A 
	wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;  // B  
	wmma::fragment<wmma::accumulator,16,16,16,float>             c_frag;  // C 
	wmma::fill_fragment(c_frag,0.0f);       // set C = 0

	for(int k=0;k<Ax/16;k++){
		if(idy<16) as[idy*16+idx] =  A[Atile_pos+idy*Ax+idx];  // requires T>= 256
		__syncthreads();
		for(int p=0;p<8;p++)  bs[wb][p*32+tyw*16+txw] = B[p*2*Bx+Btile_pos+tyw*Bx+txw];
		__syncwarp();
		wmma::load_matrix_sync(a_frag,&as[0],16);      // load A as 16x16 tile
		//__syncwarp();
		wmma::load_matrix_sync(b_frag,&bs[wb][0],16);  // load B as 16x16 tile	
		//wmma::load_matrix_sync(b_frag,&B[Btile_pos],Bx); // This always fails
		//__syncwarp();
		wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);   // C = A*B + C
		Atile_pos += 16;
		Btile_pos += 16*Bx;
		//__syncwarp();
	}
	wmma::store_matrix_sync(&C[(wy*Bx+wx)*16],c_frag,Bx,wmma::mem_row_major);
}

int main(int argc,char *argv[])
{
	if(argc <2){
		printf("usage matmulT reps|100 size|24 Arow|1024 Acol|1024 Bcol|1024 show|0 use host|0 use gpu|1 use TCS|1 TCS threads|256\n");
		return 0;
	}
	int reps     = (argc > 1) ? atoi(argv[1]) : 100;
	int Arow     = (argc > 2) ? atoi(argv[2]) : 1024; // default 
	int Acol     = (argc > 3) ? atoi(argv[3]) : Arow;
	int Brow     = Acol;
	int Bcol     = (argc > 4) ? atoi(argv[4]) : Brow;
	int show     = (argc > 5) ? atoi(argv[5]) : 0;
	int host     = (argc > 6) ? atoi(argv[6]) : 0;
	int gpu      = (argc > 7) ? atoi(argv[7]) : 1;   // this for exmple 2.16
	int TCS      = (argc > 8) ? atoi(argv[8]) : 256;
	int threadsT = (argc > 9) ? atoi(argv[9]) : 256; //  allow 64, 128, 256, 512 and 1024
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

	dim3 threads ={16,16,1}; // force square
	dim3 blocks ={(Bcol+threads.x-1)/threads.x,(Arow+threads.y-1)/threads.y,1};

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

	
	int blocksT = Arow*Bcol/(8*threadsT);
	tim.reset();
	if(TCS){
		if(threadsT ==256) {
			for(int k=0;k<reps;k++){
				matmulTS<<<blocksT,threadsT>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
			}
		}
		else if(threadsT == 512) {
			for(int k=0;k<reps;k++){
				matmulTTS<16><<<blocksT,threadsT>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
			}
		}
		else if(threadsT == 1024) {
			for(int k=0;k<reps;k++){
				matmulTTS<32><<<blocksT,threadsT>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
			}
		}
		else if(threadsT == 128) {  // this seems optimal on RTX 2070
			for(int k=0;k<reps;k++){
				matmulTTS<4><<<blocksT,threadsT>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
			}
		}
		else if(threadsT == 64) {
			for(int k=0;k<reps;k++){
				matmulTTS<2><<<blocksT,threadsT>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
			}
		}
		else {printf("thread block size %d not permittd\n",threadsT); return 1; }
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
	printf("A %d x %d B %d x %d host %.3f gpu time %.3f TCS time %.3f ms GFlops %.3f %.3f speedup %.2f\n",Arow,Acol,Brow,Bcol,t1,t2,t3,gflops2,gflops3,speedup);

	return 0;
}
