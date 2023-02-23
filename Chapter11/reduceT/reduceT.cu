// program reduceT example 11.3 and 11.4

// RTX 2070
// C:\Users\Richard\OneDrive\toGit2>bin\reduceT.exe 1000 28 256 256
// generation time 10256.541 ms
// reduceT sums host 134217190.1 TC 134215272.0  gpu 134217184.0
// reduceT times host (1 call) 614.057 TC 1265.159 gpu 1260.436 ms
//
// RTX 3080
// c:\Users\Richard\OneDrive\toGit2>bin\reduceT.exe 1000 28 256 256
// generation time 8622.503 ms
// reduceT sums host 134217190.1 TC 134215272.0  gpu 134217184.0
// reduceT times host (1 call) 534.644 TC 1580.658 gpu 1386.739 ms
// 
// NB for these timings the GPU calculation uses FP16 values and takes
// about the same time as the TC version. This differers from the book where 
// FP32 values were used and the gpu calculation took twice as long as the TC 
// version. These differences are explained by memory access times.

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "helper_math.h"
#include <random>
#include "mma.h"   // this for TC support

namespace cg = cooperative_groups;
using namespace nvcuda;  // this for mma functions

//  best fp32 reduction from chapter 2
__global__ void reduce7_vl(r_Ptr<float> sums,cr_Ptr<float> data,int n)
{
	// This kernel assumes the array sums is set to zeros on entry
	// also blockSize is multiple of 32 (should always be true)
	auto grid =  cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp =  cg::tiled_partition<32>(block);

	float4 v4 ={0.0f,0.0f,0.0f,0.0f};  // accumulate thread sums in register variable v
	for(int tid = grid.thread_rank(); tid < n/4; tid += grid.size()) {
		v4 += reinterpret_cast<const float4 *>(data)[tid];
	}
	float v =  v4.x + v4.y + v4.z + v4.w;
	warp.sync();

	v += warp.shfl_down(v,16);  // warp level
	v += warp.shfl_down(v,8);  // reduce only
	v += warp.shfl_down(v,4);  // here
	v += warp.shfl_down(v,2);
	v += warp.shfl_down(v,1);  //atomic add sums over blocks
	if(warp.thread_rank()==0) atomicAdd(&sums[block.group_index().x],v);
}

// example 11.4 fp16 version of reduce_vl
// this verssion differs from the book in that we use intrinsics for type half2
// We still add pairs of fp16 as fp16 but then sum these pair sums as fp32. 
__global__ void reduce_half_vl(r_Ptr<float> sums,cr_Ptr<half> data,int n)
{
	// This kernel assumes the array sums is set to zeros on entry
	// also blockSize is multiple of 32 (should always be true)
	auto grid =  cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp =  cg::tiled_partition<32>(block);

	float v = 0.0f;
	half2 v4[4];   
	for(int tid = grid.thread_rank(); tid < n/8; tid += grid.size()) {
		reinterpret_cast<int4 *>(v4)[0] = reinterpret_cast<const int4 *>(data)[tid];
		half2 t = __hadd2(v4[0],v4[1]);             // t.x = v4[0].x+v4[1].x, t.y= v4[0].y+v4[1].y
		v += __half2float(t.x) + __half2float(t.y); // v   = t.x+t.y as fp32
		t =  __hadd2(v4[2],v4[3]);                  // t.x = v4[2].x+v4[3].x, t.y= v4[2].y+v4[3].y
		v += __half2float(t.x) + __half2float(t.y); // v  += t.x+t.y as fp32 
	}
	warp.sync();

	v += warp.shfl_down(v,16); // warp level
	v += warp.shfl_down(v,8);  // reduce only
	v += warp.shfl_down(v,4);  // here
	v += warp.shfl_down(v,2);
	v += warp.shfl_down(v,1);  //atomic add sums over blocks
	if(warp.thread_rank()==0) atomicAdd(&sums[block.group_index().x],v);
}

// TC based fp16 reduction notice we use conventional reduction
// for the second step summing the first row 
__global__ void reduceT(r_Ptr<float> sums,cr_Ptr<half> data,int n)
{
	extern __shared__ float fs[][256];   // one tile for each warp

	auto grid  = cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp  = cg::tiled_partition<32>(block);

	int tid    = grid.thread_rank();     // thread rank in grid
	int wid    = warp.thread_rank();     // thread rank in warp
	int wb     = warp.meta_group_rank(); // warp rank in block
	int wpoint = (tid/32)*256;           // warp offset in data
	int wstep  = grid.size()*8;          // total warps*256

	// Declare the fragments
	wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;  // A 16x16 matrix of 1's
	wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;  // B data 
	wmma::fragment<wmma::accumulator,16,16,16,float>             c_frag;  // C accumulator matrix
	wmma::fill_fragment(c_frag,0.0f);       // C = 0
	wmma::fill_fragment(a_frag,(half)1.0);  // A = 1

	// stream data through tensor cores, each warp handles 256 values
	while(wpoint < n){
		wmma::load_matrix_sync(b_frag,&data[wpoint],16); // load B as 16x16 tile
		wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);     // C = A*B + C where A is all 1's
		wpoint += wstep;  // warp linear address here
	}
	wmma::store_matrix_sync(fs[wb],c_frag,16,wmma::mem_row_major); // copy C to fs as fp32

	// reduce first row of fs which holds column sums
	float v = fs[wb][wid];
	v += warp.shfl_down(v,8);  //
	v += warp.shfl_down(v,4);  // row sum
	v += warp.shfl_down(v,2);  // here
	v += warp.shfl_down(v,1);  //
	if(wid==0) atomicAdd(&sums[block.group_index().x],v); // add per warp row sums here
}

// alternative version of reduceT which uses TC method for both
// reduction steps. It does not perfrom as well as reduceT
__global__ void reduceTT(r_Ptr<float> sums,cr_Ptr<half> data,int n)
{
	extern __shared__ half  hs[][256];

	auto grid  = cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp  = cg::tiled_partition<32>(block);

	int tid    = grid.thread_rank();     // thread rank in grid
	int wid    = warp.thread_rank();     // thread rank in warp
	int wb     = warp.meta_group_rank(); // warp rank in block
	int wpoint = (tid/32)*256;           // warp offset in data
	int wstep  = grid.size()*8;          // total warps*256

	// Declare the fragments
	wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;  // A 16x16 matrix of 1's
	wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;  // B data 
	wmma::fragment<wmma::accumulator,16,16,16,half>              c_frag;  // C accumulator matrix NB type half
	wmma::fill_fragment(c_frag,0.0f);       // C = 0
	wmma::fill_fragment(a_frag,(half)1.0f);  // A = 1

	// stream data through tensor cores, each warp handles 256 values
	while(wpoint < n){
		wmma::load_matrix_sync(b_frag,&data[wpoint],16); // load B as 16x16 tile
		wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);     // C = A*B + C where A is all 1's
		wpoint += wstep;
	}

	wmma::store_matrix_sync(hs[wb],c_frag,16,wmma::mem_row_major); // copy C to hs as fp16

	// sum rows of hs using TCs. Note we are unable to use the result in c_frag 
	// from previous step directly. Instead we have to copy it to hs and then 
	// back to a_frag. This is because a_frag and c_frag are different types.
	// Probably the overhead of these copies is why simple warp reduction performs better.

	wmma::load_matrix_sync(a_frag,&hs[wb][0],16);                  // load A from hs
	wmma::fill_fragment(b_frag,(half)1.0);                         // B = 1
	wmma::fill_fragment(c_frag,(half)0.0f);                        // C = 0
	wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);                   // C = A*B + C where B is all 1's

	wmma::store_matrix_sync(hs[wb],c_frag,16,wmma::mem_row_major); // copy C to hs, the warp sum is in all elements

	if(wid==0) atomicAdd(&sums[block.group_index().x],(float)hs[wb][0]);
}


int host_mma(float *d,half *a,half *b,float *c,int m,int n,int k)
{
	for(int i = 0; i<m; i++){
		for(int j=0;j<m;j++){
			for(int p=0;p<k;p++) d[n*i+j] += (float)a[k*i+p]*(float)b[n*p+j];
			d[n*i+j] += c[n*i+j];
		}
	}
	return 0;
}

template <typename T> int show_mat(cchar *name,T *a,int m,int n)
{
	printf("%s\n",name);
	for(int i=0; i<n; i++){
		for(int j=0;j<m;j++) printf(" %10.3f",(float)a[n*i+j]);
		printf("\n");
	}
	return 0;
}


int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage reduceT loops|1000 N|24 blocks|256 threads|256\n");
		return 0;
	}

	int nacc =    (argc > 1) ? atoi(argv[1]) : 1000;
	int N       = (argc > 2) ? 1 << atoi(argv[2]) : 1 << 24; // default 2^24
	int blocks  = (argc > 3) ? atoi(argv[3]) : 256;
	int threads = (argc > 4) ? atoi(argv[4]) : 128;  // expect power of 2 in [64,1024]

	thrust::host_vector<half>     data(N);      // data is fp16
	thrust::device_vector<half>   dev_data(N);  // 

	thrust::host_vector<float>    sums(blocks);      // sums are fp32
	thrust::device_vector<float>  dev_sums(blocks);
	thrust::device_vector<float>  dev_tot(1);

	cx::timer tgen;
	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<N; k++) data[k] = fran(gen);
	dev_data = data;  // H2D copy (N words)
	double gen_time = tgen.lap_ms();
	printf("generation time %.3f ms\n",gen_time);
	//-------------------------- host sum -------------------
	cx::timer tim;
	double host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += (double)data[k]; // host reduce!
	double host_time = tim.lap_ms();

	//---------------------------------------------------------
	// tensor core version
	//
	// single call to get sum for correctness check
	reduceT<<<blocks,threads,32*threads>>>(dev_sums.data().get(),dev_data.data().get(),N);
	reduce7_vl<<<1,blocks>>>(dev_tot.data().get(),dev_sums.data().get(),blocks); // non TC reduce for block sum
	cudaDeviceSynchronize();
	double TC_sum = dev_tot[0];

	tim.reset();  // loop for timing test
	for(int acc=0;acc<nacc;acc++){
		// NB shared mem 256x4 bytes per warp, or 8-bytes per thread
		reduceT<<<blocks,threads,32*threads>>>(dev_sums.data().get(),dev_data.data().get(),N);
	}
	cudaDeviceSynchronize();
	cx::ok(cudaGetLastError());
	sums = dev_sums;
	double TC_time = tim.lap_ms();

	//---------------------------------------------------------
	// best gpu reduce here for comparison
	for(int k=0;k<blocks;k++) sums[k] = 0.0f;
	dev_sums = sums;
	dev_tot[0] =  0.0f;

	// this to get sum for comparison
	reduce_half_vl<<<blocks,threads>>>(dev_sums.data().get(),dev_data.data().get(),N);
	reduce7_vl<<<1,blocks>>>(dev_tot.data().get(),dev_sums.data().get(),blocks);
	cudaDeviceSynchronize();
	double gpu_sum = dev_tot[0];

	// this for timing
	tim.reset();
	for(int acc=0;acc<nacc;acc++){
		reduce_half_vl<<<blocks,threads>>>(dev_sums.data().get(),dev_data.data().get(),N);
	}
	cudaDeviceSynchronize();
	cx::ok(cudaGetLastError());

	sums = dev_sums;
	double gpu_time = tim.lap_ms();
	//--------------------------end-------------------------------
	
	printf("reduceT sums host %.1f TC %.1f  gpu %.1f\n",host_sum,TC_sum,gpu_sum);
	printf("reduceT times host (1 call) %.3f TC %.3f gpu %.3f ms\n",host_time,TC_time,gpu_time);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}