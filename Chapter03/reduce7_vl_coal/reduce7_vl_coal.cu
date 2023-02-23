// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// reduce7_vl_coal example 3.10
// RTX 2070
// C:\bin\reduce7_vl_coal.exe 26 256 256 1000
// sum of 67108864 numbers: host 33557315.6 56.781 ms GPU 33557317632.0 1.263 ms
// 
// RTX 3080
// C:\bin\reduce7_vl_coal.exe 26 256 256  1000
// sum of 67108864 numbers: host 33557315.6 64.562 ms GPU 33557317632.0 0.850 ms

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include <random>
#include "helper_math.h"

namespace cg = cooperative_groups;

__device__ void reduce7_vl_coal(r_Ptr<float>sums,cr_Ptr<float>data,int n)
{
	// This function assumes that a.size() is power of 2 in [1,32]
	// and that n is a multiple of 4
	auto g = cg::this_grid();
	auto b = cg::this_thread_block();
	auto a = cg::coalesced_threads(); // active threads in warp
	float4 v4 ={0,0,0,0};
	for(int tid = g.thread_rank(); tid < n/4; tid += g.size())
		v4 += reinterpret_cast<const float4 *>(data)[tid];

	float v = v4.x + v4.y + v4.z + v4.w;
	a.sync();
	if(a.size() > 16) v += a.shfl_down(v,16); // NB no new
	if(a.size() >  8) v += a.shfl_down(v,8);  // thread 
	if(a.size() >  4) v += a.shfl_down(v,4);  // divergence
	if(a.size() >  2) v += a.shfl_down(v,2);  // allowed
	if(a.size() >  1) v += a.shfl_down(v,1);  // here               // rank here is within coal group therefore 
	if(a.thread_rank() == 0) atomicAdd(&sums[b.group_index().x],v); // rank==0 OK even for odd only threads
}

__global__ void reduce7_vl(r_Ptr<float> sums,cr_Ptr<float> data,int n)
{
	// This kernel assumes the array sums is set to zeros on entry
	// and n is a multiple of 4.
	auto grid =  cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp =  cg::tiled_partition<32>(block);
	float4 v4 ={0.0f,0.0f,0.0f,0.0f};  // use v4 to read global memory
	for(int tid = grid.thread_rank(); tid < n/4; tid += grid.size())
		v4 += reinterpret_cast<const float4 *>(data)[tid];
	float v =  v4.x + v4.y + v4.z + v4.w;  // accumulate thread sums in v
	warp.sync();
	v += warp.shfl_down(v,16); // |
	v += warp.shfl_down(v,8);  // | warp level
	v += warp.shfl_down(v,4);  // | reduce here
	v += warp.shfl_down(v,2);  // |
	v += warp.shfl_down(v,1);  // |
							   //     use atomicAdd to sum over warps
	if(warp.thread_rank()==0) atomicAdd(&sums[block.group_index().x],v);
}

__global__ void reduce7_coal_even_odd(r_Ptr<float>sumeven,r_Ptr<float>sumodd,cr_Ptr<float>data,int n)
{
	// divergent code here
	if(threadIdx.x%2==0) reduce7_vl_coal(sumeven,data,n);
	else                 reduce7_vl_coal(sumodd,data,n);
}

int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 256;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;  // multiple of 32
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000; // set this to 1 for correct answer or >> 1 for timing tests
	thrust::host_vector<float>    x(N);
	thrust::device_vector<float>  dx(N);
	thrust::device_vector<float>  dy_even(blocks);  // only even elements are used
	thrust::device_vector<float>  dy_odd(blocks);   // only odd elements are used

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<N; k++) x[k] = fran(gen);
	dx = x;  // H2D copy (N words)
	cx::timer tim;
	double host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!
	double t1 = tim.lap_ms();

	tim.reset();
	// NB tacit assumtion that output array preset to zero. There only get correct result
	// if nreps=1. larger values only useful for timing purposes.	
	for(int rep=0;rep<nreps;rep++){
		reduce7_coal_even_odd<<<blocks,threads>>>(dy_even.data().get(),dy_odd.data().get(),dx.data().get(),N);
	}

	// use reduce7_vl for final step
	dx[0] = 0.0f; // clear output buffer
	reduce7_vl<<<1,blocks>>>(dx.data().get(),dy_even.data().get(),blocks);
	reduce7_vl<<<1,blocks>>>(dx.data().get(),dy_odd.data().get(),blocks);
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms()/nreps;

	double gpu_sum = dx[0];  // D2H copy (1 word)
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",N,host_sum,t1,gpu_sum,t2);
	return 0;
}
