#pragma once
// to reduce an array a of size n two calls are reqired
//
// (a)  warp_reduce<<<blocks,threads>>>(b,a,n);
// (b)  warp_reduce<<<1,blocks>>>(c,n,blocks);
// were b is an array of size blocks and c is an array of size 1
// the final sum is in c[0].
//
template <typename T> __global__ void warp_reduce(T *sums,T *data,int N)
{
	// This kernel assumes the array sums is set to zeros on entry
	// also blockSize is multiple of 32 (should always be true)
	auto grid =  cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp =  cg::tiled_partition<32>(block);

	float v = 0.0f;  // accumulate thread sums in register variable v
	for(int tid = grid.thread_rank(); tid < n; tid += grid.size()) v += data[tid];
	warp.sync();

	v += warp.shfl_down(v,16); // warp level
	v += warp.shfl_down(v,8);  // only reduce
	v += warp.shfl_down(v,4);  // here
	v += warp.shfl_down(v,2);  //
	v += warp.shfl_down(v,1);  // then atomicAdd to sum over blocks
	if(warp.thread_rank()==0) atomicAdd(&sums[block.group_index().x],v);
}
