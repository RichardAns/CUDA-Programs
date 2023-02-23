// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// coop3D example 3.2

#include "cooperative_groups.h"
#include "cx.h"

namespace cg = cooperative_groups;

__device__  int   a[256][512][512];  // file scope
__device__  float b[256][512][512];  // file scope

__global__ void coop3D(int nx,int ny,int nz,int id)
{
	auto grid  = cg::this_grid();
	auto block = cg::this_thread_block();

	int x = block.group_index().x*block.group_dim().x+block.thread_index().x; // find (x,y,z) in
	int y = block.group_index().y*block.group_dim().y+block.thread_index().y; // in arrays
	int z = block.group_index().z*block.group_dim().z+block.thread_index().z; // 	
	if(x >=nx || y >=ny || z >=nz) return;     // out of range?

	int array_size = nx*ny*nz;
	int block_size =    block.size();               // threads in one block
	int grid_size  =    grid.size()/block.size();   // blocks in grid
	int total_threads = grid.size();                // threads in whole grid
	int thread_rank_in_block = block.thread_rank();
	int block_rank_in_grid   = grid.thread_rank()/block.size();
	int thread_rank_in_grid  = grid.thread_rank();

	// do some work here
	a[z][y][x] = thread_rank_in_grid;
	b[z][y][x] = sqrtf((float)a[z][y][x]);
	if(thread_rank_in_grid == id) {
		printf("array size   %3d x %3d x %3d = %d\n",nx,ny,nz,array_size);
		printf("thread block %3d x %3d x %3d = %d\n",blockDim.x,blockDim.y,blockDim.z,block_size);
		printf("thread  grid %3d x %3d x %3d = %d\n",gridDim.x,gridDim.y,gridDim.z,grid_size);
		printf("total number of threads in grid %d\n",total_threads);
		printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n",z,y,x,a[z][y][x],z,y,x,b[z][y][x]);
		printf("for thread with 3D-rank %d 1D-rank %d block rank in grid %d\n",thread_rank_in_grid,thread_rank_in_block,block_rank_in_grid);
	}
}

int main(int argc,char *argv[])
{
	int id = (argc > 1) ? atoi(argv[1]) : 12345;
	dim3 thread3d(32,8,2); // 32*8*2    = 512
	dim3  block3d(16,64,128); // 16*64*128 = 131072
	coop3D<<<block3d,thread3d>>>(512,512,256,id);
    cudaDeviceSynchronize(); // necessary in Linux to see kernel printf
	return 0;
}
