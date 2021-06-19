// grid3D example 2.3

#include "cx.h"

__device__  int   a[256][512][512];  // file scope
__device__  float b[256][512][512];  // file scope

__global__ void grid3D(int nx,int ny,int nz,int id)
{
	int x = blockIdx.x*blockDim.x+threadIdx.x; // find (x,y,z) in
	int y = blockIdx.y*blockDim.y+threadIdx.y; // in arrays
	int z = blockIdx.z*blockDim.z+threadIdx.z; // 
	if(x >=nx || y >=ny || z >=nz) return;     // out of range?

	int array_size = nx*ny*nz;
	int block_size = blockDim.x*blockDim.y*blockDim.z;
	int grid_size  =  gridDim.x* gridDim.y* gridDim.z;
	int total_threads = block_size*grid_size;
	int thread_rank_in_block = (threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
	int block_rank_in_grid  =  (blockIdx.z*gridDim.y+ blockIdx.y)*gridDim.x+ blockIdx.x;
	int thread_rank_in_grid = thread_rank_in_block + block_size*block_rank_in_grid;

	// do some work here
	a[z][y][x] = thread_rank_in_grid;
	b[z][y][x] = sqrtf((float)a[z][y][x]);
	if(thread_rank_in_grid == id) {
		printf("array size   %3d x %3d x %3d = %d\n",nx,ny,nz,array_size);
		printf("thread block %3d x %3d x %3d = %d\n",blockDim.x,blockDim.y,blockDim.z,block_size);
		printf("thread  grid %3d x %3d x %3d = %d\n",gridDim.x,gridDim.y,gridDim.z,grid_size);
		printf("total number of threads in grid %d\n",total_threads);
		printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n",z,y,x,a[z][y][x],z,y,x,b[z][y][x]);
		printf("rank_in_block = %d rank_in_grid = %d rank of block_rank_in_grid = %d\n",thread_rank_in_block,thread_rank_in_grid,block_rank_in_grid);
	}
}

int main(int argc,char *argv[])
{
	int id = (argc > 1) ? atoi(argv[1]) : 12345;
	dim3 thread3d(32,8,2); // 32*8*2    = 512
	dim3  block3d(16,64,128); // 16*64*128 = 131072
	grid3D<<<block3d,thread3d>>>(512,512,256,id);
	return 0;
}
