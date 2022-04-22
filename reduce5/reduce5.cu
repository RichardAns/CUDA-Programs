// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 3.1 reduce5
// 
// RTX 2070
// C:\bin\reduce5.exe
// sum of 16777216 numbers: host 8388314.9 14.944 ms GPU 8388314.5 0.215 ms
// 
// RTX 3080
// C:\bin\reduce5.exe
// sum of 16777216 numbers: host 8388314.9 14.830 ms GPU 8388314.5 0.108 ms

#include "cx.h"
#include "cxtimers.h"
#include <random>

// NB the template parameter is the thread-block size
template <int blockSize> __global__ void reduce5(r_Ptr<float> sums,cr_Ptr<float> data,int n)
{
	// This template kernel assumes that blockDim.x = blockSize, 
	// that blockSize is power of 2 between 64 and 1024 
	__shared__ float s[blockSize];
	int id = threadIdx.x;       // rank in block   
	s[id] = 0;
	for(int tid = blockSize*blockIdx.x+threadIdx.x;
		tid < n; tid += blockSize*gridDim.x) s[id] += data[tid];
	__syncthreads();
	if(blockSize > 512 && id < 512 && id+512 < blockSize)s[id] += s[id+512];
	__syncthreads();
	if(blockSize > 256 && id < 256 && id+256 < blockSize)s[id] += s[id+256];
	__syncthreads();
	if(blockSize > 128 && id < 128 && id+128 < blockSize)s[id] += s[id+128];
	__syncthreads();
	if(blockSize >  64 && id <  64 && id+ 64 < blockSize)s[id] += s[id+64];
	__syncthreads();
	if(id < 32) {  //  single warp from here
		s[id]             += s[id + 32]; __syncwarp(); // the syncwarps
		if(id < 16) s[id] += s[id + 16]; __syncwarp(); // are required
		if(id <  8) s[id] += s[id +  8]; __syncwarp(); // for devices of
		if(id <  4) s[id] += s[id +  4]; __syncwarp(); // CC = 7 and above
		if(id <  2) s[id] += s[id +  2]; __syncwarp(); // for CC < 7
		if(id <  1) s[id] += s[id +  1]; __syncwarp(); // they do nothing

		if(id == 0) sums[blockIdx.x] = s[0]; // store block sum
	}
}

__global__ void reduce4(float * y,float * x,int N)
{
	extern __shared__ float tsum[];
	int id = threadIdx.x;
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	tsum[id] = 0.0f;
	for(int k=tid;k<N;k+=stride) tsum[id] += x[k];
	__syncthreads();
	if(id<256 && id+256 < blockDim.x) tsum[id] += tsum[id+256]; __syncthreads();
	if(id<128) tsum[id] += tsum[id+128]; __syncthreads();
	if(id< 64) tsum[id] += tsum[id+ 64]; __syncthreads();
	if(id< 32) tsum[id] += tsum[id+ 32]; __syncthreads();
	// warp 0 only from here
	if(id< 16) tsum[id] += tsum[id+16]; __syncwarp();
	if(id< 8)  tsum[id] += tsum[id+ 8]; __syncwarp();
	if(id< 4)  tsum[id] += tsum[id+ 4]; __syncwarp();
	if(id< 2)  tsum[id] += tsum[id+ 2]; __syncwarp();
	if(id==0)  y[blockIdx.x] = tsum[0]+tsum[1];
}


int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 256;  
	int threads = (argc > 3) ? atoi(argv[3]) : 256;  // must be power of 2
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000; // set this to 1 for correct answer or >> 1 for timing tests
	thrust::host_vector<float>    x(N);
	thrust::device_vector<float>  dx(N);
	thrust::device_vector<float>  dy(blocks);

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
	// this is how to use a value of threads runtime set at runtime 
	// as a compile time template parameter. (Its quite ugly but works)
	if(threads==64)	for(int rep=0;rep<nreps;rep++){
		reduce5<64><<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
	}
	else if(threads==128)	for(int rep=0;rep<nreps;rep++){
		reduce5<128><<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
	}
	else if(threads==256)	for(int rep=0;rep<nreps;rep++){
		reduce5<256><<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
	}
	else if(threads==512)	for(int rep=0;rep<nreps;rep++){
		reduce5<512><<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
	}
	else if(threads==1024)	for(int rep=0;rep<nreps;rep++){
		reduce5<1024><<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
	}
	else {printf("bad value for threads %d\n",threads); return 1; }

	// do single reduce of partial sums here using non-templated kernel for simplicity
	reduce4<<<1,blocks,blocks*sizeof(float)>>>(dx.data().get(),dy.data().get(),blocks);
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms()/nreps;

	double gpu_sum = dx[0];  // D2H copy (1 word)
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",N,host_sum,t1,gpu_sum,t2);
	return 0;
}

