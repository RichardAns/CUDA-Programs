// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 4.14 batcher9PT
// 
// RTX 2070
// C:\Users\Richard\OneDrive\toGit2>bin\batcher9PT.exe data\ives512.raw data\test.raw 512 512  10000
// file data\ives512.raw read
// median9PT iterations 10000 time 75.079 ms
// file data\test.raw written
// 
// RTX 3080
// C:\Users\Richard\OneDrive\toGit2>bin\batcher9PT.exe data\ives512.raw data\test.raw 512 512  10000
// file data\ives512.raw read
// median9PT iterations 10000 time 44.322 ms
// file data\test.raw written

#include "cx.h"
#include "cxbinio.h"
#include "cxtimers.h"

template <typename T> __device__ __inline__ void a_less(T &a,T &b)
{
	T temp = a;        // NB this is much faster than using an 
	a = min(a,b);      // if statement on both host and gpu
	b = max(temp,b);   // a ≤ b on exit
}

template <typename T> __host__ __device__ __inline__ void medsort6(T &a1,T &a2,T &a3,T &a4,T &a5,T &a6) {
	a_less(a1,a2); a_less(a3,a4); a_less(a5,a6);
	a_less(a1,a3); a_less(a2,a4);
	a_less(a1,a5); a_less(a4,a6);  // a1 and a6 now min and max of a1-a6
}

template <typename T> __host__ __device__ __inline__ void medsort5(T &a1,T &a2,T &a3,T &a4,T &a5) {
	a_less(a1,a2); a_less(a3,a4);
	a_less(a1,a3); a_less(a2,a4);
	a_less(a1,a5); a_less(a4,a5);  // a1 and a5 now min and max of a1-a5
}
template <typename T> __host__ __device__ __inline__ void medsort4(T &a1,T &a2,T &a3,T &a4) {
	a_less(a1,a2); a_less(a3,a4);
	a_less(a1,a3); a_less(a2,a4); // a1 and a4 now min and max of a1-a4
}
template <typename T> __host__ __device__ __inline__ void medsort3(T &a1,T &a2,T &a3) {
	a_less(a1,a2);
	a_less(a1,a3); a_less(a2,a3); // a1 and a3 now min and max of a1-a3
}

template <typename T> __host__ __device__ __inline__ T batcher9(T a1,T a2,T a3,T a4,T a5,T a6,T a7,T a8,T a9)
{
	// implement modified Batcher network as per figure 4.7
	medsort6<T>(a4,a5,a6,a7,a8,a9); // extremes of {   a4-a9}    
	medsort5<T>(a3,a5,a6,a7,a8);    // extremes of {a3,a5-a8} 
	medsort4<T>(a2,a5,a6,a7);       // extremes of {a2,a5-a7} 
	medsort3<T>(a1,a5,a6);          // extremes of {a1,a5,a6}    

	return a5;
}

__global__ void batcher9PT(cr_Ptr<uchar> a,r_Ptr<uchar> b,int nx,int ny)
{
	__shared__ uchar as[18][66];
	auto idx = [&nx](int y,int x){ return y*nx+x; };

	int x0 = blockIdx.x*64; int y0 = blockIdx.y*16;     // (y0,x0) origin of tile in a
	int xa = x0+threadIdx.x*4; int ya = y0+threadIdx.y; // (ya,xa) index in a 
	int x = threadIdx.x*4 + 1; int y = threadIdx.y + 1; // (y ,x ) index in shared memory

	const uchar4 a4 = reinterpret_cast<const uchar4 *>(a)[idx(ya,xa)/4];
	as[y][x] = a4.x; as[y][x+1] = a4.y; as[y][x+2] = a4.z; as[y][x+3] = a4.w;

	if(y==1){  // warp 0  copy top (y0-1) row to halo
		int ytop = max(0,y0-1);
		as[0][x] = a[idx(ytop,xa)]; as[0][x+1] = a[idx(ytop,xa+1)];
		as[0][x+2] = a[idx(ytop,xa+2)]; as[0][x+3] = a[idx(ytop,xa+3)];
		if(threadIdx.x==0) {
			int xlft = max(0,x0-1); as[0][0] = a[idx(ytop,xlft)]; // (0,0) corner
			int xrgt = min(nx-1,x0+64); as[0][65] = a[idx(ytop,xrgt)]; // (0,65) corner
		}
		int xlft = max(0,x0-1);
		as[threadIdx.x+1][0] = a[idx(y0+threadIdx.x,xlft)]; // left halo
	}

	if(y==3){  // warp 1 copy bottom row (y0+16) to halo
		int ybot = min(ny-1,y0+16);
		as[17][x] = a[idx(ybot,xa)]; as[17][x+1] = a[idx(ybot,xa+1)];
		as[17][x+2] = a[idx(ybot,xa+2)]; as[17][x+3] = a[idx(ybot,xa+3)];
		if(threadIdx.x==0) {
			int xbot = max(0,x0-1); as[17][0] = a[idx(ybot,xbot)]; // (17,0) corner
			int xrgt = min(nx-1,x0+64); as[17][65] = a[idx(ybot,xrgt)]; // (17,65) corner
		}
		int xrgt = min(nx-1,x0+64);
		as[threadIdx.x+1][65] = a[idx(y0+threadIdx.x,xrgt)]; // right halo
	}
	__syncthreads();
	uchar bout[4];
	for(int k=0;k<4;k++){
		bout[k] = batcher9<uchar>(as[y-1][x-1],as[y-1][x],as[y-1][x+1],as[y][x-1],as[y][x],as[y][x+1],as[y+1][x-1],as[y+1][x],as[y+1][x+1]);
		x++;
	}
	reinterpret_cast<uchar4 *>(b)[idx(ya,xa)/4] = reinterpret_cast<uchar4 *>(bout)[0];
}


int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage median9PT <in file> <outfile> nx|512 ny|512 iter|10000\n");
	}
	thrustHvec<float> c(9);

	uint nx      = (argc > 3) ? atoi(argv[3]) : 512;
	uint ny      = (argc > 4) ? atoi(argv[4]) : 512;
	uint threadx = 16;  // fixed thread blocks size
	uint thready = 16;  // for batcher9PT
	uint iter    = (argc > 5) ? atoi(argv[5]) : 10000; // this for timing
	uint size = nx*ny;

	cx::ok(cudaSetDevice(0));  // Choose which GPU to run on

	thrustHvec<uchar> a(size);
	thrustHvec<uchar> b(size);
	thrustDvec<uchar> dev_a(size);
	thrustDvec<uchar> dev_b(size);

	if(cx::read_raw(argv[1],a.data(),size)) return 1;
	dev_a = a;

	dim3 threads ={threadx,thready,1};
	dim3 blocks64 ={(nx+63)/64,(ny+15)/16,1}; // NB for explict 16x16 thread block size & vl

	cx::timer tim;
	for(uint k=0;k<iter;k++){  //filterered version of a placed in b on each pass
		batcher9PT<<<blocks64,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
	}

	cudaDeviceSynchronize();
	double t1 = tim.lap_ms();
	printf("median9PT iterations %d time %.3f ms\n",iter,t1);

	// save result in b
	b = dev_b;
	cx::write_raw(argv[2],b.data(),size);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}
