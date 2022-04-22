// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 5.1 rotate1  This version uses bilinear interpolation on GPU
// 
// RTX 2070
// C:\bin\rotate1.exe data\ives512.raw data\test.raw 512 512 1.0 10000
// file data\ives512.raw read
// file data\test.raw written
// rotate1 iterations 10000 time 83.691 ms
// 
// RTX 3080
// C:\bin\rotate1.exe data\ives512.raw data\test.raw 512 512 1.0 10000
// file data\ives512.raw read
// file data\test.raw written
// rotate1 iterations 10000 time 47.140 ms 

#include "cx.h"
#include "helper_math.h"  // for lerp
#include "cxtimers.h"
#include "cxbinio.h"

// this version assumes (x,y) are coodinates of pixel CENTRE
template <typename T> __host__ __device__ T bilinear(cr_Ptr<T> a,float x,float y,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; };
	if(x < -1.0f ||  x >= nx || y < -1.0f || y >= ny) return (T)0;

	float x1 = floorf(x-0.5f); // (x,y) is the
	float y1 = floorf(y-0.5f); // pixel CENTRE
	float ax = x - x1 - 0.5f;  // gap between pixel left sides
	float ay = y - y1 - 0.5f;  // gap between pixel bottoms

	int kx1 = max(0,(int)x1); int kx2 = min(nx-1,kx1+1);
	int ky1 = max(0,(int)y1); int ky2 = min(ny-1,ky1+1);

	float ly1 = lerp(a[idx(ky1,kx1)],a[idx(ky1,kx2)],ax);  // x interp at y1
	float ly2 = lerp(a[idx(ky2,kx1)],a[idx(ky2,kx2)],ax);  // x interp at y2
	return (T)lerp(ly1,ly2,ay);     // y interp of the x interpolated values              
}

// this version assumes (x,y) are coodinates of pixel lower left hand CORNER
template <typename T> __host__ __device__ T bilinear_corner(cr_Ptr<T> a,float x,float y,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; };
	if(x < -1.0f ||  x >= nx || y < -1.0f || y >= ny) return (T)0;

	float x1 = floorf(x); // (x,y) is lower
	float y1 = floorf(y); // left hand CORNER
	float ax = x - x1;   // gap between left sides
	float ay = y - y1;   // gap between bottoms

	int kx1 = max(0,(int)x1); int kx2 = min(nx-1,kx1+1);
	int ky1 = max(0,(int)y1); int ky2 = min(ny-1,ky1+1);

	float ly1 = lerp(a[idx(ky1,kx1)],a[idx(ky1,kx2)],ax);  // x interp at y1
	float ly2 = lerp(a[idx(ky2,kx1)],a[idx(ky2,kx2)],ax);  // x interp at y2
	return (T)lerp(ly1,ly2,ay);     // y interp of the x interpolated values              
}


template <typename T> __global__ void rotate1(r_Ptr<T> b,cr_Ptr<T> a,float angle,int nx,int ny)

{
	cint x = blockIdx.x*blockDim.x + threadIdx.x;
	cint y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x >= nx || y >= ny) return; // Check if within image bounds
	auto idx = [&nx](int y,int x){ return y*nx+x; };

	float xt = x - nx/2.0f;  // translate to make the centre of the 
	float yt = y - ny/2.0f;  // image the centre of rotation
	float xr =  xt*cosf(angle)+ yt*sinf(angle) + nx/2.0f;
	float yr = -xt*sinf(angle)+ yt*cosf(angle) + ny/2.0f;

	// choose one of these interpoalaton functions
	b[idx(y,x)] = bilinear(a,xr,yr,nx,ny);
	//b[idx(y,x)] = bilinear_corner(a,xr,yr,nx,ny);
}

int main(int argc,char *argv[])
{
	if(argc <3){
		printf("usage rotate1 <infile> <outfile> nx|512 ny|512 angle|30 iterations|1");
		return 0;
	}
	int nx      = (argc > 3) ? atoi(argv[3]) : 512;
	int ny      = (argc > 4) ? atoi(argv[4]) : nx;
	float angle = (argc > 5) ? atoi(argv[5]) : 30.0f;
	int iter    = (argc > 6) ? atoi(argv[6]) : 1;
	angle *= cx::pi<float>/180.0f;  // to radians
	int size = nx*ny;

	thrustHvec<uchar> a(size);
	thrustHvec<uchar> b(size);
	thrustDvec<uchar> dev_a(size);
	thrustDvec<uchar> dev_b(size);

	if(cx::read_raw(argv[1],a.data(),size)) return 1;
	dev_a = a;  // copy to device

	dim3 threads ={16,16,1};
	dim3 blocks ={(uint)(nx+15)/16,(uint)(ny+15)/16,1};
	cx::timer tim;
	for(int k=0;k<iter;k++){
		rotate1<uchar><<<blocks,threads>>>(dev_b.data().get(),dev_a.data().get(),angle,nx,ny);
	}

	cx::ok(cudaGetLastError());
	cx::ok(cudaDeviceSynchronize());
	double t1 = tim.lap_ms();

	b = dev_b; // get results
	cx::write_raw(argv[2],b.data(),size);
	printf("rotate1 iterations %d time %.3f ms\n",iter,t1);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}