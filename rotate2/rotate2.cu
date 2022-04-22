// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 5.3 rotate2 using textures
// 
// RTX 2070
// C:\bin\rotate2.exe data\ives512.raw data\test.raw 512 512  1.0 10000
// file data\ives512.raw read
// file data\test.raw written
// rotate2 iterations 10000 time 70.429 ms
// 
// RTX 3080
// C:\bin\rotate2.exe data\ives512.raw data\test.raw 512 512  1.0 10000
// file data\ives512.raw read
// file data\test.raw written
// rotate2 iterations 10000 time 41.244 ms

#include "cx.h"
#include "helper_math.h" 
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxtextures.h"  // add cx texture support

// NB this kernel uses uchar textures and is not templated 
__global__ void rotate2(r_Ptr<uchar> b,cudaTextureObject_t atex,float angle,int nx,int ny)
{
	cint x = blockIdx.x*blockDim.x + threadIdx.x;
	cint y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x >= nx || y >= ny) return; // Check if within image bounds

	float xt = x - nx/2.0f;  // translate to make the centre of the 
	float yt = y - ny/2.0f;  // image the centre of rotation

	float xr =  xt*cosf(angle)+ yt*sinf(angle) + nx/2.0f;  // rotate and move origin back
	float yr = -xt*sinf(angle)+ yt*cosf(angle) + ny/2.0f;  // to (0,0) corner of image

	// perform bilinear interpolation by texture lookup
	// NB this texture lookup returns floats in range [0.0,1.0]
	b[y*nx+x] = (uchar)(255*tex2D<float>(atex,xr+0.5f,yr+0.5f));
}

int main(int argc,char *argv[])
{
	if(argc <3){
		printf("usage rotate2 <infile> <outfile> nx|512 ny|512 angle|30 iterations|1");
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
	thrustDvec<uchar> dev_b(size);

	if(cx::read_raw(argv[1],a.data(),size)) return 1;
	
	// copy source image to texture atex NOT thrust device array
	int2 nxy ={nx,ny};
	cx::txs2D<uchar> atex(nxy,a.data(),   // NB pass host array
		cudaFilterModeLinear,             // Enable fast linear interpolation
		cudaAddressModeBorder,            // set out of range pixels to zero
		cudaReadModeNormalizedFloat,      // return floats in [0,1]
		cudaCoordNatural);                // coords in [0,nx] & [0,ny]

	dim3 threads ={16,16,1};
	dim3 blocks ={(uint)(nx+15)/16,(uint)(ny+15)/16,1};
	cx::timer tim;
	for(int k=0;k<iter;k++){
		rotate2<<<blocks,threads>>>(dev_b.data().get(),atex.tex,angle,nx,ny);
	}

	cx::ok(cudaGetLastError());
	cx::ok(cudaDeviceSynchronize());
	double t1 = tim.lap_ms();

	b = dev_b; // get results
	cx::write_raw(argv[2],b.data(),size);
	printf("rotate2 iterations %d time %.3f ms\n",iter,t1);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}