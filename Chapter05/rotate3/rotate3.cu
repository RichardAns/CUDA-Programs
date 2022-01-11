// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 5.3 rotate3 using textures and normalized coodinates
// 
// RTX 2070
// C:\bin\rotate3.exe data\ives512.raw data\test.raw 512 512 512 512 1.0  1.01 10000
// scale 1.010000 angle 0.017453
// file data\ives512.raw read
// file data\test.raw written
// rotate3 iterations 10000 time 71.757 ms
// 
// RTX 3080
// C:\bin\rotate3.exe data\ives512.raw data\test.raw 512 512 512 512 1.0  1.01 10000
// scale 1.010000 angle 0.017453
// file data\ives512.raw read
// file data\test.raw written
// rotate3 iterations 10000 time 41.102 ms

#include "cx.h"
#include "helper_math.h" 
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxtextures.h"

__global__ void rotate3(r_Ptr<uchar> b, cudaTextureObject_t utex, float angle, int mx, int my, float scale)
{
	cint x = blockIdx.x*blockDim.x + threadIdx.x;
	cint y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x >= mx || y >= my) return; // Check if within image bounds
	auto idx = [&mx](int y,int x){ return y*mx+x; };
	float xt = x - mx/2.0f;  // translate to make the centre of the 
	float yt = y - my/2.0f;  // image the centre of rotation

	float xr =  xt*cosf(angle)+ yt*sinf(angle) + mx/2.0f; // rotate and restore
	float yr = -xt*sinf(angle)+ yt*cosf(angle) + my/2.0f; // image origin
	float xs = (xr+0.5f)*scale/mx-0.5f*scale+0.5f;  // scale and preserve
	float ys = (yr+0.5f)*scale/my-0.5f*scale+0.5f;  // image centre
	// texture lookup using normalised coordinates
	b[idx(y,x)] = (uchar)(255*tex2D<float>(utex,xs,ys));
}


int main(int argc,char *argv[])
{
	if(argc <3){
		printf("usage rotate3 <infile> <outfile> nx|512 ny|512 mx|nx my|ny angle|30 scale|1 iterations|1");
		return 0;
	}
	int nx      = (argc > 3) ? atoi(argv[3]) : 512;
	int ny      = (argc > 4) ? atoi(argv[4]) : nx;
	int mx      = (argc > 5) ? atoi(argv[5]) : nx;
	int my      = (argc > 6) ? atoi(argv[6]) : ny;
	float angle = (argc > 7) ? atoi(argv[7]) : 30.0f;
	float scale = (argc > 8) ? atof(argv[8]) : 1.0f;
	int iter    = (argc > 9) ? atoi(argv[9]) : 1;
	angle *= cx::pi<float>/180.0f;  // to radians

	printf("scale %f angle %f\n",scale,angle);
	scale = 1.0f/scale;  // For friendly user input (i.e. 2 gives zoom by 2 not shrink by 2) 

	int asize = nx*ny;
	int bsize = mx*my;
	thrustHvec<uchar> a(asize);
	thrustHvec<uchar> b(bsize);
	thrustDvec<uchar> dev_b(bsize);

	if(cx::read_raw(argv[1],a.data(),asize)) return 1;

	// copy source image to texture atex NOT thrust device array
	int2 nxy ={nx,ny};
	cx::txs2D<uchar> atex(nxy,a.data(), // NB pass data in host memory
		cudaFilterModeLinear,           // Enable fast linear interpolation
		cudaAddressModeBorder,          // set out of range pixels to zero
		cudaReadModeNormalizedFloat,    // return floats in [0,1)
		cudaCoordNormalized  );         // coords in [0,1.0] [0,1.0]

	dim3 threads ={16,16,1};
	dim3 blocks ={(uint)(nx+15)/16,(uint)(ny+15)/16,1};

	cx::timer tim;
	for(int k=0;k<iter;k++){
		rotate3<<<blocks,threads>>>(dev_b.data().get(),atex.tex,angle,mx,my,scale);
	}

	cx::ok(cudaGetLastError());
	cx::ok(cudaDeviceSynchronize());
	double t1 = tim.lap_ms();

	b = dev_b; // get results
	cx::write_raw(argv[2],b.data(),bsize);
	printf("rotate3 iterations %d time %.3f ms\n",iter,t1);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}