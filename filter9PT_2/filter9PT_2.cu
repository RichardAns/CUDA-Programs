// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 4.9 filter9PT_2
// 
// RTX 2070
// C:\bin\filter9PT_2.exe data\ives512.raw data\test.raw 512 512 16 16 10000 1.0 1.0 1.0   1.0 1.0 1.0   1.0 1.0 1.0
// file data\ives512.raw read
// filter9PT_2 iterations 10000 time 103.863 ms
// file data\test.raw written
// 
// RTX 3080
// C:\bin\filter9PT_2.exe data\ives512.raw data\test.raw 512 512 16 16 10000 1.0 1.0 1.0   1.0 1.0 1.0   1.0 1.0 1.0
// file data\ives512.raw read
// filter9PT_2 iterations 10000 time 47.575 ms
// file data\test.raw written

#include "cx.h"
#include "cxbinio.h"
#include "cxtimers.h"

// data explicilty in constnat memory must be declared at file scope
// arrays sizes must be known at compile time.
__constant__ float fc[9];  

// filter constants stored in device global memory
__global__ void filter9PT_2(cr_Ptr<uchar> a,r_Ptr<uchar> b,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; };

	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x<0 || y <0 || x >= nx || y >= ny)return;
	int xl = max(0,x-1); int yl = max(0,y-1);
	int xh = min(nx-1,x+1); int yh = min(ny-1,y+1);

	float v = fc[0]*a[idx(yl,xl)] + fc[1]*a[idx(yl,x)] + fc[2]*a[idx(yl,xh)] +
		      fc[3]*a[idx(y,xl)]  + fc[4]*a[idx(y,x)]  + fc[5]*a[idx(y,xh)]  +
		      fc[6]*a[idx(yh,xl)] + fc[7]*a[idx(yh,x)] + fc[8]*a[idx(yh,xh)];

	uint f = (uint)(v+0.5f);
	b[idx(y,x)] = (uchar)min(255,max(0,f)); // b in [0,255]
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage filter9PT_2 <in file> <outfile> nx|512 ny|512 tx|16 ty|16 iter|10000 c[0] ... c[9]\n");
	}
	thrustHvec<float> c(9);

	uint nx      = (argc > 3) ? atoi(argv[3]) : 512;
	uint ny      = (argc > 4) ? atoi(argv[4]) : 512;
	uint threadx = (argc > 5) ? atoi(argv[5]) : 16;
	uint thready = (argc > 6) ? atoi(argv[6]) : 16;
	uint iter    = (argc > 7) ? atoi(argv[7]) : 10000; // this for timing
	for(int k=0;k<9;k++) c[k] = (argc > 8+k) ? atof(argv[8+k]) : 1.0;
	uint size = nx*ny;

	cx::ok(cudaSetDevice(0));  // Choose which GPU to run on

	thrustHvec<uchar> a(size);
	thrustHvec<uchar> b(size);
	thrustDvec<uchar> dev_a(size);
	thrustDvec<uchar> dev_b(size);

	// normalise filter coefficients
	float csum = 0.0f;
	for(int k=0;k<9;k++) csum += c[k];
	if(fabs(csum) > 0.001) for(int k=0;k<9;k++) c[k] /= csum;

	if(cx::read_raw(argv[1],a.data(),size)) return 1;
	cudaMemcpyToSymbol(fc, c.data(),9*sizeof(float));  // replaces dev_c = c
	dev_a = a;

	dim3 threads ={threadx,thready,1};
	dim3 blocks ={(nx+threads.x-1)/threads.x,(ny+threads.y-1)/threads.y,1};

	cx::timer tim;
	for(uint k=0;k<iter;k++){  //filterered version of a placed in b on each pass
		filter9PT_2<<<blocks,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
	}

	cudaDeviceSynchronize();
	double t1 = tim.lap_ms();
	printf("filter9PT_2 iterations %d time %.3f ms\n",iter,t1);

	// save result in b
	b = dev_b;
	cx::write_raw(argv[2],b.data(),size);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}
