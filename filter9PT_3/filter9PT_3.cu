// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 4.10 filter9PT_3
// 
// RTX 2070
// C:\>bin\filter9PT_3.exe data\ives512.raw data\test.raw 512 512  10000 1.0 1.0 1.0   1.0 1.0 1.0   1.0 1.0 1.0
// file data\ives512.raw read
// filter9PT_3 iterations 10000 time 69.219 ms
// file data\test.raw written
// 
// RTX 3080
// C:\bin\filter9PT_3.exe data\ives512.raw data\test.raw 512 512 10000 1.0 1.0 1.0   1.0 1.0 1.0   1.0 1.0 1.0
// file data\ives512.raw read
// filter9PT_3 iterations 10000 time 41.062 ms
// file data\test.raw written

#include "cx.h"
#include "cxbinio.h"
#include "cxtimers.h"

// data explicilty in conststant memory must be declared at file scope
// arrays sizes must be known at compile time.
__constant__ float fc[9];

// filter constants stored in device global memory
__global__ void filter9PT_3(cr_Ptr<uchar> a,r_Ptr<uchar> b,int nx,int ny)
{
	__shared__ uchar as[18][66];   // 16 x 64 tile plus 1 element wide halo
	auto idx = [&nx](int y,int x){ return y*nx+x; };

	int x0 = blockIdx.x*64; int y0 = blockIdx.y*16; // (y0,x0) tile origin in a
	int xa = x0+threadIdx.x*4; int ya = y0+threadIdx.y; // (ya,xa) index in a 
	int x = threadIdx.x*4 + 1; int y = threadIdx.y + 1; // (y,x) in shared mem

	const uchar4 a4 = reinterpret_cast<const uchar4 *>(a)[idx(ya,xa)/4];
	as[y][x] = a4.x; as[y][x+1] = a4.y; as[y][x+2] = a4.z; as[y][x+3] = a4.w;

	if(y==1){  // warp 0 threads 0-15: copy top (y0-1) row to halo
		int ytop = max(0,y0-1);
		as[0][x]   = a[idx(ytop,xa)]; as[0][x+1] = a[idx(ytop,xa+1)];
		as[0][x+2] = a[idx(ytop,xa+2)]; as[0][x+3] = a[idx(ytop,xa+3)];
		if(threadIdx.x==0) {  // top corners
			int xleft = max(0,x0-1);   as[0][0]= a[idx(ytop,xleft)]; //(0,0) 
			int xright= min(nx-1,x0+64);as[0][65]= a[idx(ytop,xright)];//(0,65)
		}
		int xlft = max(0,x0-1);
		as[threadIdx.x+1][0] = a[idx(y0+threadIdx.x,xlft)]; // left edge halo
	}

	if(y==3){  // warp 1 threads 0-15: copy bottom row (y0+16) to halo
		int ybot = min(ny-1,y0+16);
		as[17][x] = a[idx(ybot,xa)]; as[17][x+1] = a[idx(ybot,xa+1)];
		as[17][x+2] = a[idx(ybot,xa+2)]; as[17][x+3] = a[idx(ybot,xa+3)];
		if(threadIdx.x==0) { // bottom corners
			int xleft = max(0,x0-1);   as[17][0]= a[idx(ybot,xleft)];//(17,0)
			int xright= min(nx-1,x0+64);as[17][65]= a[idx(ybot,xright)];//(17,65)
		}
		int xrgt = min(nx-1,x0+64);
		as[threadIdx.x+1][65] = a[idx(y0+threadIdx.x,xrgt)]; // right edge halo
	}
	__syncthreads();
	uchar bout[4];
	for(int k=0;k<4;k++){
		float v = fc[0]*as[y-1][x-1] + fc[1]*as[y-1][x] + fc[2]*as[y-1][x+1] +
			      fc[3]*as[y][x-1]   + fc[4]*as[y][x]   + fc[5]*as[y][x+1] +
			      fc[6]*as[y+1][x-1] + fc[7]*as[y+1][x] + fc[8]*as[y+1][x+1];

		uint kf = (uint)(v+0.5f);
		bout[k] = (uchar)min(255,max(0,kf)); // b in [0,255]
		x++;
	}
	reinterpret_cast<uchar4 *>(b)[idx(ya,xa)/4] = reinterpret_cast<uchar4 *>(bout)[0];
}


int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage filter9PT_3 <in file> <outfile> nx|512 ny|512 iter|10000 c[0] ... c[9]\n");
	}
	thrustHvec<float> c(9);

	uint nx      = (argc > 3) ? atoi(argv[3]) : 512;
	uint ny      = (argc > 4) ? atoi(argv[4]) : 512;
	uint threadx = 16;  // fixed thread blocks size
	uint thready = 16;  // for filter9PT_3
	uint iter    = (argc > 5) ? atoi(argv[5]) : 10000; // this for timing
	for(int k=0;k<9;k++) c[k] = (argc > 6+k) ? atof(argv[6+k]) : 1.0;
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
	cudaMemcpyToSymbol(fc,c.data(),9*sizeof(float));  // replaces dev_c = c
	dev_a = a;

	dim3 threads ={threadx,thready,1};
	dim3 blocks64 ={(nx+63)/64,(ny+15)/16,1}; // NB for explict 16x16 thread block size & vl

	cx::timer tim;
	for(uint k=0;k<iter;k++){  //filterered version of a placed in b on each pass
		filter9PT_3<<<blocks64,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
	}

	cudaDeviceSynchronize();
	double t1 = tim.lap_ms();
	printf("filter9PT_3 iterations %d time %.3f ms\n",iter,t1);

	// save result in b
	b = dev_b;
	cx::write_raw(argv[2],b.data(),size);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}
