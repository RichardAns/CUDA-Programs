// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 4.3 Stencil9PT
// 
// RTX 2070
// C:\bin\stencil9PT.exe 2048 2048 50 50000
// file stencil9PT_host.raw written
// file stencil9PT_gpu.raw written
// stencil2d size 2048 x 2048 speedup 74.446
// host iter       50 time   546.071 ms GFlops    1.536
// gpu  iter    50000 time  7335.109 ms GFlops  114.362
// 
// RTX 3080
// C:\bin\stencil9PT.exe 2048 2048 50 50000
// file stencil9PT_host.raw written
// file stencil9PT_gpu.raw written
// stencil2d size 2048 x 2048 speedup 157.609
// host iter       50 time   430.421 ms GFlops    1.949
// gpu  iter    50000 time  2730.948 ms GFlops  307.168

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

__global__ void stencil9PT(cr_Ptr<float> a,r_Ptr<float> b,int nx,int ny,cr_Ptr<float> c)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; };

	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x<1 || y <1 || x >= nx-1 || y >= ny-1)return;

	b[idx(y,x)] = c[0]*a[idx(y-1,x-1)] + c[1]*a[idx(y-1,x)] + c[2]*a[idx(y-1,x+1)] +
		c[3]*a[idx(y,x-1)] + c[4]*a[idx(y,x)] + c[5]*a[idx(y,x+1)] +
		c[6]*a[idx(y+1,x-1)] + c[7]*a[idx(y+1,x)] + c[8]*a[idx(y+1,x+1)];
}

// templated shared memory version
template <int Nx,int Ny> __global__ void stencil9PT_sm(cr_Ptr<float> a,r_Ptr<float> b,int nx,int ny,cr_Ptr<float> c)
{
	__shared__ float s[Ny][Nx]; // tile includes halo

	auto idx = [&nx](int y,int x){ return y*nx+x; };

	// tiles overlap hence x0 & y0 strides reduced by twice halo width
	int x0 = (blockDim.x-2)*blockIdx.x; // x tile origin in array 
	int y0 = (blockDim.y-2)*blockIdx.y; // y tile origin in array
	int xa = x0+threadIdx.x;  // thread x in array       
	int ya = y0+threadIdx.y;  // thread y in array
	int xs = threadIdx.x;     // thread x in tile 
	int ys = threadIdx.y;     // thread y in tile
	if(xa >= nx || ya >= ny) return; // out of range check

	s[ys][xs] = a[idx(ya,xa)];   // fill Nx x Ny active points
	__syncthreads();

	if(xa < 1 || ya < 1 || xa >= nx-1 || ya >= ny-1) return; // inside array 
	if(xs < 1 || ys < 1 || xs >= Nx-1 || ys >= Ny-1) return; // inside tile 
	b[idx(ya,xa)] = c[0]*s[ys-1][xs+1] + c[1]*s[ys-1][xs-1] + c[2]*s[ys-1][xs] +
		c[3]*s[ys][xs+1] + c[4]*s[ys][xs-1] + c[5]*s[ys][xs] +
		c[6]*s[ys+1][xs+1] + c[7]*s[ys+1][xs-1] + c[8]*s[ys+1][xs];
}

int stencil9PT_host(cr_Ptr<float> a,r_Ptr<float> b,int nx,int ny,cr_Ptr<float> c)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; };
	// omit edges
	for(int y=1;y<ny-1;y++) for(int x=1;x<nx-1;x++) {
		b[idx(y,x)] = 
		c[0]*a[idx(y-1,x-1)] + c[1]*a[idx(y-1,x)] + c[2]*a[idx(y-1,x+1)] +
		c[3]*a[idx(y,x-1)] + c[4]*a[idx(y,x)] + c[5]*a[idx(y,x+1)] +
		c[6]*a[idx(y+1,x-1)] + c[7]*a[idx(y+1,x)] + c[8]*a[idx(y+1,x+1)];
	}
	return 0;
}

int main(int argc,char *argv[])
{
	int nx =        (argc>1) ? atoi(argv[1]) : 1024;
	int ny =        (argc>2) ? atoi(argv[2]) : 1024;
	int iter_host = (argc>3) ? atoi(argv[3]) : 10;   // host version very slow
	int iter_gpu =  (argc>4) ? atoi(argv[4]) : 10000;
	uint threadx  = (argc>5) ? atoi(argv[5]) : 16;  // 8, 16, 32 or 64
	uint thready  = (argc>6) ? atoi(argv[6]) : 16;  // product <= 1024

	int size = nx*ny;

	thrustHvec<float>     a(size);
	thrustHvec<float>     b(size);
	thrustDvec<float> dev_a(size);
	thrustDvec<float> dev_b(size);

	// 9-coefficient array for general 3x3 stencil
	thrustHvec<float>     c(9);
	thrustDvec<float> dev_c(9);
	for(int k=0;k<9;k++) c[k] = 1.0f/9.0f;  // simple smoothing filter here
	dev_c = c;

	// iterate inside a nx x ny rectangle.
	// set edges at x=0 and x=nx-1 to 1  
	auto idx = [&nx](int y,int x){ return y*nx+x; };
	for(int y=0;y<ny;y++) a[idx(y,0)] = a[idx(y,nx-1)] = 1.0f;
	// corner adjustment
	a[idx(0,0)] = a[idx(0,nx-1)] = a[idx(ny-1,0)] = a[idx(ny-1,nx-1)] = 0.5f;

	dev_a = a;  // copy to both
	dev_b = a;  // dev_a and dev_b 
	b = a;      // and for host_gpu

	cx::timer tim; // apply stencil iter_host times
	for(int k=0;k<iter_host/2;k++){ // ping pong buffers a and b
		stencil9PT_host(a.data(),b.data(),nx,ny,c.data()); // a=>b
		stencil9PT_host(b.data(),a.data(),nx,ny,c.data()); // b=>a
	}
	double t1 = tim.lap_ms();
	double gflops_host  = (double)(iter_host*4)*(double)size/(t1*1000000);
	cx::write_raw("stencil9PT_host.raw",a.data(),size);

	dim3 threads ={threadx,thready,1};
	dim3 blocks ={(nx+threads.x-1)/threads.x,(ny+threads.y-1)/threads.y,1};

	tim.reset();  // apply stencil iter_gpu times
	for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
		stencil9PT<<<blocks,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny,dev_c.data().get()); // a=>b
		stencil9PT<<<blocks,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny,dev_c.data().get()); // b=>a
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms();

	a = dev_a;
	//
	// do something with result
	cx::write_raw("stencil9PT_gpu.raw",a.data(),size);
	//
	double gflops_gpu = (double)(iter_gpu*4)*(double)size/(t2*1000000);
	double speedup = gflops_gpu/gflops_host;
	printf("stencil2d size %d x %d speedup %.3f\n",nx,ny,speedup);
	printf("host iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu  iter %8d time %9.3f ms GFlops %8.3f\n",iter_gpu,t2,gflops_gpu);

	// for logging
	FILE* flog = fopen("stencil9PT_gpu.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_gpu, gflops_gpu);
	fclose(flog);
	flog = fopen("stencil9PT_host.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_host, gflops_host);
	fclose(flog);
	return 0;
}
