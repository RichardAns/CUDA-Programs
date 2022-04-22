// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 4.1 Stencil2D
// 
// RTX 2070
// C:\bin\stencil2D.exe 2048 2048 50 50000
// file stencil2D_host.raw written
// file stencil2D_gpu.raw written
// stencil2d size 2048 x 2048 speedup 41.330
// host iter       50 time   188.957 ms GFlops    4.439
// gpu  iter    50000 time  4571.898 ms GFlops  183.482
//  
// RTX 3080
// C:\bin\stencil2D.exe 2048 2048 50 50000
// file stencil2D_host.raw written
// file stencil2D_gpu.raw written
// stencil2d size 2048 x 2048 speedup 56.462
// host iter       50 time   147.285 ms GFlops    5.695
// gpu  iter    50000 time  2608.547 ms GFlops  321.582

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

__global__ void stencil2D(cr_Ptr<float> a,r_Ptr<float> b,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; }; // C order suffices
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x<1 || y <1 || x >= nx-1 || y >= ny-1) return; // omit edges
	b[idx(y,x)] = 0.25f*(a[idx(y,x+1)] + a[idx(y,x-1)] + a[idx(y+1,x)] + a[idx(y-1,x)]);
}

int stencil2D_host(cr_Ptr<float> a,r_Ptr<float> b,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; };
	// omit edges
	for(int y=1;y<ny-1;y++) for(int x=1;x<nx-1;x++) b[idx(y,x)] =
		0.25f*(a[idx(y,x+1)] + a[idx(y,x-1)] + a[idx(y+1,x)] + a[idx(y-1,x)]);
	return 0;
}

int main(int argc,char *argv[])
{
	int nx =        (argc>1) ? atoi(argv[1]) : 1024;
	int ny =        (argc>2) ? atoi(argv[2]) : 1024;
	int iter_host = (argc>3) ? atoi(argv[3]) : 1000;
	int iter_gpu =  (argc>4) ? atoi(argv[4]) : 10000;
	uint threadx  = (argc>5) ? atoi(argv[5]) : 16;  // 8, 16, 32 or 64
	uint thready  = (argc>6) ? atoi(argv[6]) : 16;  // product <= 1024

	int size = nx*ny;

	thrustHvec<float>     a(size);
	thrustHvec<float>     b(size);
	thrustDvec<float> dev_a(size);
	thrustDvec<float> dev_b(size);

	// solve Poisson's equation inside a nx x ny rectangle.
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
		stencil2D_host(a.data(),b.data(),nx,ny); // a=>b
		stencil2D_host(b.data(),a.data(),nx,ny); // b=>a
	}
	double t1 = tim.lap_ms();
	double gflops_host  = (double)(iter_host*4)*(double)size/(t1*1000000);
	cx::write_raw("stencil2D_host.raw",a.data(),size);

	dim3 threads ={threadx,thready,1};
	dim3 blocks ={(nx+threads.x-1)/threads.x,(ny+threads.y-1)/threads.y,1};

	tim.reset();  // apply stencil iter_gpu times
	for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
		stencil2D<<<blocks,threads>>>(dev_a.data().get(), dev_b.data().get(), nx, ny); // a=>b
		stencil2D<<<blocks,threads>>>(dev_b.data().get(), dev_a.data().get(), nx, ny); // b=>a
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms();

	a = dev_a; 
	//
	// do something with result
	cx::write_raw("stencil2D_gpu.raw",a.data(),size);
	//
	double gflops_gpu = (double)(iter_gpu*4)*(double)size/(t2*1000000);
	double speedup = gflops_gpu/gflops_host;
	printf("stencil2d size %d x %d speedup %.3f\n",nx,ny,speedup);
	printf("host iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu  iter %8d time %9.3f ms GFlops %8.3f\n",iter_gpu, t2,gflops_gpu);

	// for logging
	FILE* flog = fopen("stencil4PT_gpu.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_gpu, gflops_gpu);
	fclose(flog);
	flog = fopen("stencil4PT_host.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_host, gflops_host);
	fclose(flog);
	return 0;
}
