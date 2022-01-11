// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 4.2 Stencil2D_sm
// 
// RTX 2070
// C:\Users\Richard\OneDrive\toGit2>bin\stencil2D_sm.exe 2048 2048 50 50000
// file stencil2Dsm_host.raw written
// file stencil2Dsm_gpu.raw written
// stencil2d size 2048 x 2048 speedup 31.141
// host iter       50 time   189.784 ms GFlops    4.420
// gpu  iter    50000 time  6094.385 ms GFlops  137.645
//
// RTX 3080
// C:\Users\Richard\OneDrive\toGit2>bin\stencil2D_sm.exe 2048 2048 50 50000
// file stencil2Dsm_host.raw written
// file stencil2Dsm_gpu.raw written
// stencil2d size 2048 x 2048 speedup 52.483
// host iter       50 time   147.909 ms GFlops    5.671
// gpu  iter    50000 time  2818.238 ms GFlops  297.654

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

template <int Nx,int Ny> __global__ void stencil2D_sm(cr_Ptr<float> a,r_Ptr<float> b,int nx,int ny)
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
	b[idx(ya,xa)] = 0.25f*(s[ys][xs+1] + s[ys][xs-1] + s[ys+1][xs] + s[ys-1][xs]);
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
	int iter_gpu  = (argc>4) ? atoi(argv[4]) : 10000;
	uint threadx  = (argc>5) ? atoi(argv[5]) : 16;  // must be 32 or 16
	uint thready  = (argc>6) ? atoi(argv[6]) : 16;  // must be 32, 16 or 8 & <= threadx

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
	cx::write_raw("stencil2Dsm_host.raw",a.data(),size);

	// this for stencil2D_sm_in ========================================================================

	dim3 threads ={threadx,thready,1};
	// need extra thread blocks for halos, hence the -2 terms. 
	dim3 blocks_sm ={(nx+threads.x-1-2)/(threads.x-2),(ny+threads.y-1-2)/(threads.y-2),1};

	tim.reset();
	if(threadx==16 && thready==16){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<16,16><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<16,16><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else if(threadx==32 && thready==32){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<32,32><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<32,32><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else if(threadx==32 && thready==16){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<32,16><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<32,16><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else if(threadx==32 && thready==8){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<32,8><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<32,8><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else {printf("bad sm config\n"); return 1;}

	cudaDeviceSynchronize();
	double t2 = tim.lap_ms();

	a = dev_a;
	//
	// do something with result
	cx::write_raw("stencil2Dsm_gpu.raw",a.data(),size);
	//
	double gflops_gpu = (double)(iter_gpu*4)*(double)size/(t2*1000000);
	double speedup = gflops_gpu/gflops_host;
	printf("stencil2d size %d x %d speedup %.3f\n",nx,ny,speedup);
	printf("host iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu  iter %8d time %9.3f ms GFlops %8.3f\n",iter_gpu,t2,gflops_gpu);

	// for logging
	FILE* flog = fopen("stencil4PTsm_gpu.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_gpu, gflops_gpu);
	fclose(flog);
	flog = fopen("stencil4PTsm_host.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_host, gflops_host);
	fclose(flog);
	return 0;
}
