// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// reduce_maxdiff  example 4.5
// 
// RTX 2070
// C:\bin\reduce_maxdiff.exe
// file stencil2D_host.raw written
// file stencilD2conv_gpu.raw written
// stencil2d size 1024 x 1024 speedup 85.407
// host iter     1000 time  2344.336 ms GFlops    1.789
// gpu  iter    10000 time   274.490 ms GFlops  152.803
// 
// RTX 3080
// C:\bin\reduce_maxdiff.exe
// file stencil2D_host.raw written
// file stencilD2conv_gpu.raw written
// stencil2d size 1024 x 1024 speedup 115.530
// host iter     1000 time  1816.242 ms GFlops    2.309
// gpu  iter    10000 time   157.209 ms GFlops  266.798

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

namespace cg = cooperative_groups;

template <typename T> __global__ void reduce_maxdiff(r_Ptr<T> smax,cr_Ptr<T>a,cr_Ptr<T>b,int n)
{
	//based on reduce6 for fixed blocksize of 256
	auto grid =  cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp =  cg::tiled_partition<32>(block); // explicit 32 thread warp

	__shared__ T s[256];
	int id = block.thread_rank();  // rank in block   
	s[id] = 0.0f;

	for(int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
		if(b != nullptr)  s[id] = fmaxf(s[id],fabs(a[tid]-b[tid]));  // first pass
		else             s[id] = fmaxf(s[id],a[tid]);               // second pass
	}
	block.sync();
	if(id < 128) s[id] = fmaxf(s[id],s[id + 128]); block.sync();
	if(id <  64) s[id] = fmaxf(s[id],s[id +  64]); block.sync();
	if(warp.meta_group_rank()==0) {
		s[id] = fmaxf(s[id],s[id + 32]); warp.sync();
		s[id] = fmaxf(s[id],warp.shfl_down(s[id],16));
		s[id] = fmaxf(s[id],warp.shfl_down(s[id],8));
		s[id] = fmaxf(s[id],warp.shfl_down(s[id],4));
		s[id] = fmaxf(s[id],warp.shfl_down(s[id],2));
		s[id] = fmaxf(s[id],warp.shfl_down(s[id],1));
		if(id == 0) smax[blockIdx.x] = s[0]; // store block max difference
	}
}

// templated version of stencil2D (T= float or double sensible choices here)
template <typename T> __global__ void stencil2D(cr_Ptr<T> a,r_Ptr<T> b,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; }; // C order suffices
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x<1 || y <1 || x >= nx-1 || y >= ny-1) return; // omit edges
	b[idx(y,x)] = (T)0.25*(a[idx(y,x+1)] + a[idx(y,x-1)] + a[idx(y+1,x)] + a[idx(y-1,x)]);
}

template <typename T> T array_diff_max(cr_Ptr<T> a,cr_Ptr<T> b,int nx,int ny)
{

	thrustDvec<T> c(256);
	thrustDvec<T> d(1);
	reduce_maxdiff<T><<<256,256>>>(c.data().get(),a,b,nx*ny);
	reduce_maxdiff<T><<<  1,256>>>(d.data().get(),c.data().get(),nullptr,256);
	cudaDeviceSynchronize();	
	return d[0];
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
	int conv     =  (argc>7) ? atoi(argv[7]) : 5000; // check for convergence after each 2*conv iterations
	double diff_cut = (argc>8) ? atof(argv[8]) : 1.0e-12; // convergence cut

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
	tim.reset();  // apply stencil max of iter_gpu times

	int iter_cv = iter_gpu;
	for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
		stencil2D<float><<<blocks,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);  // a => b
		stencil2D<float><<<blocks,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);  // b => a
		if(k>0 && k%conv==0){  // check once every nconv iterations
			cudaDeviceSynchronize();
			double diff = array_diff_max<float>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			if(diff<diff_cut){ iter_cv = k*2; printf("converged %d iterations diff = %10.3e\n",iter_cv,diff); break; }
		}
	}

	cudaDeviceSynchronize();
	double t2 = tim.lap_ms();

	a = dev_a;
	//
	// do something with result
	cx::write_raw("stencilD2conv_gpu.raw",a.data(),size);
	//
	double gflops_gpu = (double)(iter_cv*4)*(double)size/(t2*1000000);
	double speedup = gflops_gpu/gflops_host;
	printf("stencil2d size %d x %d speedup %.3f\n",nx,ny,speedup);
	printf("host iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu  iter %8d time %9.3f ms GFlops %8.3f\n",iter_cv,t2,gflops_gpu);
	return 0;
}
