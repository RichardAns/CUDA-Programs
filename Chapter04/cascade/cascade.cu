// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 4.6 cascade
// RTX 2070
// C:\bin\cascade.exe 0 1024 1024 1000000 (floats)
// mx    32 iter 2500
// mx    64 iter 2500
// mx   128 iter 2500
// mx   256 iter 2500
// mx   512 iter 2500
// mx  1024 iter 20000
// mx 1024 diff 4.76837e-07
// file cascade1024_4.raw written
// cascade time 1134.328 ms
// 
// RTX 3080
// C:\bin\cascade.exe 0 1024 1024 1000000  (floats)
// mx    32 iter 2500
// mx    64 iter 2500
// mx   128 iter 2500
// mx   256 iter 2500
// mx   512 iter 2500
// mx  1024 iter 20000
// mx 1024 diff 4.76837e-07
// file cascade1024_4.raw written
// cascade time 693.533 ms
// 
// RTX 2070
// C:\bin\cascade.exe 1 1024 1024 1000000 (doubles)
// mx    32 iter 2500
// mx    64 iter 2500
// mx   128 iter 2500
// mx   256 iter 5000
// mx   512 iter 12500
// mx  1024 iter 120000
// mx 1024 diff 7.72109e-11
// file cascade1024_8.raw written
// cascade time 11507.192 ms
// 
// RTX 3080
// C:\bin\cascade.exe 1 1024 1024 1000000 (doubles)
// mx    32 iter 2500
// mx    64 iter 2500
// mx   128 iter 2500
// mx   256 iter 5000
// mx   512 iter 12500
// mx  1024 iter 120000
// mx 1024 diff 7.72109e-11
// file cascade1024_8.raw written
// cascade time 6823.324 ms

#include "cooperative_groups.h"
#include "cx.h"
#include "cxbinio.h"
#include "cxtimers.h"

namespace cg = cooperative_groups;

template <typename T> __global__ void stencil2D(cr_Ptr<T> a,r_Ptr<T> b,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; }; // C order suffices
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x<1 || y <1 || x >= nx-1 || y >= ny-1) return; // omit edges
	b[idx(y,x)] = (T)0.25*(a[idx(y,x+1)] + a[idx(y,x-1)] + a[idx(y+1,x)] + a[idx(y-1,x)]);
}

template <typename T> __global__ void zoomfrom(r_Ptr<T> a,cr_Ptr<T> aold,int nx,int ny)
{
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	if(x >= nx || y >= ny) return;

	int mx = nx/2;
	auto idx = [&nx](int y,int x){ return y*nx+x; };
	auto mdx = [&mx](int y,int x){ return y*mx+x; };

	if(x>0 && x<nx-1 && y>0 && y<ny-1) a[idx(y,x)] = aold[mdx(y/2,x/2)];
	else if(y==0 && x>0 && x<nx-1)     a[idx(y,x)] = (T)0;   // top & bottom
	else if(y==ny-1 && x>0 && x<nx-1)  a[idx(y,x)] = (T)0;
	else if(x==0 && y>0 && y<ny-1)     a[idx(y,x)] = (T)1;   // sides
	else if(x==nx-1 && y>0 && y<ny-1)  a[idx(y,x)] = (T)1;
	else if(x==0 && y==0)              a[idx(y,x)] = (T)0.5; //corners
	else if(x==nx-1 && y==0)           a[idx(y,x)] = (T)0.5;
	else if(x==0 && y==ny-1)           a[idx(y,x)] = (T)0.5;
	else if(x==nx-1 && y==ny-1)        a[idx(y,x)] = (T)0.5;
}

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

template <typename T> T array_diff_max(cr_Ptr<T> a,cr_Ptr<T> b,int nx,int ny)
{
	thrustDvec<T> c(256);
	thrustDvec<T> d(1);
	reduce_maxdiff<T><<<256,256>>>(c.data().get(),a,b,nx*ny);
	reduce_maxdiff<T><<<  1,256>>>(d.data().get(),c.data().get(),nullptr,256);
	cudaDeviceSynchronize();
	return d[0];
}


template <typename T> int cascade(int nx,int ny,int iter)
{

	int nx_start = std::min(nx,32);
	int size = nx_start*nx_start;  // square array only!
	thrustDvec<T>  dev_a(size);    // initial buffers
	thrustDvec<T>  dev_b(size);
	thrustDvec<T>  dev_aold(size);

	cx::timer tim;                         // nx is final size
	double diff = 0.0f;
	for(int mx=nx_start; mx<=nx; mx *= 2){ // mx is current size
		int my = mx; // assume square
		dim3 threads(16,16,1);
		dim3 blocks((mx+15)/16,(my+15)/16,1);
		int size = mx*my;
		if(mx>nx_start){
			dev_a.resize(size);
			dev_b.resize(size);
		}
		zoomfrom<T><<<blocks,threads>>>(dev_a.data().get(),dev_aold.data().get(),mx,my);
		dev_b = dev_a;

		int check = (mx==nx) ? 5000 : 2500;  // convergence check frequency 
		double           diff_cut = (mx==nx) ? 5.0e-7 : 1.0e-05;  // convergence accuracy for floats
		if(sizeof(T)==8) diff_cut = (mx==nx) ? 1.0e-10 : 1.0e-07;  // convergence accuracy for doubles
		for(int k=0; k<iter/2; k++){
			stencil2D<T><<<blocks,threads>>>(dev_a.data().get(),dev_b.data().get(),mx,my);
			stencil2D<T><<<blocks,threads>>>(dev_b.data().get(),dev_a.data().get(),mx,my);
			if(k>0 && k%check==0){
				cudaDeviceSynchronize();
				//double diff = array_diff_max<T>(dev_a.data().get(),dev_b.data().get(),mx,my);
				diff = array_diff_max<T>(dev_a.data().get(),dev_b.data().get(),mx,my);
				//if(diff<diff_cut)  break;
				if (diff<diff_cut) { printf("mx %5d iter %d\n", mx, k); break; }
			}
		}
		if(mx==nx)printf("mx %d diff %g\n", mx, diff);
		cudaDeviceSynchronize();
		if(mx>nx_start) dev_aold.resize(size);
		if(mx<nx) dev_aold = dev_a;
	}
	double t1 = tim.lap_ms();

	thrustHvec<T> a(nx*nx);
	a = dev_a;
	char name[256]; sprintf(name,"cascade%d_%d.raw",nx,(int)sizeof(T));
	cx::write_raw(name,a.data(),nx*nx); // square
	printf("cascade time %.3f ms\n",t1);
	return 0;
}

int main(int argc,char *argv[])
{
	int type =      (argc>1) ? atoi(argv[1]) : 0;  // default use float
	int nx =        (argc>2) ? atoi(argv[2]) : 1024;
	int ny =        (argc>3) ? atoi(argv[3]) : 1024;
	int iter_gpu =  (argc>4) ? atoi(argv[4]) : 10000;

	if(type==1)cascade<double>(ny,nx,iter_gpu);   // use double precision
	else       cascade<float>(ny,nx,iter_gpu);   // use single precision

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}


