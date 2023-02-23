// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// stencil3D_1 and stendcil3D_2 example 4.7
// either kernel can be run for either float or double array types.
// 
// RTX 2070
// C:\bin\stencil3D.exe 1 256 256 256 10 10000 0 5000 32 16
// stencil3d sizeof(T) 4 size 256 x 256 z 256 save 0
// stencil3d_1 size 256 x 256 x 256 speedup 69.943
// host   iter       10 time   232.452 ms GFlops    2.887
// gpu    iter    10000 time  4985.205 ms GFlops  201.924 bandwidth  942.312
// final convergence diff  2.205372e-05 for type float  method 1
// 
// C:\bin\stencil3D.exe 2 256 256 256 10 10000 0 5000 32 16
// stencil3d sizeof(T) 4 size 256 x 256 z 256 save 0
// stencil3d_0 size 256 x 256 x 256 speedup 85.546
// host   iter       10 time   231.502 ms GFlops    2.899
// gpu    iter    10000 time  4059.246 ms GFlops  247.985 bandwidth 1157.264
// final convergence diff  2.205372e-05 for type float  method 2
// 
// C:\bin\stencil3D.exe 3 256 256 256 10 10000 0 5000 32 16
// stencil3d sizeof(T) 8 size 256 x 256 z 256 save 0
// stencil3d_1 size 256 x 256 x 256 speedup 31.957
// host   iter       10 time   269.914 ms GFlops    2.486
// gpu    iter    10000 time 12669.234 ms GFlops   79.455 bandwidth  741.579
// final convergence diff  2.200371e-05 for type double method 1
// 
// C:\bin\stencil3D.exe 4 256 256 256 10 10000 0 5000 32 16
// stencil3d sizeof(T) 8 size 256 x 256 z 256 save 0
// stencil3d_0 size 256 x 256 x 256 speedup 44.136
// host   iter       10 time   267.842 ms GFlops    2.506
// gpu    iter    10000 time  9102.832 ms GFlops  110.585 bandwidth 1032.123
// final convergence diff  2.200371e-05 for type double method 2
// 
// RTX 3080
// C:\bin\stencil3D.exe 1 256 256 256 10 10000 0 5000
// stencil3d sizeof(T) 4 size 256 x 256 z 256 save 0
// stencil3d_1 size 256 x 256 x 256 speedup 122.495
// host   iter       10 time   198.744 ms GFlops    3.377
// gpu    iter    10000 time  2433.695 ms GFlops  413.623 bandwidth 1930.242
// final convergence diff  2.205372e-05 for type float  method 1

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

namespace cg = cooperative_groups;

template <typename T> int main2(int argc,char *argv[]);

// Threads span 2D x-y slice and for loop used for z
template <typename T>__global__ void stencil3D_1(cr_Ptr<T> a,r_Ptr<T> b,int nx,int ny,int nz)
{
	auto idx = [&nx,&ny](int z,int y,int x){ return (z*ny+y)*nx+x; };

	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x < 1 || y < 1 || x >= nx-1 || y >= ny-1) return;

	for(int z=1;z<nz-1;z++)  b[idx(z,y,x)] =
		(T)(0.16666666666667)*(a[idx(z,y,x+1)] + a[idx(z,y,x-1)] +
			a[idx(z,y+1,x)] + a[idx(z,y-1,x)] +
			a[idx(z+1,y,x)] + a[idx(z-1,y,x)]);
}

// Threads span 3D volume, one thread per element
template <typename T>__global__ void stencil3D_2(cr_Ptr<T> a,r_Ptr<T> b,int nx,int ny,int nz)
{
	auto idx = [&nx,&ny](int z,int y,int x){ return (z*ny+y)*nx+x; };

	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	int z = blockIdx.z*blockDim.z+threadIdx.z;
	if(x < 1 || y < 1 || x >= nx-1 || y >= ny-1 || z <1  || z >= nz-1) return;

	b[idx(z,y,x)] = (T)(0.16666666666667)*(a[idx(z,y,x+1)] + a[idx(z,y,x-1)] +
		a[idx(z,y+1,x)] + a[idx(z,y-1,x)] +
		a[idx(z+1,y,x)] + a[idx(z-1,y,x)]);
}

template <typename T> int stencil3D_host(cr_Ptr<T> a,r_Ptr<T> b,int nx,int ny,int nz)
{
	auto idx = [&nx,&ny](int z,int y,int x){ return (z*ny+y)*nx+x; };

	for(int z=1;z<nz-1;z++) for(int y=1;y<ny-1;y++) for(int x=1;x<nx-1;x++){  // omit all faces
		b[idx(z,y,x)] = (T)(0.1666666666666667)*
			(a[idx(z,y,x+1)] + a[idx(z,y,x-1)] +
				a[idx(z,y+1,x)] + a[idx(z,y-1,x)] +
				a[idx(z+1,y,x)] + a[idx(z-1,y,x)]);
	}
	return 0;
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
		if(b != nullptr) s[id] = fmaxf(s[id],fabs(a[tid]-b[tid]));  // first pass
		else             s[id] = fmaxf(s[id],a[tid]);                // second pass
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

template <typename T> T array_diff_max(cr_Ptr<T> a,cr_Ptr<T> b,int n)
{
	thrustDvec<T> c(256);
	thrustDvec<T> d(1);
	reduce_maxdiff<T><<<256,256>>>(c.data().get(),a,b,n);
	reduce_maxdiff<T><<<  1,256>>>(d.data().get(),c.data().get(),nullptr,256);
	cudaDeviceSynchronize();
	T hd = d[0];
	return hd;
}

int main(int argc,char * argv[])
{
	int type =  (argc>1) ? atoi(argv[1]) : 0;
	if(     type==1 || type==2 ) return main2<float>(argc,argv);
	else if(type==3 || type==4)  return main2<double>(argc,argv);

	printf("usage stencil3D type nx|256 ny|256 nz|256 iter_h|10 iter_g|10000 save|0 conv|5000 threadsx|32 threadsy|16\n");
	printf("type = 1: method 1 type float\n");
	printf("type = 2: method 2 type float\n");
	printf("type = 3: method 1 type double\n");
	printf("type = 4: method 2 type double\n");
	return 0;
}

// this is really the main routine but templated for float or double
template <typename T> int main2(int argc,char *argv[])
{
	int ver3d =     (argc>1) ?  atoi(argv[1]) : 0;     // 1 or 3: float kernels method 1 or 2; 2 or 4: double kernels method 1 or 2 
	int nx =        (argc>2) ?  atoi(argv[2]) : 256;   
	int ny =        (argc>3) ?  atoi(argv[3]) : 256;   // 3D array size
	int nz =        (argc>4) ?  atoi(argv[4]) : 256;  
	int iter_host = (argc>5) ?  atoi(argv[5]) : 10;    // host iterations
	int iter_gpu =  (argc>6) ?  atoi(argv[6]) : 10000; // gpu max iterations
	int save     =  (argc>7) ?  atoi(argv[7]) : 0;     // save result (big file)?
	int conv     =  (argc>8) ?  atoi(argv[8]) : 5000;  // convergence check frequency
	uint threadx =  (argc>9) ?  atoi(argv[9]) : 32;    //  try 32 16 1 for stencil3D_1
	uint thready =  (argc>10) ? atoi(argv[10]) :16;    //  and 64  4 1 for stencil3D_2
	uint threadz =  (argc>11) ? atoi(argv[11]) : 1;    //


	int size = nx*ny*nz;

	printf("stencil3d sizeof(T) %d size %d x %d z %d save %d\n",(int)sizeof(T),nx,ny,nz,save);

	thrustHvec<T>     a(size);
	thrustHvec<T>     b(size);
	thrustDvec<T> dev_a(size);
	thrustDvec<T> dev_b(size);

	auto idx = [&nx,&ny](int z,int y,int x){ return (z*ny+y)*nx+x; };

	// set x=0 and x=nx-1 planes to 1	
	for(int z=0;z<nz;z++) for(int y=0;y<ny;y++) a[idx(z,y,0)] = a[idx(z,y,nx-1)]  = (T)1.0;

	for(int z=0;z<nz;z++){
		a[idx(z,0,0)] = a[idx(z,0,nx-1)] = a[idx(z,ny-1,0)] = a[idx(z,ny-1,nx-1)] = (T)0.5; // 4 horizontal edges
	}
	for(int y=0;y<ny;y++){
		a[idx(0,y,0)] = a[idx(0,y,nx-1)] = a[idx(nz-1,y,0)] =a[idx(nz-1,y,nx-1)] = (T)0.5; // 4 vertical
	}
	a[idx(0,0,0)] =    a[idx(0,0,nx-1)] =    a[idx(0,ny-1,0)]    = a[idx(0,ny-1,nx-1)]    = (T)(1.0/3.0);  //corners
	a[idx(nz-1,0,0)] = a[idx(nz-1,0,nx-1)] = a[idx(nz-1,ny-1,0)] = a[idx(nz-1,ny-1,nx-1)] = (T)(1.0/3.0);
	dev_a = a;  // copy to both
	dev_b = a;  // dev_a and dev_b  
	b = a;      // and to host b
	if(save>1)cx::write_raw("stencil3d_start.raw",a.data(),size);

	// this for host ==========================================
	cx::timer tim;
	for(int k=0;k<iter_host/2;k++){ // ping pong buffers a and b
		stencil3D_host<T>(a.data(),b.data(),nx,ny,nz);
		stencil3D_host<T>(b.data(),a.data(),nx,ny,nz);
	}
	double t1 = tim.lap_ms();
	double gflops_host  = (double)(iter_host*4)*(double)size/(t1*1000000);
	if(save>2)cx::write_raw("stencil3d_host.raw",a.data(),size);

	dim3 threads ={threadx, thready,threadz};
	dim3 blocks ={(nx+threads.x-1)/threads.x,(ny+threads.y-1)/threads.y,1};
	dim3 blocks_2 ={(nx+threads.x-1)/threads.x,(ny+threads.y-1)/threads.y,(nz+threads.z-1)/threads.z};

	// GPU version here
	tim.reset();
	int iter_conv = 0;
	while(iter_conv < iter_gpu){
		if(ver3d==1 || ver3d==3) for(int k=0;k<conv/2;k++){  // ping pong buffers dev_a and dev_b
			stencil3D_1<T><<<blocks,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny,nz);
			stencil3D_1<T><<<blocks,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny,nz);
		}
		else if(ver3d==2 || ver3d==4) for(int k=0;k<conv/2;k++){  // ping pong buffers dev_a and dev_b
			stencil3D_2<T><<<blocks_2,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny,nz);
			stencil3D_2<T><<<blocks_2,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny,nz);
		}
		else { printf("bad ver3d = %d\n", ver3d); return 1; }
		cudaDeviceSynchronize();
		T diff = array_diff_max<T>(dev_a.data().get(),dev_b.data().get(),size);
		iter_conv += conv;
		if(diff<1.0e-16){ printf("stencid3D_%d converged k=%d diff %13.6e\n",ver3d%2,iter_conv,diff); break; }
	}
	cudaDeviceSynchronize();

	double t2 = tim.lap_ms();     
	double bytes_per_call = 7.0*(double)sizeof(T);                         // 6 reads and 1 write
	double gflops_gpu = (double)(iter_conv*6)*(double)size/(t2*1000000);   // 5 adds and one multiply
	double bandw = (double)(nx*ny*nz)*(double)(iter_conv)*bytes_per_call/(t2*1000000.0);  
	double speedup = gflops_gpu/gflops_host;
	b = dev_a;
	if (save) {
		char name[256]; sprintf(name, "stencil3d_%d.raw", ver3d);
		cx::write_raw(name, b.data(), size);
	}

	printf("stencil3d_%d size %d x %d x %d speedup %.3f\n",ver3d%2,nx,ny,nz,speedup);
	printf("host   iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu    iter %8d time %9.3f ms GFlops %8.3f bandwidth %8.3f\n",iter_conv,t2,gflops_gpu,bandw);
	T diff_final = array_diff_max<T>(dev_a.data().get(),dev_b.data().get(),size);
	if     (ver3d==1)printf("final convergence diff %13.6e for type float  method 1\n",diff_final);
	else if(ver3d==2)printf("final convergence diff %13.6e for type float  method 2\n",diff_final);
	else if(ver3d==3)printf("final convergence diff %13.6e for type double method 1\n",diff_final);
	else if(ver3d==4)printf("final convergence diff %13.6e for type double method 2\n",diff_final);

	return 0;
}