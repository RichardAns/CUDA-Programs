// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// gpulog.cu example 10.1 & 10.2 includes modifications for cuda-memcheck bug.
//
// RTX 2070 
// C:\bin\gpulog.exe 16 256 256 100 100
// steps 65536
// gpu log(2) 0.688172 frac err  7.178e-01%
// gpu  int   0.386245 frac err  1.268e-02%
// host int   0.386245 frac err  1.268e-02%
// times gpu 0.370 host 11.732 gpujob 223.730 ms speedup 31.7
// 
// C:\bin\gpulog.exe 24 256 256 100 100
// steps 16777216
// gpu log(2) 0.688172 frac err  7.178e-01%
// gpu  int   0.386245 frac err  1.267e-02%
// host int   0.386245 frac err  1.267e-02%
// times gpu 22.259 host 2985.363 gpujob 207.787 ms speedup 134.1
// 
// C:\bin\gpulog.exe 24 256 256 100000 100
// steps 16777216
// gpu log(2) 0.693134 frac err  1.874e-03%
// gpu  int   0.386294 frac err -8.701e-06%
// host int   0.386245 frac err  1.267e-02%
// times gpu 1827.215 host 2988.903 gpujob 2016.170 ms speedup 1635.8
// 
// RTX 3080
// c:\bin\gpulog.exe 16 256 256 100 100
// steps 65536
// gpu log(2) 0.688172 frac err  7.178e-01%
// gpu  int   0.386245 frac err  1.269e-02%
// host int   0.386245 frac err  1.268e-02%
// times gpu 1.203 host 6.643 gpujob 120.156 ms speedup 5.5
// 
// c:\bin\gpulog.exe 24 256 256 100 100
// steps 16777216
// gpu log(2) 0.688172 frac err  7.178e-01%
// gpu  int   0.386245 frac err  1.267e-02%
// host int   0.386245 frac err  1.267e-02%
// times gpu 19.045 host 1509.881 gpujob 142.498 ms speedup 79.3
// 
// c:\bin\gpulog.exe 24 256 256 100000 100
// steps 16777216
// gpu log(2) 0.693134 frac err  1.874e-03%
// gpu  int   0.386294 frac err -8.701e-06%
// host int   0.386245 frac err  1.267e-02%
// times gpu 2203.229 host 1521.715 gpujob 2318.414 ms speedup 690.7

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "helper_math.h"

namespace cg = cooperative_groups;

// calculate log(1+x)  series valid for  -1 < x <= +1
__host__ __device__ inline float logsum(float x,int terms)
{
	float xpow = x;
	float xn = 1.0f;
	float termsum = x;
	for(int k=1;k<terms;k++) {
		xn += 1.0f;
		xpow *= -x;
		termsum += xpow/xn; // x - x^2/2 + x^3/3 - x^4/4 ...
	}
	return termsum;
}

__global__ void gpu_log(r_Ptr<float> logs,int terms,uint steps,float step_size)
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	while(tid < steps){
		float x = step_size*(float)tid;
		logs[tid] = logsum(x,terms);
		tid += gridDim.x*blockDim.x;
	}
}

__global__ void mem_put(float *a,int loc)                   // This for cuda-memcheck deliberate bug.
{                                                           // Overwrite element loc of input array 
	if(threadIdx.x==0 && blockIdx.x==0) a[loc] = 12345.0f;  // without check index is in range
}                                                           //

float host_log(int terms,uint steps,float step_size)
{
	double sum = 0.0;  // double necessary here
	for(uint k=0; k<steps; k++){
		float x = step_size*(float)k;
		sum += logsum(x,terms);
	}
	return (float)sum;
}


// coop using int4 for 128 bit vector loads
__global__ void reduce_warp_vl(r_Ptr<float> sums,cr_Ptr<float> data,uint steps)
{
	// This kernel assumes that b.size() is power of 2 ≥ 32  
	// and that n is a multiple of 4

	auto b = cg::this_thread_block();    // thread block
	auto w = cg::tiled_partition<32>(b); // warp

	float4 v4 ={0,0,0,0};
	for(int tid = b.size()*b.group_index().x+b.thread_rank(); tid < steps/4;
		tid += b.size()*gridDim.x) v4 += reinterpret_cast<const float4 *>(data)[tid];

	float v = v4.x + v4.y + v4.z + v4.w;
	w.sync();

	v += w.shfl_down(v,16);
	v += w.shfl_down(v,8);
	v += w.shfl_down(v,4);
	v += w.shfl_down(v,2);
	v += w.shfl_down(v,1);
	if(w.thread_rank() == 0) atomicAdd_block(&sums[b.group_index().x],v);
}

// identical to above
__global__ void reduce_warp_vlB(r_Ptr<float> sums,cr_Ptr<float> data,uint steps)
{
	// This kernel assumes that b.size() is power of 2 ≥ 32  
	// and that n is a multiple of 4

	auto b = cg::this_thread_block();    // thread block
	auto w = cg::tiled_partition<32>(b); // warp

	float4 v4 ={0,0,0,0};
	for(int tid = b.size()*b.group_index().x+b.thread_rank(); tid < steps/4;
		tid += b.size()*gridDim.x) v4 += reinterpret_cast<const float4 *>(data)[tid];

	float v = v4.x + v4.y + v4.z + v4.w;
	w.sync();

	v += w.shfl_down(v,16);
	v += w.shfl_down(v,8);
	v += w.shfl_down(v,4);
	v += w.shfl_down(v,2);
	v += w.shfl_down(v,1);
	if(w.thread_rank() == 0) atomicAdd_block(&sums[b.group_index().x],v);
}


int main(int argc,char *argv[])
{
    // Windows only, comment out for Linux
	_controlfp_s(nullptr,_DN_FLUSH,_MCW_DN);  // flush unnormalised floats to zero on host

	int  shift =   argc > 1 ? atoi(argv[1]) : 16;
	uint steps = 1 << shift;
	int blocks =  argc > 2 ? atoi(argv[2]) : 128;
	int threads = argc > 3 ? atoi(argv[3]) : 512;
	int terms =   argc > 4 ? atoi(argv[4]) : 1000;
	int hterms =  argc > 5 ? atoi(argv[5]) : terms;
	int loc     = argc > 6 ? atoi(argv[6]) : 0;  // this for cuda-memcheck bug
	printf("steps %d\n",steps);
	float *logs = (float *)malloc(steps*sizeof(float));  // host
	cx::timer gpuall;            // start cuda block timer
	float *dev_logs; cudaMalloc(&dev_logs,steps*sizeof(float));
	float *dev_sums; cudaMalloc(&dev_sums,blocks*sizeof(float));
	float *dev_tot;  cudaMalloc(&dev_tot,1*sizeof(float));

	float step_size = 1.0f/(float)(steps-1);

	cx::timer tim;
	gpu_log<<<blocks,threads>>>(dev_logs,terms,steps,step_size);
	cudaMemcpy(logs,dev_logs,steps*sizeof(float),cudaMemcpyDeviceToHost);
	reduce_warp_vl<<<blocks,threads>>>(dev_sums,dev_logs,steps);  // 2 step
	reduce_warp_vlB<<<    1,blocks >>>(dev_tot,dev_sums,blocks);  // reduce
	float gpuint = 0.0f;
	cudaMemcpy(&gpuint,dev_tot,1*sizeof(float),cudaMemcpyDeviceToHost); // => host
	if(loc >0)mem_put<<<1,1>>>(dev_logs,loc); // BUG for cuda-memcheck
	cudaDeviceSynchronize();
	double tgpu= tim.lap_ms();   // end cuda block timer

	// trapizodial rule correction
	gpuint -= 0.5f*(logsum(0.0f,terms)+logsum(1.0f,terms));
	gpuint *= step_size;
	double gpujob = gpuall.lap_ms();

	double log2_gpu = logs[steps-1];  // gpu calculation of log(2)
	double log2 = log(2.0);           // true value of log(2)
	double logint = 2.0*log2 - 1.0;   //  true value of integral

	double ferr1 = 100.0f*(log2-log2_gpu)/log2;  // log(2) error
	printf("gpu log(2) %f frac err %10.3e%%\n",log2_gpu,ferr1);
	double ferr2 = 100.0f*(logint-gpuint)/logint;     // gpu integral error
	printf("gpu  int   %f frac err %10.3e%% \n",gpuint,ferr2);

    // Windows only, comment out for Linux
	_controlfp_s(nullptr,_DN_FLUSH,_MCW_DN);

	tim.reset();                   // start host timer
	double hostint = host_log(hterms,steps,step_size)*step_size;
	double thost = tim.lap_ms();   // end host timer
	hostint -= 0.5f*(logsum(0.0f,hterms)+logsum(1.0f,hterms))*step_size;
	double ferr3 = 100.0f*(logint-hostint)/logint;     // host integal error
	printf("host int   %f frac err %10.3e%%\n",hostint,ferr3);
	double ratio1 = (thost*terms)/(tgpu*hterms);

	printf("times gpu %.3f host %.3f gpujob %.3f ms speedup %.1f\n",tgpu,thost,gpujob,ratio1);

	free(logs);          // tidy up
	cudaFree(dev_logs);
	cudaFree(dev_sums);
	cudaFree(dev_tot);
	cudaDeviceReset();  // not using thrust 

	return 0;
}

