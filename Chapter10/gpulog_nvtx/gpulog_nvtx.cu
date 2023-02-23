// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 10.4 gpulog_nvtx
//
// This example is the same as 10.1 execpt for the
// cudaProfilerStart/Stop() statements in main and
// an extra include.

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "helper_math.h"

#include "cuda_profiler_api.h" // NB added

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

__host__ __device__ inline float logsum_d(float x,int terms)
{
	float xpow = x;
	float xn = 1.0f;
	double termsum = x;
	for(int k=1;k<terms;k++) {
		xn += 1.0f;
		xpow *= -x;
		termsum += xpow/xn; // x - x^2/2 + x^3/3 - x^4/4 ...
	}
	return (float)termsum;
}

__host__ inline float logsum(float x,int terms,int dum)
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

float host_log(int terms,uint steps,float step_size)
{
	double sum = 0.0;  // double necessary here
	for(uint k=0; k<steps; k++){
		float x = step_size*(float)k;
		sum += logsum(x,terms);
		//sum += logsum(1.0f,terms,k);
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
	//_controlfp_s(nullptr,_DN_FLUSH,_MCW_DN);

	int  shift =   argc > 1 ? atoi(argv[1]) : 16;
	uint steps = 2 << (shift-1);
	int blocks =  argc > 2 ? atoi(argv[2]) : 128;
	int threads = argc > 3 ? atoi(argv[3]) : 512;
	int terms =   argc > 4 ? atoi(argv[4]) : 1000;
	int hterms =  argc > 5 ? atoi(argv[5]) : terms;

	float *logs = (float *)malloc(steps*sizeof(float));  // host
	cx::timer job;            // start job timer
	float *dev_logs; cudaMalloc(&dev_logs,steps*sizeof(float));
	float *dev_sums; cudaMalloc(&dev_sums,blocks*sizeof(float));
	float *dev_tot;  cudaMalloc(&dev_tot,1*sizeof(float));

	float step_size = 1.0f/(float)(steps-1);
	cx::timer tim;            // start cuda block timer
	gpu_log<<<blocks,threads>>>(dev_logs,terms,steps,step_size);
	cudaMemcpy(logs,dev_logs,steps*sizeof(float),cudaMemcpyDeviceToHost);

	cudaProfilerStart();  // start profiling
	reduce_warp_vl<<<blocks,threads >>>(dev_sums,dev_logs,steps);    // 2 step
	reduce_warp_vlB<<<     1,threads >>>(dev_tot,dev_sums,threads);  // reduce
	float gpuint = 0.0f;
	cudaMemcpy(&gpuint,dev_tot,1*sizeof(float),cudaMemcpyDeviceToHost); // => host
	cudaDeviceSynchronize();  
	cudaProfilerStop();   // stop  profiling

	double tgpu= tim.lap_ms();   // end cuda block timer
	// trapizodial rule correction
	gpuint -= 0.5f*(logsum(0.0f,terms)+logsum(1.0f,terms));
	gpuint *= step_size;
	double tgpujob = job.lap_ms(); // end job timer

	double log2_gpu = logs[steps-1];  // gpu calculation of log(2)
	double log2 = log(2.0);           // true value of log(2)
	double logint = 2.0*log2 - 1.0;   //  true value of integral

	double ferr1 = 100.0f*(log2-log2_gpu)/log2;  // log(2) error

	printf("gpu log(2) %f frac err %10.3e%%\n",log2_gpu,ferr1);

	double ferr2 = 100.0f*(logint-gpuint)/logint;     // gpu integral error
	printf("gpu  int   %f frac err %10.3e%% \n",gpuint,ferr2);

	_controlfp_s(nullptr,_DN_FLUSH,_MCW_DN);
	tim.start();                // start host timer
	double hostint = host_log(hterms,steps,step_size)*step_size;
	hostint -= 0.5f*(logsum(0.0f,hterms,0)+logsum(1.0f,hterms,0))*step_size;
	double thost = tim.lap_ms();   // end host timer
	double ferr3 = 100.0f*(logint-hostint)/logint;     // host integal error
	printf("host int   %f frac err %10.3e%%\n",hostint,ferr3);
	double ratio1 = (thost*terms)/(tgpu*hterms);

	//printf("times gpu %.3f gpujob %.3f hostjob %.3f ms speedups %.1f %.1f\n",tgpu,tgpujob,thost,ratio1,ratio2);
	printf("times gpu %.3f gpujob %.3f hostjob %.3f ms speedup %.1f\n",tgpu,tgpujob,thost,ratio1);

	FILE *flog = fopen("gpulog.txt","a");
	double ratio2 = (thost*terms)/(tgpujob*hterms);
	fprintf(flog,"%3d %6d %6d %6d %e %e %e %.3f %.3f %.3f %.2f %.2f\n",shift,steps,terms,hterms,ferr1,ferr2,ferr3,tgpu,tgpujob,thost,ratio1,ratio2);
	fclose(flog);

	free(logs);          // tidy up
	cudaFree(dev_logs);
	cudaFree(dev_sums);
	cudaFree(dev_tot);
	cudaDeviceReset();  // not using thrust 

	return 0;
}

