// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 7.3 event2 program
// 
// RTX 2070
// C:\bin\event2.exe 256 256 28 1000 1000 0
// host : ht1+ht2 1134.346 ht3 1049.983 diff 84.363
// event: et1+et2 1614.256 et1 562.878 et2 1051.378 diff 564.272 ms
// done
// 
// RTX 3080
// C:\bin\event2.exe 256 256 28 1000 1000 0
// host : ht1+ht2 306.264 ht3 287.908 diff 18.356
// event: et1+et2 455.789 et1 166.279 et2 289.510 diff 167.881 ms
// done
// 
// RTX 2070
// C:\bin\event2.exe 256 256 28 1000 1000 1
// host : ht1+ht2 1142.252 ht3 1049.787 diff 92.465
// event: et1+et2 1120.504 et1 560.244 et2 560.260 diff 70.717 ms
// done
//
// RTX 3080
// C:\bin\event2.exe 256 256 28 1000 1000 1
// host : ht1+ht2 296.242 ht3 287.545 diff 8.697
// event: et1+et2 295.742 et1 146.986 et2 148.756 diff 8.197 ms
// done

#include "cx.h" 
#include "cxtimers.h"
//#include "helper_math.h"
//#include <thread>       


__global__ void mashData(cr_Ptr<float> a,r_Ptr<float> b,uint size,int ktime)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	for(int k = id; k < size; k += stride) {
		float sum = 0.0f;
		for(int m = 0; m < ktime; m++) {
			sum += sqrtf(a[k]*a[k] + (float)(threadIdx.x % 32) + (float)m);
		}
		b[k] = sum;
	}
}

int main(int argc,char *argv[])
{
	if(argc < 2) {
		printf("usage event2 <blocks> <threads> dataSize|2^28 kt1|100  kt2|kt1 sync|0 \n");
		return 0;
	}

	int blocks = (argc > 1) ? atoi(argv[1]) : 256;
	int threads =(argc > 2) ? atoi(argv[2]) : 256;
	uint dsize = (argc > 3) ? 1 << atoi(argv[3]) : 1 << 28;
	int kt1   =  (argc > 4) ? atoi(argv[4]) : 100;
	int kt2 =    (argc > 5) ? atoi(argv[5]) : kt1;
	int sync =   (argc > 6) ? atoi(argv[6]) : 0;

	// allocate a two sets of buffers one set for each CUDA stream
	thrustHvecPin<float> host1(dsize);
	thrustDvec<float>    dev_in1(dsize);
	thrustDvec<float>    dev_out1(dsize);
	thrustHvecPin<float> host2(dsize);
	thrustDvec<float>    dev_in2(dsize);
	thrustDvec<float>    dev_out2(dsize);

	cudaStream_t s1;  cudaStreamCreate(&s1);  // create two CUDA streams
	cudaStream_t s2;  cudaStreamCreate(&s2);  // s1 and s2

	// initialise buffers
	for(uint k=0; k<dsize; k++) host2[k] = host1[k] = (float)(k%77)*sqrt(2.0);
	dev_in1 = host1;  
	dev_in2 = host2;  // thrust copies: H2D

	// test1: run kernels sequentially on s1 and s2 and time each kernel using 
	// host-blocking timers host_t1 and host_t2
	cx::timer tim;
	mashData<<<blocks,threads,0,s1>>>(dev_in1.data().get(),	dev_out1.data().get(),dsize,kt1);
	cudaStreamSynchronize(s1);  // blocking only on stream s1
	double host_t1 = tim.lap_ms();
	mashData<<<blocks,threads,0,s2>>>(dev_in2.data().get(),	dev_out2.data().get(),dsize,kt2);
	cudaStreamSynchronize(s2); // blocking only on stream s2
	double host_t2 = tim.lap_ms(); // NB host knows both kernels done now

	// test2: run kernels asynchronously and measure combined time host_t3
	mashData<<<blocks,threads,0,s1>>>(dev_in1.data().get(),dev_out1.data().get(),dsize,kt1);
	mashData<<<blocks,threads,0,s2>>>(dev_in2.data().get(),dev_out2.data().get(),dsize,kt2);
	cudaDeviceSynchronize();   // wait for all steams
	double host_t3 = tim.lap_ms();
	// test3: create CUDA events for GPU based timing of asynchronous kernels
	cudaEvent_t start1; cudaEventCreate(&start1);
	cudaEvent_t stop1;  cudaEventCreate(&stop1);
	cudaEvent_t start2; cudaEventCreate(&start2);
	cudaEvent_t stop2;  cudaEventCreate(&stop2);

	// test3: launch and time asynchronous kernels using CUDA events
	cudaEventRecord(start1,s1);
	mashData<<<blocks,threads,0,s1>>>(dev_in1.data().get(),dev_out1.data().get(),dsize,kt1);
	cudaEventRecord(stop1,s1);

	if(sync != 0) cudaStreamWaitEvent(s2,stop1,0);  // optional pause in s2

	cudaEventRecord(start2,s2);
	mashData<<<blocks,threads,0,s2>>>(dev_in2.data().get(),dev_out2.data().get(),dsize,kt2);
	cudaEventRecord(stop2,s2);
	// all work now added to CUDA streams

	cudaEventSynchronize(stop1); // wait for s1
	float event_t1 = 0.0f;
	cudaEventElapsedTime(&event_t1,start1,stop1);
	cudaEventSynchronize(stop2); // wait for s2
	float event_t2 = 0.0f;
	cudaEventElapsedTime(&event_t2,start2,stop2);

	float hsum = host_t1+host_t2; float hdiff = hsum-host_t3;
	printf("host : ht1+ht2 %.3f ht3 %.3f diff %.3f\n",hsum,host_t3,hdiff);
	float esum = event_t1+event_t2; float ediff = esum-host_t3;
	printf("event: et1+et2 %.3f et1 %.3f et2 %.3f diff %.3f ms\n",esum,event_t1,event_t2,ediff);

	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop1);

	std::atexit([]{ cudaDeviceReset(); });
	printf("done\n");
	return 0;
}
