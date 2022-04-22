// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 7.2 event1 program
// 
// RTX 2070
// C:\bin\event1.exe 256 256 26 10 10
// times 2.621 2.603 2.550 2.536 diff 0.138 ms
// 
// RTX 3080
// C:\bin\event1.exe 256 256 26 10 10
// times 1.046 1.258 0.954 0.954 diff 0.396 ms

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
		printf("usage event1 <blocks> <threads> dataSize|2^28 kt1|100  kt2|kt1  \n");
		return 0;
	}

	int blocks = (argc > 1) ? atoi(argv[1]) : 256;
	int threads =(argc > 2) ? atoi(argv[2]) : 256;
	uint dsize = (argc > 3) ? 1 << atoi(argv[3]) : 1 << 28;
	int kt1   =  (argc > 4) ? atoi(argv[4]) : 100; // kernel time
	int kt2 =    (argc > 5) ? atoi(argv[5]) : kt1;

	thrustHvecPin<float> inbuf(dsize);
	thrustDvec<float>    dev_in(dsize);
	thrustDvec<float>    dev_out(dsize);
	for(uint k=0; k<dsize; k++) inbuf[k] = (float)(k%77)*sqrt(2.0);
	dev_in = inbuf;    // thrust copy host to gpu

	// run kernel twice and time each kernel using host timers 
	cx::timer tim;
	mashData<<<blocks,threads>>>(dev_in.data().get(),dev_out.data().get(),dsize,kt1);
	cudaDeviceSynchronize();  // blocks host
	double host_t1 = tim.lap_ms();
	mashData<<<blocks,threads>>>(dev_out.data().get(),dev_in.data().get(),dsize,kt2);
	cudaDeviceSynchronize();  // blocks host
	double host_t2 = tim.lap_ms();

	cudaEvent_t start1; cudaEventCreate(&start1);  // now repeat using CUDA
	cudaEvent_t stop1;  cudaEventCreate(&stop1);   // events for timing.
	cudaEvent_t start2; cudaEventCreate(&start2);  // Two events are needed
	cudaEvent_t stop2;  cudaEventCreate(&stop2);   // for each measurement

	cudaEventRecord(start1);  //time at first kernel launch
	mashData<<<blocks,threads>>>(dev_in.data().get(),dev_out.data().get(),dsize,kt1);
	cudaEventRecord(stop1);   //time at first kernel finish (no host blocking)

	cudaEventRecord(start2);  //time at second kernel launch
	mashData<<<blocks,threads>>>(dev_out.data().get(),dev_in.data().get(),dsize,kt2);
	cudaEventRecord(stop2);   //time at second kernel finish (no host blocking)
	
	// extra asynchronous host work possible here
		
	cudaEventSynchronize(stop2); // now block host so timers become available

	float event_t1 = 0.0f;
	float event_t2 = 0.0f;
	cudaEventElapsedTime(&event_t1,start1,stop1);  // time for first kernel
	cudaEventElapsedTime(&event_t2,start2,stop2);  // time for second kernel
	float diff = host_t1 + host_t2 - event_t1- event_t2;

	printf("times %.3f %.3f %.3f %.3f diff %.3f ms\n",host_t1, host_t2,event_t1,event_t2,diff);

	cudaEventDestroy(start1);  // Note  recommended cleanup
	cudaEventDestroy(stop1);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop1);

	std::atexit([]{ cudaDeviceReset(); });
	return 0;
}

