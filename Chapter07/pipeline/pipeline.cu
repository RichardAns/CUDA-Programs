// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// pipeline example 7.1  overlap IO and multiple kernel executions on GPU 
// RTX 2070
// C:\bin\pipeline.exe 256 256 28 1 60 0
// time 215.031 ms
// 
// C:\bin\pipeline.exe 256 256 28 8 60 8
// time 120.456 ms
// 
// RTX 3080
// C:\bin\pipeline.exe 256 256 28 1 60 0
// time 178.953 ms
// 
// C:\bin\pipeline.exe 256 256 28 8 60 8
// time 107.898 ms

#include "cx.h"
#include "cxtimers.h"


__global__ void mashData(cr_Ptr<float> a,r_Ptr<float> b,uint asize,int ktime)
{
	int id =     blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	for(int k = id; k<asize; k+=stride) {
		float sum = 0.0f;
		for(int m = 0; m < ktime; m++) {
			sum += sqrtf(a[k]*a[k]+(float)(threadIdx.x%32)+(float)m);
		}
		b[k] = sum;
	}
}

int main(int argc,char *argv[])
{
	int blocks  = (argc > 1) ? atoi(argv[1]) : 256;
	int threads = (argc > 2) ? atoi(argv[2]) : 256;
	uint dsize  = (argc > 3) ? 1 << atoi(argv[3]) : 1 << 28;  // data size
	int frames  = (argc > 4) ? atoi(argv[4]) : 16;   // frames
	int  ktime  = (argc > 5) ? atoi(argv[5]) : 60;   // kernel workload
	int maxcon  = (argc > 6) ? atoi(argv[6]) : 8;    // max connections 
	uint fsize = dsize / frames;                     // frame size

	if(maxcon > 0) {
		char set_maxconnect[256];
		sprintf(set_maxconnect,"CUDA_DEVICE_MAX_CONNECTIONS=%d",maxcon);
        _putenv(set_maxconnect);    // this for Windows
        // putenv(set_maxconnect);  // this for Linux
    }
	thrustHvecPin<float>  host(dsize);     // host data buffer
	thrustDvec<float>     dev_in(dsize);   // device input data buffer
	thrustDvec<float>     dev_out(dsize);  // device output data buffer
	for(uint k = 0; k < dsize; k++) host[k] = (float)(k%77)*sqrt(2.0);

	thrustHvec<cudaStream_t> streams(frames); // buffer for stream objects
	for(int i = 0; i < frames; i++) cudaStreamCreate(&streams[i]);

	float *hptr    = host.data();          // copy
	float *in_ptr  = dev_in.data().get();  // pointers
	float *out_ptr = dev_out.data().get(); // used in for loop

	cx::timer tim;
	// data transfers and kernel launch in each asynchronous stream
	for(int f=0; f<frames; f++) {
		if(maxcon > 0) {  // here for multiple asynchronous streams
			cudaMemcpyAsync(in_ptr,hptr, sizeof(float)*fsize, cudaMemcpyHostToDevice, streams[f]);
			if(ktime > 0)mashData<<<blocks,threads,0,streams[f]>>>(in_ptr, out_ptr, fsize,ktime);
			cudaMemcpyAsync(hptr,out_ptr, sizeof(float)*fsize, cudaMemcpyDeviceToHost, streams[f]);
		}
		else {  // here for single synchronous default stream
			cudaMemcpyAsync(in_ptr, hptr, sizeof(float)*fsize, cudaMemcpyHostToDevice, 0);
			if(ktime > 0)mashData<<<blocks,threads,0,0>>>(in_ptr, out_ptr, fsize, ktime);
			cudaMemcpyAsync(hptr, out_ptr, sizeof(float)*fsize, cudaMemcpyDeviceToHost, 0);
		}
		hptr    += fsize;  // advance pointers for next frame
		in_ptr  += fsize;
		out_ptr += fsize;
	}
	cudaDeviceSynchronize();
	double t1 = tim.lap_ms(); printf("time %.3f ms\n",t1);

	// continue host calculations here
	
	std::atexit([]{cudaDeviceReset();});
	return 0;
}
