// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// atomicCAS examples B.1 and B2

#include "cx.h"

// myatomic_add examples of atomic function constructed using atomicCAS. 
// This code is based on the example in section B14 of the 
// CUDA C++ Programming Guide (SDK version 11.2) 

// this version for int values, is example B.1
__device__ int myatomic_add(int *acc,int val)
{
	int acc_now = acc[0];
	while(1) {
		int acc_test = acc_now; // current accumulator value
		acc_now = atomicCAS(acc,acc_test,acc_now+val);
		if(acc_test == acc_now) break;
	}
	return acc_now;
}

// this version for float value is example B.2
__device__ float myatomic_add(float *acc,float val)
{
	float acc_now = acc[0];
	while(1) {
		float acc_test = acc_now; // curent acculator
		acc_now = __uint_as_float(atomicCAS((uint *)acc,
			__float_as_uint(acc_test),
			__float_as_uint(acc_now+val)));
		//if(acc_test == acc_now) break;
		if(__float_as_uint(acc_test) == __float_as_uint(acc_now)) break;
	}

	return acc_now;
}



// can use templated kernel here
template <typename T>__global__ void addKernel(T *sum, T *data)
{
	int i = threadIdx.x;
	myatomic_add(sum,data[i]);  // will use argument types choose callled function
}

// non-atomic add used here
template <typename T>__global__ void addSimple(T *sum,T *data)
{
	int i = threadIdx.x;
	sum[0] += data[i];  // non atomic add - undefined results
}


int main(int argc,char *argv[])
{
	if(argc < 2) {
		printf("usage: atomicCAS.exe <thread block size|1024>");
		return 0;
	}
	int threads = (argc > 1) ? atoi(argv[1]) : 256;
	threads = std::min(1024,threads);  // max thread block size is 1024

	cx::ok(cudaSetDevice(0));  // Choose which GPU to run on
	thrustHvec<int>   idata(threads); for(int k=0;k<threads;k++) idata[k] = k+1;
	thrustHvec<float> fdata(threads); for(int k=0;k<threads;k++) fdata[k] = (float)(k+1);
	
	int answer = (threads*(threads+1))/2;  // sum of integers n(n+1)/2

	thrustDvec<int>   dev_idata(threads); 
	thrustDvec<float> dev_fdata(threads);
	thrustDvec<int>   dev_isum(1);
	thrustDvec<int>   dev_isum_simple(1);
	thrustDvec<float> dev_fsum(1);

	dev_idata = idata;  // copy to device
	dev_fdata = fdata;  // copy to device

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<int><<<  1,threads>>>(dev_isum.data().get(),dev_idata.data().get());
	addKernel<float><<<1,threads>>>(dev_fsum.data().get(),dev_fdata.data().get());

	addSimple<int><<<1,threads>>>(dev_isum_simple.data().get(),dev_idata.data().get());

	cx::ok(cudaGetLastError());
	cx::ok(cudaDeviceSynchronize());

	int   isum = dev_isum[0]; // get int result
	float fsum = dev_fsum[0]; // get float result

	int ssum = dev_isum_simple[0]; // get non-atomic result

	printf("isum = %d     for size %d\n",isum,threads);
	printf("fsum = %.3f for size %d\n",fsum,threads);
	printf("expect %d\n",answer);
	printf("non-atomic sum = %d for size %d\n",ssum,threads);


	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}