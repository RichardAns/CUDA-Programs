// example 2.10 illustating allocation of 
// multiple dynamic arrays in shared memory
// this example is not intended to be complete.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cx.h"

__global__ void shared_example(float *x,float *y,int m)
{
	// notice order of declarations, 
	// longest  variable type first
	// shortest variable type last
	extern __shared__ float sx[];   // NB sx is a pointer to the start of the
									// shared memory pool
	ushort* su = (ushort *)(&sx[blockDim.x]); // start after sx
	char*   sc =   (char *)(&su[blockDim.x]); // start after su
	int id = threadIdx.x;
	sx[id] = 3.1459*x[id];
	su[id] = id*id;
	sc[id] = id%128;
	// do something useful here . . .
}

int main(int argc, char * argv[])
{
	int threads = (argc >1) ? atoi(argv[1]) : 256;
	int size =    (argc >2) ? atoi(argv[2]) : threads*256;
	int blocks =  (size+threads-1)/threads;
	int shared = threads*(sizeof(float) + sizeof(ushort) + sizeof(char));
	thrust::device_vector<float> x(10);
	thrust::device_vector<float> y(10);
	// do something here

	shared_example<<< blocks,threads,shared >>>(x.data().get(),y.data().get(),10);
	// do more here. . .
	return 0;
}
