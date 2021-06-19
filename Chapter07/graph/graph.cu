// example 7.6 CUDA graph program

#include "cx.h"
#include "cxtimers.h"

// simple scaling kernel
__global__ void scale(r_Ptr<float> dev_out,cr_Ptr<float> dev_in,int size,float lambda)
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	while(tid<size) {
		dev_out[tid]=lambda*dev_in[tid];
		tid += blockDim.x*gridDim.x;
	}
}

int main(int argc,char *argv[])
{

	int blocks  = (argc > 1) ? atoi(argv[1]) : 256;
	int threads = (argc > 2) ? atoi(argv[2]) : 256;
	int size    = (argc > 3) ? 2 << atoi(argv[3])-1 : 2 << 15;
	int steps   = (argc > 4) ? atoi(argv[4]) : 1000;
	int kerns   = (argc > 5) ? atoi(argv[5]) : 20;

	printf("graphs %d %d size %d loops %d kernels %d\n",blocks,threads,size,steps,kerns);

	thrustHvec<float>    host_data(size);
	thrustDvec<float>    dev_out(size);
	thrustDvec<float>    dev_in(size);
	for(int k=0;k<size;k++) host_data[k] = (float)(k%419);
	dev_in = host_data; // copy to device
	float lambda = (float)pow(10.0,(double)(1.0/(steps*kerns)));
	cudaStream_t s1;  cudaStreamCreate(&s1);

	cx::timer tim;
	for(int n=0; n<steps; n++){
		for(int k=0; k<kerns/2; k++){
			scale<<<blocks,threads,0,s1>>>(trDptr(dev_out),trDptr(dev_in),size,lambda);
			scale<<<blocks,threads,0,s1>>>(trDptr(dev_in),trDptr(dev_out),size,lambda);
		}
	}
	cudaStreamSynchronize(s1);

	double t1 = tim.lap_ms();
	float x1 = dev_in[1];
	printf("standard    time %8.3f ms check %f (expect %f)\n",t1,x1,host_data[1]*10.0);

	dev_in = host_data;  // restore dev_in	 
	tim.reset();
	// capture work
	cudaStreamBeginCapture(s1,cudaStreamCaptureModeGlobal);
	for(int k=0; k<kerns/2; k++){
		scale<<<blocks,threads,0,s1>>>(trDptr(dev_out),trDptr(dev_in),size,lambda);
		scale<<<blocks,threads,0,s1>>>(trDptr(dev_in),trDptr(dev_out),size,lambda);
	}
	// create graph and instantiate
	cudaGraph_t     graph; cudaStreamEndCapture(s1,&graph);
	cudaGraphExec_t g; cudaGraphInstantiate(&g,graph,nullptr,nullptr,0);

	// run graph nsteps times in stream s2
	cudaStream_t s2;  cudaStreamCreate(&s2);
	for(int n=0; n<steps; n++)cudaGraphLaunch(g,s2);
	cudaStreamSynchronize(s2);
	double t2 = tim.lap_ms();
	float x2 = dev_in[1];

	printf("using graph time %8.3f ms check %f (expect %f)\n",t2,x2,host_data[1]*10.0);

	return 0;
}

