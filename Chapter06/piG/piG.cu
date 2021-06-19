// example 6.7 piG cuRand Device API 

#include "cx.h"  
#include "cxtimers.h"
#include "curand_kernel.h"
#include <random>
template <typename S> __global__ void

init_generator(long long seed,S *states)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed+id,0,0,&states[id]); // faster
	//curand_init(seed, id, 0, &states[id]); // statistically better
}

template <typename S> __global__ void piG(float *tsum,S *states,int points)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id]; // restore state from previous call
	float sum = 0.0f;
	for(int i = 0; i < points; i++) {
		float x = curand_uniform(&state);
		float y = curand_uniform(&state);
		if(x*x + y*y < 1.0f) sum++; // point inside circle?
	}
	tsum[id] += sum;
	states[id] = state;  // save state for next call
}

int main(int argc,char *argv[])
{
	std::random_device rd;
	int shift =      (argc > 1) ? atoi(argv[1]) : 18;
	long long seed = (argc > 2) ? atoll(argv[2]) : rd();
	int blocks =     (argc > 3) ? atoi(argv[3]) : 2048;
	int threads =    (argc > 4) ? atoi(argv[4]) : 1024;
	long long ntot = (long long)1 << shift;

	int size = threads*blocks;
	int nthread = (ntot+size-1)/size;
	ntot = (long long)nthread*size;

	thrust::device_vector<float> tsum(size);         // thread sums
	thrust::device_vector <curandState> state(size); // generator states

	cx::timer tim;   // start clock
	init_generator<<<blocks,threads>>>(seed, state.data().get());
	piG<<<blocks,threads>>>(tsum.data().get(), state.data().get(), nthread);
	double sum_inside = thrust::reduce(tsum.begin(),tsum.end());
	double t1 = tim.lap_ms(); // record time

	double pi = 4.0*sum_inside/(double)ntot;
	double frac_error = 1000000.0*(pi - cx::pi<double>)/cx::pi<double>; // ppm
	printf("piG = %10.8f err %.3f, ntot %lld, time %.3f ms\n", pi,frac_error, ntot, t1);
	return 0;
}
