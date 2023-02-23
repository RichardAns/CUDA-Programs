// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// 3D Ising Model Simulation examples 6.8-6.10
// 
// RTX 2070
// C:\bin\ising.exe 512 512 512 123456  3.5115 1000  1 32 8
// timing setup 0.516 ms, init 0.830 ms, fliping 3033.441 ms
// file ising.raw written
// 
// RTX 3080
// C:\bin\ising.exe 512 512 512 123456  3.5115 1000  1 32 8
// timing setup 0.571 ms, init 0.565 ms, fliping 1560.559 ms
// file ising.raw written

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"
#include <curand_kernel.h> 
#include <random>

// #include "Windows.h"  // Is not required on Windows and breaks Linux compilation
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

// NB because of 2-pass checkerboard update pattern the nx parameter
// in the GPU code is NX/2 where NX is x dimension of volume
__global__ void setup_randstates(r_Ptr<curandState> state, cint nx, cint ny, cllong seed)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if(x >= nx || y >= ny) return;
	int id = nx*y+x;
	curand_init(seed+id,0,0,&state[id]);  // method (b) less correct but faster

   // flush first 100 values to remove possible early correlations
	curandState myState = state[id];
	float sum = 0.0f;
	for(int k=0;k<100;k++) sum += curand_uniform(&myState);
	state[id] = myState;
}

template <typename T> __global__ void init_spins(r_Ptr<curandState> state, r_Ptr<T> spin, cint nx, cint ny, cint nz)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if(x >= nx || y >= ny) return;

	int id = nx*y+x;
	int nx2 = 2*nx;  // actual x-dimension NX of volume
	auto idx = [&nx2,&ny,&nz](int z,int y,int x){ return (ny*z+y)*nx2+x; };

	curandState myState = state[id];
	for(int z=0; z<nz; z++){         //random fill but other patterns possible
		if(curand_uniform(&myState) <= 0.5f) spin[idx(z,y,x)] =  1;
		else                                 spin[idx(z,y,x)] = -1;
		if(curand_uniform(&myState) <= 0.5f) spin[idx(z,y,x+nx)] =  1;
		else                                 spin[idx(z,y,x+nx)] = -1;
	}
	state[id] = myState;
}

template <typename T> __global__ void flip_spins(r_Ptr<curandState> state, r_Ptr<T> spin, cint nx, cint ny, cint nz, cfloat temp, cint colour)
{
	int xt = threadIdx.x + blockIdx.x*blockDim.x; // NB thread x not spin x
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if(xt >= nx || y >= ny) return;

	int id = nx*y+xt;
	int nx2 = 2*nx;     // NB nx2 is volume x dimension
	auto idx = [&nx2,&ny,&nz](int z,int y,int x){ return (ny*z+y)*nx2+x; };
	// for cyclic boundary conditions
	auto cyadd = [](int a,int bound) { return a < bound-1 ? a+1 : 0; };
	auto cysub = [](int a,int bound) { return a > 0 ? a-1 : bound-1; };

	curandState myState = state[id];

	int yl=  cysub(y,ny);   // l low
	int yh = cyadd(y,ny);   // h high
	for(int z = 0; z < nz; z++){
		int zl = cysub(z,nz);
		int zh = cyadd(z,nz);
		int x = 2*xt+(y+z+colour)%2;  // 3D chess board offset (0,0,0) is white
		int xl = cysub(x,nx2);
		int xh = cyadd(x,nx2);

		float sum = spin[idx(z,y,xl)] + spin[idx(z,y,xh)] + spin[idx(z,yl,x)] +
			spin[idx(z,yh,x)] + spin[idx(zl,y,x)] + spin[idx(zh,y,x)];

		float energy = 2.0f*sum*(float)spin[idx(z,y,x)];
		if(energy <= 0.0f || expf(-energy/temp) >= curand_uniform(&myState)) {
			spin[idx(z,y,x)] = -spin[idx(z,y,x)];
		}
	}
	state[id] = myState;
}

int main(int argc,char *argv[])
{
	if(argc< 2){
		printf("Usage ising nx|256 ny|256 nz|256 seed|0=random Temp|4.4 iter|100 save|1 tx|32 ty|8 view|0 wait|0\n\n");
		printf("Interactive mode controlled by parameters view and wait\n");
		printf("If view > 0 the program displays the mid slice for every view'th iterataion and waits wait ms\n");
		printf("The following keys are active while waiting:\n\n");
		printf(" s: save the current volume\n");
		printf(" c: set temperature to critial value\n");
		printf(" +: raise temperture by tstep\n");
		printf(" =: raise temperture by tstep\n");
		printf(" -: lower temperture by tstep\n");
		printf(" ESC: exit program\n");
		return 0;
	}
	std::random_device rd;   // many user settable parameters here
	int nx =       (argc >1) ? atoi(argv[1]) : 256;   // image nx
	int ny =       (argc >2) ? atoi(argv[2]) : nx;    // image ny
	int nz =       (argc >3) ? atoi(argv[3]) : nx;    // image nz
	llong seed =   (argc >4) ? atoll(argv[4]) : rd(); // seed
	seed =   (seed > 0) ? seed : rd();          // 0 => random seed
	double temp =  (argc >5) ? atof(argv[5]) : 4.4;   // temperature
	int steps =    (argc >6) ? atoi(argv[6]) : 100;   // Ising iterations
	int dosave =   (argc >7) ? atoi(argv[7]) : 1;     // save result
	uint threadx = (argc >8) ? atoi(argv[8]) : 32;    // x threads
	uint thready = (argc >9) ? atoi(argv[9]) : 8;     // y threads
	uint view   = (argc >10) ? atoi(argv[10]) : 0;    // view progress ?
	uint wait   = (argc >11) ? atoi(argv[11]) : 0;    // ms delay per frame

	if(nx%2==1)   nx += 1; // force nx even
	int nxby2   = nx/2;    // index for checkerboard
	int volsize = nx*ny*nz;
	int slice   = nx*ny;

	double tc = 4.5115;  // critical temperature
	double tstep = 0.2;  // temperature step for +/- key

	// define thread blocks
	dim3 threads ={threadx,thready,1};
	dim3 blocks ={(nxby2+threads.x-1)/threads.x,(ny+threads.y-1)/threads.y,1};
	int statesize = blocks.x*blocks.y*blocks.z*threads.x*threads.y*threads.z;

	thrustDvec<char> dspin(volsize); // device 3D spin array
	thrustHvec<char> hspin(volsize); // host   3D spin array
	thrustDvec<char> dslice(slice);  // x-y slice of
	thrustHvec<char> hslice(slice);  // spin array

	// openCV image
	Mat view_image(ny,nx,CV_8UC1,Scalar(0));
	if(view) namedWindow("Spins",WINDOW_NORMAL |
		WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);

	// device buffer for random generator state for each threads.
	curandState *dstates;  // load random number states to device array
	cx::ok(cudaMalloc((void **)&dstates,statesize*sizeof(curandState)));

	cx::timer tim;
	setup_randstates<<<blocks,threads>>>(dstates,nxby2,ny,seed);
	cudaDeviceSynchronize();

	double state_time = tim.lap_ms();
	init_spins<char><<<blocks,threads>>>(dstates, dspin.data().get(), nxby2, ny, nz);
	cudaDeviceSynchronize();
	double init_time = tim.lap_ms();

	for(int k = 0; k<steps; k++){
		flip_spins<char><<<blocks,threads>>>(dstates, dspin.data().get(), nxby2, ny, nz, temp, 0);                        // white squares
		flip_spins<char><<<blocks,threads>>>(dstates, dspin.data().get(), nxby2, ny, nz, temp, 1);                        // black squares
		if(view > 0 && (k%view==0 || k==steps-1)){
			thrust::copy(dspin.begin()+slice*nz/2, dspin.begin()+(slice+slice*nz/2), hslice.begin()); // copy slice at z=nx/2
			for(int i=0;i<slice;i++) view_image.data[i] = hslice[i];
			imshow("Spins",view_image);
			char key = (waitKey(wait) & 0xFF); // mini event loop here
			if(key == ESC) break;
			else if(key =='s'){
				char name[256];
				sprintf(name,"ising_%d_%.3f.png",k,temp);
				imwrite(name,view_image);
			}
			else if(key== '+' || key == '='){ temp += tstep; printf("temp %.3f\n",temp); }
			else if(key== '-'){
				temp = std::max(0.0,temp-tstep);
				printf("temp %.3f\n",temp);
			}
			else if(key== 'c'){ temp = tc;     printf("temp %.3f\n",temp); }
		}
	}      // end spin uupdate loop over k
	cudaDeviceSynchronize();
	double flip_time = tim.lap_ms();
	printf("timing setup %.3f ms, init %.3f ms, fliping %.3f ms\n",state_time, init_time, flip_time);
	if(dosave > 0){
		hspin = dspin;
		cx::write_raw<char>("ising.raw",hspin.data(),volsize);
	}

	std::atexit([]{cudaDeviceReset();});
	return 0;
}


