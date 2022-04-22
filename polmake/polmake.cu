// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// chapter 8 polmake
// This is a support program for chapter 8 PET simualtion
// It generates the lookup table required to convert polar voxels into 
// convetional catiesian voxels. Only the transverse plane is transformed
// i.e. (r,phi) goes to (x,y). Although an analyical calculation is possible
// it is much simpler to use an MC method which on the GPU is very fast.
//
// This is a good example of where knowing a bit of CUDA can be very helpful
// for specific one-off tasks.

// RTX 2070
// C:\bin\polmake.exe  polcav.tab 123456 10000000
// done time 10309.635 ms
// file polcav.tab written
// 
// RTX 3080
// C:\bin\polmake.exe pol2cart.tab 123456 10000000
// done time 7063.882 ms
// file pol2cart.tab written

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"
#include "scanner.h"  // 

#include "curand_kernel.h"
#include <random>

// For the case of N equally spaced rings this program spans the
// ciruclar ROI in the tranverse plane with a square 2N x 2N Cartesian grid
// just containing the ROI. 
//
// N is set to radNum defined as 100 in scanner.h
// other parametrs defined in scanner.h are
//
// voxBox = 5;                      -- a 5 x 5 cartesian box spanning a polar voxel
// voxBoxOffset = voxBox/2;         -- spanning box offset from polar voxel
// voxNum  =  radNum*2;             -- 2N 
// voxSize  = 2.0f*roiRadius/voxNum -- size of Cartesian grid square
// voxStep  = 1.0f/voxSize          -- 1/size         


struct cp_grid {  
	uint b[voxBox][voxBox];   // 5x5 Cartesian box spanning one polar voxel
	uint good;
	uint bad;
	int x; //  Box Cartesian origin
	int y; //
	int phi;  // polar voxel
	int r;    // coodinates
};

struct cp_grid_map {         // smaller version just for output file.
	float b[voxBox][voxBox];
	int x; // Box Cartesian origin
	int y; //
	int phi;  // polar voxel
	int r;    // position
};

template <typename S> __global__ void init_generator(long long seed,S *states)
{
	// minimal version
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed + id,0,0,&states[id]);
	//curand_init(seed, id , 0, &states[id]);  
}

// this kernel uses one thread per polar voxel. Each thread fills its spanning 5x5 grid with
// hits from random points generated uniformly within that polar voxel.
template <typename S> __global__ void cpfill(float * cgrid,cp_grid * cp,int voxNum, uint tries,S *states)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	cp_grid g;
	for(int i=0;i<voxBox;i++)for(int j=0;j<voxBox;j++)g.b[i][j] = 0;
	g.good = 0;
	g.bad = 0;
	g.phi = threadIdx.x;  // tid = phi
	g.r =   blockIdx.x;   // bid = r

	float p1 = (float)threadIdx.x*cx::pi2<float>/(float)cryNum;

	float p2 = p1 + cx::pi2<float>/(float)cryNum;
	float pmean = 0.5f*(p1+p2);
	float r1 = (float)blockIdx.x*voxSize;
	//float r1 = (float)myr*voxSize;
	float r2 = r1+voxSize;
	float rmean = 0.5f*(r1+r2);
	float x = rmean*cosf(pmean)+roiRadius;   //cos/sin swap 05/07/19
	float y = rmean*sinf(pmean)+roiRadius;
	int xc = (int)(x/voxSize+0.5f);
	int yc = (int)(y/voxSize+0.5f);
	g.x = max(0,min(xc-voxBoxOffset,voxNum-voxBox));
	g.y = max(0,min(yc-voxBoxOffset,voxNum-voxBox));

	float r1sq = r1*r1;
	float r2sq = r2*r2-r1sq;
	float dphi = cx::pi2<float>/(float)cryNum;
	S state = states[id];  // get state
	for(uint k=0;k<tries;k++){
		// generate decay point at a
		float phi = p1 + dphi*curand_uniform(&state);
		float r =   sqrtf(r1sq +r2sq*curand_uniform(&state));
		float x = r*cosf(phi) + roiRadius;
		float y = r*sinf(phi) + roiRadius;
		//printf("generated x/y %.3f %.3f\n",x,y);
		if(x>=0 && x <= 2.0f*roiRadius && y>=0 && y <= 2.0f*roiRadius){
			int ix = (int)(x/voxSize)-g.x;
			int iy = (int)(y/voxSize)-g.y;
			if(ix<0 || ix >= voxBox || iy<0 || iy >= voxBox) {
				// printf("out of box error (%3d %3d) good %d p/r %.3f %.2f x/y %.2f %.2f ixy %d %d gxy %d %d\n",threadIdx.x,blockIdx.x,g.good,phi,r,x,y,ix,iy,g.x,g.y);
				g.bad++;
				continue;
			}
			else g.b[iy][ix]++;
			g.good++;
		}
		else {
			// this may appear if spanning grid is too small (5x5 is fine for our case)
			printf("out of roi error (%3d %3d) good %d p/r %.3f %.2f xy %.2f %.2f \n",threadIdx.x,blockIdx.x,g.good,phi,r,x,y);
			g.bad++;
			break;
		}
	}
	cp[id] = g;
	states[id] = state;  // save state so can continue
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage >polmake.exe <poluse file> <seed|rd()> <ngen|1000000>  domaps|0 myp|0 myr|0\n");
		printf("parameters myp and myr select polar voxel for display if either is non zero\n");
		return 0;
	}


	std::random_device rd;
	long long seed = (argc > 2) ? atoi(argv[2]) : rd();
	long long ngen = (argc > 3) ? atoll(argv[3]) : 1000000;
	int domaps     = (argc > 4) ? atoi(argv[4]) : 0;
	int myp        = (argc > 5) ? atoi(argv[5]) : 0;
	int myr        = (argc > 6) ? atoi(argv[6]) : 0;


	//int voxNum = (int)(voxNum + 0.5f);
	int csize = voxNum*voxNum;
	thrustHvec<float>          cgrid(csize);  // 2D cartesian grid
	thrustDvec<float>      dev_cgrid(csize);  //

	int psize = cryNum*radNum;                   // number of 2D polar voxels
	thrustHvec<cp_grid>          cpgrid(psize);  // for 5x5 mini grid around each polar voxel
	thrustDvec<cp_grid>      dev_cpgrid(psize);

	int threads = cryNum; // fixed launch parametrs
	int blocks =  radNum; // 400 x 100 traverse plane polar voxels

	// use XORWOW
	thrustDvec<curandState> states(psize);  // this for curand_states
	cx::timer tim;
	init_generator<<<blocks,threads>>>(seed,states.data().get());
	cpfill<<<blocks,threads>>>(dev_cgrid.data().get(),dev_cpgrid.data().get(),voxNum,ngen,states.data().get());
	checkCudaErrors(cudaDeviceSynchronize());
	tim.add();

	
	cpgrid =dev_cpgrid;

	printf("done time %.3f ms\n",tim.time());
	thrustHvec<cp_grid_map>          cpmap(psize);  // mini grid around each r/phi voxel
	for(int k=0;k<psize;k++){
		float div = 1.0f/(float)(cpgrid[k].good);
		cpmap[k].phi = cpgrid[k].phi;
		cpmap[k].r   = cpgrid[k].r;
		cpmap[k].x   = cpgrid[k].x;
		cpmap[k].y   = cpgrid[k].y;
		for(int i=0;i<voxBox;i++) for(int j=0;j<voxBox;j++) cpmap[k].b[i][j] = (float)cpgrid[k].b[i][j]*div;
	}

	cx::write_raw(argv[1],cpmap.data(),psize);

	// this for debug display map for selected polar voxel
	if(myr != 0 || myp != 0){
		int index = myr*cryNum+myp;
		printf("box[%3d][%3d] good %d bad %u\n",myr,myp,cpgrid[index].good,cpgrid[index].bad);
		printf("\n        "); for(int i=0;i<voxBox;i++) printf("   %3d   ",cpgrid[index].x+i); printf("\n");
		for(int i=0;i<voxBox;i++){
			printf(" %3d ",cpgrid[index].y+i);
			for(int j=0;j<voxBox;j++) printf(" %8u",cpgrid[index].b[i][j]);
			printf("\n");
		}

		printf("\n        "); for(int i=0;i<voxBox;i++) printf("   %3d   ",cpmap[index].x+i); printf("\n");
		for(int i=0;i<voxBox;i++){
			printf(" %3d ",cpmap[index].y+i);
			for(int j=0;j<voxBox;j++) printf(" %8.6f",cpmap[index].b[i][j]);
			printf("\n");
		}
	}


	// for debug 
	// the output file is a stack of 5x5 slices showing local map for each polar voxel.
	// The file can be inspected with ImageJ
	if(domaps){
		int vsize = voxBox*voxBox;
		thrustHvec<float> maps(psize*vsize);
		for(int k=0;k<psize;k++) for(int i=0;i<voxBox;i++) for(int j=0;j<voxBox;j++) maps[k*vsize+i*voxBox+j] =cpmap[k].b[i][j];
		cx::write_raw("polmake_maps.raw",maps.data(),psize*vsize);
	}

	return 0;
}