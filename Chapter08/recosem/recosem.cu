// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// RTX 2070
// C:\bin\recosem.exe derenzo_full.raw derenzo petSM.raw petSMtab.raw 16 1
// using blocks = 144000, thread = 400
// file petSMtab.raw read
// sm_size = 12829371, lor_size 167232000 vol_size 2560000 osem steps 1
// file petSM.raw read
// file derenzo_full.raw read
// subsums:  1.565370e+14
// normalization done for 100 rings and 64 slices
// file smnorm_osem1.raw written
// Host normalize call 999.295 ms
// total activity 33232349840, activity density 264455
// iteration   1 chi 6.187530e+16 times times fwd 1814.443 bwd 1709.669 rsc 0.162 chi 0.315 all 3524.609 ms
// iteration   2 chi 5.228257e+16 times times fwd 3565.640 bwd 3435.638 rsc 0.325 chi 0.580 all 7002.405 ms
// iteration   3 chi 4.512538e+16 times times fwd 5319.682 bwd 5164.483 rsc 0.494 chi 0.904 all 10485.965 ms
// iteration   4 chi 3.954602e+16 times times fwd 7071.938 bwd 6893.930 rsc 0.655 chi 1.232 all 13968.304 ms
// iteration   5 chi 3.499340e+16 times times fwd 8828.670 bwd 8630.334 rsc 0.808 chi 1.539 all 17462.106 ms
// iteration   6 chi 3.122855e+16 times times fwd 10586.103 bwd 10368.698 rsc 0.967 chi 1.838 all 20958.545 ms
// iteration   7 chi 2.809372e+16 times times fwd 12345.819 bwd 12105.367 rsc 1.130 chi 2.137 all 24455.590 ms
// iteration   8 chi 2.546026e+16 times times fwd 14104.933 bwd 13847.816 rsc 1.258 chi 2.389 all 27957.725 ms
// file derenzo_osem1_008.raw written
// iteration   9 chi 2.322437e+16 times times fwd 15876.183 bwd 15595.249 rsc 1.418 chi 2.630 all 31482.704 ms
// iteration  10 chi 2.130530e+16 times times fwd 17646.921 bwd 17344.460 rsc 1.568 chi 2.927 all 35003.253 ms
// iteration  11 chi 1.964148e+16 times times fwd 19411.732 bwd 19093.713 rsc 1.714 chi 3.206 all 38517.914 ms
// iteration  12 chi 1.818611e+16 times times fwd 21178.513 bwd 20842.837 rsc 1.871 chi 3.510 all 42034.455 ms
// iteration  13 chi 1.690341e+16 times times fwd 22947.247 bwd 22595.606 rsc 2.020 chi 3.771 all 45556.560 ms
// iteration  14 chi 1.576562e+16 times times fwd 24716.255 bwd 24350.408 rsc 2.177 chi 4.068 all 49081.017 ms
// iteration  15 chi 1.475094e+16 times times fwd 26486.944 bwd 26104.555 rsc 2.336 chi 4.369 all 52606.516 ms
// iteration  16 chi 1.384195e+16 times times fwd 28257.554 bwd 27862.078 rsc 2.489 chi 4.660 all 56135.263 ms
// file derenzo_osem1_016.raw written
// All time 56140.820  ref time 56122.121 ms
// 
// C:\bin\recosem.exe derenzo_full.raw derenzo sysmat_osem2.raw systab_osem2.raw 8 2
// . . . 
// total activity 33232349840, activity density 264455
// iteration   1 chi 5.228652e+16 times times fwd 1758.570 bwd 1689.545 rsc 0.284 chi 0.286 all 3448.714 ms
// iteration   2 chi 3.953583e+16 times times fwd 3466.887 bwd 3387.164 rsc 0.601 chi 0.596 all 6855.494 ms
// iteration   3 chi 3.121194e+16 times times fwd 5178.496 bwd 5087.280 rsc 0.896 chi 0.891 all 10267.987 ms
// iteration   4 chi 2.544178e+16 times times fwd 6890.244 bwd 6792.599 rsc 1.215 chi 1.187 all 13685.902 ms
// iteration   5 chi 2.128623e+16 times times fwd 8604.706 bwd 8499.228 rsc 1.466 chi 1.441 all 17107.751 ms
// iteration   6 chi 1.816672e+16 times times fwd 10321.038 bwd 10204.918 rsc 1.785 chi 1.734 all 20530.568 ms
// iteration   7 chi 1.574618e+16 times times fwd 12035.324 bwd 11912.712 rsc 2.083 chi 2.034 all 23953.443 ms
// iteration   8 chi 1.382286e+16 times times fwd 13750.757 bwd 13619.856 rsc 2.376 chi 2.338 all 27376.798 ms
// file derenzo_osem2_008.raw written
// All time 27382.953  ref time 27372.990 ms
// 
// C:\bin\recosem.exe derenzo_full.raw derenzo sysmat_osem4.raw systab_osem4.raw 4 4
// . . .
// total activity 33232349840, activity density 264455
// iteration   1 chi 3.953037e+16 times times fwd 2352.356 bwd 1991.640 rsc 0.580 chi 0.314 all 4344.945 ms
// iteration   2 chi 2.544103e+16 times times fwd 4660.160 bwd 3986.881 rsc 1.220 chi 0.619 all 8649.212 ms
// iteration   3 chi 1.816969e+16 times times fwd 6966.157 bwd 5985.276 rsc 1.852 chi 0.910 all 12954.741 ms
// iteration   4 chi 1.382747e+16 times times fwd 9273.665 bwd 7983.054 rsc 2.483 chi 1.171 all 17261.156 ms
// file derenzo_osem4_004.raw written
// All time 17266.996  ref time 17259.203 ms
// 
// C:\bin\recosem.exe derenzo_full.raw derenzo sysmat_osem8.raw systab_osem8.raw 2 8
// . . .
// total activity 33232349840, activity density 264455
// iteration   1 chi 2.588363e+16 times times fwd 3352.960 bwd 2630.162 rsc 1.197 chi 0.314 all 5984.710 ms
// iteration   2 chi 1.457712e+16 times times fwd 6686.587 bwd 5255.009 rsc 2.451 chi 0.619 all 11945.004 ms
// file derenzo_osem8_002.raw written
// All time 11950.913  ref time 11944.048 ms
// 
// C:\bin\recosem.exe derenzo_full.raw derenzo sysmat_osem16.raw systab_osem16.raw 1 16
// . . .
// total activity 33232349840, activity density 264455
// iteration   1 chi 2.840100e+16 times times fwd 4104.525 bwd 3136.454 rsc 2.430 chi 0.306 all 7243.864 ms
// file derenzo_osem16_001.raw written
// All time 7249.627  ref time 7243.410 ms
// 
// // RTX 3080
// C:\bin\recosem.exe derenzo_full.raw derenzo petSM.raw petSMtab.raw 16 1
// using blocks = 144000, thread = 400
// file petSMtab.raw read
// sm_size = 12829371, lor_size 167232000 vol_size 2560000 osem steps 1
// file petSM.raw read
// file derenzo_full.raw read
// subsums:  1.565370e+14
// normalization done for 100 rings and 64 slices
// file smnorm_osem1.raw written
// Host normalize call 828.052 ms
// total activity 33232349840, activity density 264455
// iteration   1 chi 6.187530e+16 times times fwd 1038.039 bwd 950.314 rsc 0.075 chi 0.146 all 1988.601 ms
// iteration   2 chi 5.228257e+16 times times fwd 2067.875 bwd 1908.261 rsc 0.145 chi 0.268 all 3976.726 ms
// iteration   3 chi 4.512538e+16 times times fwd 3099.054 bwd 2866.957 rsc 0.217 chi 0.403 all 5966.938 ms
// iteration   4 chi 3.954602e+16 times times fwd 4129.771 bwd 3826.685 rsc 0.314 chi 0.539 all 7957.758 ms
// iteration   5 chi 3.499340e+16 times times fwd 5160.812 bwd 4787.454 rsc 0.379 chi 0.660 all 9949.915 ms
// iteration   6 chi 3.122855e+16 times times fwd 6192.798 bwd 5750.106 rsc 0.468 chi 0.778 all 11944.892 ms
// iteration   7 chi 2.809372e+16 times times fwd 7225.676 bwd 6714.966 rsc 0.537 chi 0.900 all 13942.957 ms
// iteration   8 chi 2.546026e+16 times times fwd 8259.178 bwd 7678.818 rsc 0.611 chi 1.032 all 15940.664 ms
// file derenzo_osem1_008.raw written
// iteration   9 chi 2.322437e+16 times times fwd 9292.652 bwd 8644.369 rsc 0.685 chi 1.184 all 17944.761 ms
// iteration  10 chi 2.130530e+16 times times fwd 10325.773 bwd 9611.097 rsc 0.763 chi 1.329 all 19944.970 ms
// iteration  11 chi 1.964148e+16 times times fwd 11359.615 bwd 10578.548 rsc 0.843 chi 1.452 all 21946.589 ms
// iteration  12 chi 1.818611e+16 times times fwd 12393.908 bwd 11545.442 rsc 0.917 chi 1.580 all 23948.213 ms
// iteration  13 chi 1.690341e+16 times times fwd 13427.411 bwd 12513.739 rsc 0.988 chi 1.701 all 25950.346 ms
// iteration  14 chi 1.576562e+16 times times fwd 14460.803 bwd 13482.502 rsc 1.056 chi 1.822 all 27952.828 ms
// iteration  15 chi 1.475094e+16 times times fwd 15495.831 bwd 14451.201 rsc 1.126 chi 1.943 all 29956.889 ms
// iteration  16 chi 1.384195e+16 times times fwd 16529.299 bwd 15420.368 rsc 1.200 chi 2.076 all 31959.823 ms
// file derenzo_osem1_016.raw written
// All time 31964.520  ref time 31950.867 ms
// 
// C:\bin\recosem.exe derenzo_full.raw derenzo sysmat_osem2.raw systab_osem2.raw 8 2
// . . .
// total activity 33232349840, activity density 264455
// iteration   1 chi 5.228653e+16 times times fwd 1012.457 bwd 921.863 rsc 0.134 chi 0.143 all 1934.640 ms
// iteration   2 chi 3.953583e+16 times times fwd 2029.202 bwd 1847.760 rsc 0.274 chi 0.291 all 3877.727 ms
// iteration   3 chi 3.121194e+16 times times fwd 3045.915 bwd 2773.802 rsc 0.395 chi 0.428 all 5820.889 ms
// iteration   4 chi 2.544178e+16 times times fwd 4062.415 bwd 3701.982 rsc 0.523 chi 0.565 all 7765.994 ms
// iteration   5 chi 2.128623e+16 times times fwd 5079.362 bwd 4630.461 rsc 0.645 chi 0.696 all 9711.775 ms
// iteration   6 chi 1.816672e+16 times times fwd 6096.136 bwd 5559.541 rsc 0.774 chi 0.816 all 11658.027 ms
// iteration   7 chi 1.574618e+16 times times fwd 7112.694 bwd 6488.967 rsc 0.894 chi 0.952 all 13604.381 ms
// iteration   8 chi 1.382286e+16 times times fwd 8129.204 bwd 7417.417 rsc 1.015 chi 1.072 all 15549.740 ms
// file derenzo_osem2_008.raw written
// All time 15554.599  ref time 15547.636 ms
// 
// C:\bin\recosem.exe derenzo_full.raw derenzo sysmat_osem4.raw systab_osem4.raw 4 4
// . . .
// total activity 33232349840, activity density 264455
// iteration   1 chi 3.953037e+16 times times fwd 1426.341 bwd 1111.731 rsc 0.237 chi 0.145 all 2538.509 ms
// iteration   2 chi 2.544103e+16 times times fwd 2853.612 bwd 2224.320 rsc 0.474 chi 0.267 all 5078.913 ms
// iteration   3 chi 1.816969e+16 times times fwd 4281.897 bwd 3336.456 rsc 0.717 chi 0.405 all 7619.861 ms
// iteration   4 chi 1.382747e+16 times times fwd 5709.329 bwd 4449.495 rsc 0.951 chi 0.542 all 10160.882 ms
// file derenzo_osem4_004.raw written
// All time 10165.683  ref time 10159.774 ms
// 
// C:/bin\recosem.exe derenzo_full.raw derenzo sysmat_osem8.raw systab_osem8.raw 2 8
// . . .
// total activity 33232349840, activity density 264455
// iteration   1 chi 2.588363e+16 times times fwd 1915.805 bwd 1502.125 rsc 0.462 chi 0.155 all 3418.650 ms
// iteration   2 chi 1.457712e+16 times times fwd 4081.965 bwd 3336.741 rsc 0.923 chi 0.275 all 7420.237 ms
// file derenzo_osem8_002.raw written
// All time 7424.839  ref time 7419.629 ms
//
// C:\bin\recosem.exe derenzo_full.raw derenzo sysmat_osem16.raw systab_osem16.raw 1 16
// . . .
// total activity 33232349840, activity density 264455
// iteration   1 chi 2.840100e+16 times times fwd 3201.487 bwd 2843.830 rsc 0.909 chi 0.150 all 6046.564 ms
// file derenzo_osem16_001.raw written
// All time 6051.453  ref time 6046.225 ms

// recosem program from section 8.7
// The is a version of reco modifed to use OSEM
// This version expects a fixed thread block size of 400 (cryNum)

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"
#include "scanner.h"


__host__ __device__ int c2_to_dc2(cint c1,cint c2) {
	return cyc_sub(c2,c1)-cryDiffMin;
}

__host__ __device__ int  zdz_slice(int z)
{
	return detZdZNum - (zNum-z)*(zNum-z+1)/2;
}

// assumes threads = cryNum i.e. 400 so that one thread blocks process all phis for fixed sm value
__global__  void forward_project(cr_Ptr<smPart> sm,uint smstart,uint smend,cr_Ptr<float> vol,int ring,r_Ptr<float> FP,int dzcut,float valcut)
{
	int phi = threadIdx.x;
	uint smpos = smstart+blockIdx.x;
	while(smpos < smend) {
		smLor tl = key2lor(sm[smpos].key);
		tl.c1 = cyc_add(tl.c1,phi);     // rotate by phi		
		tl.c2 = cyc_add(tl.c2,phi);     // rotate by phi
		int dc = c2_to_dc2(tl.c1,tl.c2);         // sm has actual c2 not delta	
		int dz = tl.zsm1+tl.zsm2;
		float val= sm[smpos].val;
		uint lor_index = zdz_slice(dz)*cryCdCNum + dc*cryNum + tl.c1;
		uint vol_index = (ring*zNum + tl.zsm1)*cryNum + phi;      // z+z1 here as voxel index
		for(int zs1 = 0; zs1 < zNum-dz; zs1++) {  // zs1 is sliding posn of lh end of lor
			float element = vol[vol_index] * val;
			atomicAdd(&FP[lor_index],element);
			lor_index += cryCdCNum;  // for zs1++
			vol_index += cryNum;     // for zs1++
		}
		smpos += gridDim.x;
	}
}

// assumes threads = cryNum i.e. 400 so that one thread blocks process all phis for fixed sm value
__global__  void backward_project(cr_Ptr<smPart> sm,uint smstart,uint smend,cr_Ptr<uint> meas,int ring,cr_Ptr<float> FP,r_Ptr<float> BP,int dzcut,float valcut)
{
	int phi = threadIdx.x;
	uint smpos = smstart+blockIdx.x;

	while(smpos < smend) {
		smLor tl = key2lor(sm[smpos].key);
		tl.c1 = cyc_add(tl.c1,phi);     // rotate by phi		
		tl.c2 = cyc_add(tl.c2,phi);     // rotate by phi
		int dc = c2_to_dc2(tl.c1,tl.c2);         // sm has actual c2 not delta		
		int dz = tl.zsm1+tl.zsm2;  // net delta z
		float val= sm[smpos].val;

		uint lor_index = zdz_slice(dz)*cryCdCNum + dc*cryNum + tl.c1;  // new july 6 
		uint vol_index = (ring*zNum + tl.zsm1)*cryNum + phi;       // z1+zs1 here as voxel index
		for(int zs1 = 0; zs1 < zNum-dz; zs1++) {  // zs1 is sliding posn of lh end of lor
			float FPdiv = max(1.0f,FP[lor_index]);
			float element = val * meas[lor_index] / FPdiv;
			atomicAdd(&BP[vol_index],element);
			lor_index += cryCdCNum;  // for zs1++
			vol_index += cryNum;     // for zs1++
		}
		smpos += gridDim.x;
	}
}

__global__ void rescale(r_Ptr<float> vol,cr_Ptr<float> BP,cr_Ptr<float> norm,int ostep)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int offset = ostep*radNum*zNum;
	while(id < zNum*radNum*cryNum){

		vol[id] *= BP[id] / norm[offset+id/cryNum];
		id += blockDim.x*gridDim.x;
	}
}

__global__ void calc_chisd(r_Ptr<float> vol,cr_Ptr<float> gold,r_Ptr<float> chisd)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int ids = id;
	float sum = 0.0f;
	while(ids < zNum*radNum*cryNum){
		sum  += (vol[ids]-gold[ids])*(vol[ids]-gold[ids]);  // += if grid size < vol size
		ids += blockDim.x*gridDim.x;
	}
	chisd[id] = sum;
}


template <typename T> __global__ void clear_vector(r_Ptr<float> a,uint len)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	while(id < len){
		a[id] = (T)0;
		id += blockDim.x*gridDim.x;
	}
}

int normalise_sm(thrustHvec<smPart> &sm,thrustHvec<float> &norm,thrustHvec<smTab> &systab,int dzcut,float valcut,int osteps)
{
	uint nslice = radNum*zNum;
	thrustHvec<double> subsum(osteps);
	float sum = 0.0;
	for(int r = 0;r<radNum*osteps;r++){
		int subset = r/radNum;
		for(uint k=systab[r].start;k<systab[r].end;k++){
			smLor tl = key2lor(sm[k].key);
			float val = sm[k].val;
			int dz = tl.zsm2+tl.zsm1;
			if(dz > dzcut || val < valcut) continue;
			for(int z = tl.zsm1; z < zNum - tl.zsm2; z++)  {
				norm[r*zNum+z] += val;	//vertex posn here
				subsum[subset] += val;
				sum += val;
			}
		}
	}
	printf(" subsums: "); for(int k=0;k<osteps;k++)printf(" %5e",subsum[k]); printf("\n");
	printf("normalization done for %d rings and %d slices\n",radNum,zNum);
	for(int set=0;set < osteps;set++){
		//float scale = sum/(subsum[set]*sysMatScale);  // approx osteps * 10^-11;
		//printf("osems scale %d %5e\n",set,scale);
		for(uint k=0;k<nslice;k++){
			//norm[set*nslice+k] *= scale;
			norm[set*nslice+k] /= sysMatScale;
			if(osteps > 2)norm[set*nslice+k] *= 0.5f*(float)osteps;  // only 4 needs scaling why?
		}
	}

	//cx::write_raw("norm_new.raw",norm.data(),nslice*osteps);
	//cx::write_raw("norm_recip.raw",norm.data(),norm_size);
	return 0;
}

// save direclty are Cartesian 200x200x64 image
int pol_save(const char* name, float* vol)
{
	struct cp_grid_map {
		float b[voxBox][voxBox];
		int x; // carteisian origin
		int y;
		int phi;  // polar voxel
		int r;
	};

	//int pol_size =  cryNum*zNum*radNum;  // NB order [ring, z, phi]
	int cart_size = voxNum*voxNum*zNum;  //          [2*z,    y,   x]
	int map_size =  cryNum*radNum;       //          [ring, phi]

	std::vector<float>       cart(cart_size);
	std::vector<cp_grid_map>  map(map_size);
	if(cx::read_raw("pol2cart.tab",map.data(),map_size,0)){printf("bad read on pol2cart.tab\n"); return 1;}
	for(int r=0;r<radNum;r++) for(int z=0;z<zNum;z++) for(int p=0;p<cryNum;p++){
		float val = vol[(r*zNum+z)*cryNum+p];

		float vol_fraction =  1.0f;  //2*r+1;
		int index = r*cryNum+p;
		if(val > 0.0f){
			int x0 = map[index].x;
			int y0 = map[index].y;
			for(int i=0;i<voxBox;i++) {
				int y = y0+i;
				if(y>=0 && y<voxNum) for(int j= 0;j<voxBox;j++){
					int x = x0+j;
					if(x>=0 && x <voxNum && map[index].b[i][j]>0.0f) cart[(z*voxNum+y)*voxNum+x] += vol_fraction*val*map[index].b[i][j];
				}
			}
		}
	}
	cx::write_raw(name,cart.data(),cart_size);
	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage reco <pet file (phantom)> <result file> <sm file> <sm tab file> <iterations> <osem|8> [ dzcut|63] valcut|0  blocks|18000\n");
		return 0;
	}

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	int blscale  = 1024;
	int thscale  = 256;
	int niter    = (argc> 5) ? atoi(argv[5]) : 10;
	int osteps   = (argc >6) ? atoi(argv[6]) : 8;
	int dzcut    = (argc> 7) ? dzcut = atoi(argv[7]) : 63;
	float valcut = (argc> 8) ? valcut = atof(argv[8]) : 0.0f;
	int threads  = cryNum;
	int blocks   = (argc> 9) ? atoi(argv[9]): 144000; 
	printf("using blocks = %d, thread = %d\n",blocks,threads);

	// set up system matix
	char name[256];

	// this for new sysmat_tab
	thrustHvec<smTab> systab(radNum*osteps);
	if(cx::read_raw(argv[4],systab.data(),radNum*osteps)){ printf("bad read %s\n",argv[4]); return 1; }

	uint sm_size = systab[radNum*osteps-1].end;
	uint lor_size = cryCdCNum*detZdZNum;
	uint vol_size = cryNum*radNum*zNum;
	uint norm_size = radNum*zNum*osteps;
	uint zphi_size = cryNum*zNum;

	printf("sm_size = %u, lor_size %u vol_size %u osem steps %d\n",sm_size,lor_size,vol_size,osteps);

	thrustHvec<smPart>      sm(sm_size);
	thrustDvec<smPart>  dev_sm(sm_size);
	if(cx::read_raw(argv[3],sm.data(),sm_size)) { printf("bad read on sm_file %s\n",argv[3]); return 1; }
	dev_sm = sm;

	thrustHvec<uint>      meas(lor_size);
	thrustDvec<uint>  dev_meas(lor_size);
	if(cx::read_raw(argv[1],meas.data(),lor_size)) { printf("bad read on pet file %s\n",argv[1]); return 1; }
	dev_meas = meas;

	thrustHvec<float>     FP(lor_size); // working space for forward projection (voxels => lors)
	thrustDvec<float> dev_FP(lor_size);

	thrustHvec<float>     BP(vol_size); // working space for backward projection  (lors => voxels)
	thrustDvec<float> dev_BP(vol_size);

	thrustHvec<float>     vol(vol_size);
	thrustDvec<float> dev_vol(vol_size);

	thrustHvec<float>     norm(norm_size); // voxel normaliztions depend on ring and z
	thrustDvec<float> dev_norm(norm_size);

	int usegold = 1;
	thrustHvec<float> gold(vol_size);    // gold standard answer from simuation
	thrustDvec<float> dev_gold(vol_size);   // a chisqd will be calculated if the
	thrustDvec<float> dev_chisd(vol_size);  // file gold.raw is successfully read	
	if( cx::read_raw("gold.raw",gold.data(),vol_size,0) ) usegold = 0;  
	if (usegold) dev_gold = gold; 



	cx::timer ntim;
	ntim.start();

	normalise_sm(sm,norm,systab,dzcut,valcut,osteps); // new version each subset requires its own normaisation
	sprintf(name,"smnorm_osem%d.raw",osteps);  // optional save
	cx::write_raw(name,norm.data(),norm_size); // here
	dev_norm = norm;
	ntim.add();
	printf("Host normalize call %.3f ms\n",ntim.time());

	double tot_activity = 0.0;
	for(uint k = 0; k < lor_size; k++) tot_activity += meas[k];

	// new initialisation accounting for voxel volumes (makes little difference)
	float roi_volume = cx::pi<float>*roiRadius*roiRadius;
	float act_density = tot_activity/roi_volume;
	//float act_pervox = tot_activity/vol_size;
	float r1 = 0.0f;
	float r2 = voxSize;
	for(int r=0;r<radNum;r++){
		float dr2 = r2*r2-r1*r1;
		float voxvol = cx::pi<float>*dr2/cryNum;
		for(uint k=0;k<zphi_size;k++) vol[r*zphi_size+k] = act_density*voxvol;
		r1 = r2;
		r2 += voxSize;
	}

	dev_vol = vol;
	printf("total activity %.0f, activity density %.0f\n",tot_activity,act_density);

	cx::timer tim1;
	cx::timer tim2;
	cx::timer tim3;
	cx::timer tim4;
	cx::timer all;

	FILE *flog = fopen("osem.log","a");
	all.reset();

	for(int iter = 0;iter< niter;iter++) {
		for(int ostep=0;ostep<osteps;ostep++) {
			clear_vector<float><<<blscale,thscale>>>(dev_FP.data().get(),lor_size);
			clear_vector<float><<<blscale,thscale>>>(dev_BP.data().get(),vol_size);
			//cx::ok(cudaDeviceSynchronize());
			tim1.start();
			for(int r = radNum*ostep; r < radNum*(ostep+1); r++) {
				forward_project<<<blocks,threads>>>(dev_sm.data().get(),systab[r].start,systab[r].end,dev_vol.data().get(),r%radNum,dev_FP.data().get(),dzcut,valcut);
				//cx::ok(cudaDeviceSynchronize());
			}
			cx::ok(cudaDeviceSynchronize());
			tim1.add();
			tim2.start();
			for(int r = radNum*ostep; r < radNum*(ostep+1); r++) {
				backward_project<<<blocks,threads>>>(dev_sm.data().get(),systab[r].start,systab[r].end,dev_meas.data().get(),r%radNum,dev_FP.data().get(),dev_BP.data().get(),dzcut,valcut);
			}
			cx::ok(cudaDeviceSynchronize());
			tim2.add();
			tim3.start();
			rescale<<<blscale,thscale>>>(dev_vol.data().get(),dev_BP.data().get(),dev_norm.data().get(),ostep);
			cx::ok(cudaDeviceSynchronize());
			tim3.add();
			cx::ok(cudaDeviceSynchronize());
			all.add();
		}
		tim4.start();
		float xhisd = 0.0f;
		if(usegold) {
			calc_chisd<<<blscale, thscale>>>(dev_vol.data().get(), dev_gold.data().get(), dev_chisd.data().get());
			cx::ok(cudaDeviceSynchronize());
			xhisd = thrust::reduce(dev_chisd.begin(), dev_chisd.end());
		}
		tim4.add();
		all.add();
		printf("iteration %3d chi %5e times times fwd %.3f bwd %.3f rsc %.3f chi %.3f all %.3f ms\n",iter+1,xhisd,tim1.time(),tim2.time(),tim3.time(),tim4.time(),all.time());
		fprintf(flog," %3d  %3d %5e %.3f\n",osteps,iter,xhisd,all.time());
		int iout = iter+1;
		//if(iout<=5 || (iout<=10 && iout%2==0) || iout==15 || (iout<=40 && iout%10==0) || iout%50==0 || iout==niter ){  // this for osem2
		//if(iout==1 || iout==2 || iout==4|| iout==8 || iout==16 || iout==32 || iout==64 || iout==128 || iout==niter ){  // this for osem2
		if(iout%8==0 || iout == niter){  // optional progress monitor
			vol = dev_vol;
			//sprintf(name,"%s_osem%d_%3.3d.raw",argv[2],osteps,iout);
			//cx::write_raw(name,vol.data(),vol_size);
			sprintf(name,"%s_cart%d_%3.3d.raw",argv[2],osteps,iout);
			pol_save(name, vol.data());
		}

	}

	all.add();
	double reft = tim1.time()+tim2.time()+tim3.time();
	printf("All time %.3f  ref time %.3f ms\n",all.time(),reft);
	fprintf(flog," \n");
	fclose(flog);
	//vol = dev_vol;
	//sprintf(name,"%s_%d_final.raw",argv[2],niter);
	//cx::write_raw(name, vol.data(), vol_size);

	//sprintf(name,"FPbug%3.3d.raw",niter);
	//FP = dev_FP;
	//cx::write_raw(name, FP.data(), lor_size);

	//BP = dev_BP;
	//sprintf(name,"BPbug%3.3d.raw",niter);
	//cx::write_raw(name, BP.data(), vol_size);

	return 0;
}
