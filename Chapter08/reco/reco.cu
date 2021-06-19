// reco program includes examples 8.7 and 8.8.

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"
#include "scanner.h"


__host__ __device__ int c2_to_dc2(cint c1,cint c2) {
	return cyc_sub(c2,c1)-cryDiffMin;
}

//NB this can be called with either z1 or (z2-z1) as argument
//   steps in the other variable will then be adjacent in memory
//   Using (z2-z1) are argument turns out to be a bit faster.
__host__ __device__ int  zdz_slice(int z)
{
	return detZdZNum - (zNum-z)*(zNum-z+1)/2;
}

// Host version of example 8.7
// assumes threads = cryNum i.e. 400 so that one thread blocks processes all the phis for single sm value
int forward_project_host(smPart *sm,uint smstart,uint smend,uint *meas,float* vol,int ring,float* FP,int dzcut,float valcut)
{
	uint smpos = smstart;
	//uint count = 0;
	while(smpos < smend) {
		for(int phi=0;phi<cryNum;phi++){  // for loop over threads
			smLor tl = key2lor(sm[smpos].key);
			tl.c1 = cyc_add(tl.c1,phi);     // rotate by phi		
			tl.c2 = cyc_add(tl.c2,phi);     // rotate by phi
			if(tl.zsm1==0 && tl.zsm2==0 && tl.c2 <= tl.c1) continue;  // skip unused case
			int dc = c2_to_dc2(tl.c1,tl.c2);         // sm has actual c2 not delta	
			int dz = tl.zsm1+tl.zsm2;
			float val= sm[smpos].val;
			if(dz > dzcut || val <valcut) break;
			uint lor_index = zdz_slice(dz)*cryCdCNum + dc*cryNum + tl.c1;
			uint vol_index = (ring*zNum + tl.zsm1)*cryNum + phi;      // z+z1 here as voxel index
			for(int zs1 = 0; zs1 < zNum-dz; zs1++) {  // zs1 is sliding posn of lh end of lor 
				if(meas[lor_index]>0){
					float element = vol[vol_index] * val;
					FP[lor_index] += element;
				}
				lor_index += cryCdCNum;  // for zs1++
				vol_index += cryNum;     // for zs1++
			}
		}   // end phi loop
		smpos++;  // one host thread
	}
	return 0;
}

// Host version of example 8.8
int backward_project_host(smPart* sm,uint smstart,uint smend,uint *meas,int ring,float* FP,float* BP,int dzcut,float valcut)
{
	uint smpos = smstart;
	while(smpos < smend) {
		for(int phi=0;phi<cryNum;phi++){  // for loop over threads
			smLor tl = key2lor(sm[smpos].key);
			tl.c1 = cyc_add(tl.c1,phi);     // rotate by phi		
			tl.c2 = cyc_add(tl.c2,phi);     // rotate by phi
			if(tl.zsm1==0 && tl.zsm2==0 && tl.c2 <= tl.c1) continue; // skip unused case
			int dc = c2_to_dc2(tl.c1,tl.c2);         // sm has actual c2 not delta		
			int dz = tl.zsm1+tl.zsm2;  // net delta z
			float val= sm[smpos].val;
			if(dz > dzcut || val < valcut) break;
			uint lor_index = zdz_slice(dz)*cryCdCNum + dc*cryNum + tl.c1;  // new july 6 
			uint vol_index = (ring*zNum + tl.zsm1)*cryNum + phi;       // z1+zs1 here as voxel index
			for(int zs1 = 0; zs1 < zNum-dz; zs1++) {  // zs1 is sliding posn of lh end of lor 
				if(meas[lor_index]>0){
					//if(FP[lor_index == 0.0f]) printf("zero FP  found index %u\n",lor_index);
					float element = val * meas[lor_index] / FP[lor_index];  // val added 27/06/19!!
					BP[vol_index] += element;
				}
				lor_index += cryCdCNum;  // for zs1++
				vol_index += cryNum;     // for zs1++
			}
		}
		smpos++;
	}
	return 0;
}

int rescale_host(r_Ptr<float> vol,cr_Ptr<float> BP,cr_Ptr<float> norm)
{
	for(int id=0;id<zNum*radNum;id++){
		float scale = 1.0f/norm[id];
		for(int phi=0;phi<cryNum;phi++) vol[id*cryNum+phi] *= BP[id*cryNum+phi]*scale;
	}
	return 0;

}

// example 8.7 (GPU version)
// Uses thread linear addressing for flexible thread bock sizes.
__global__  void forward_project(cr_Ptr<smPart> sm,uint smstart,uint smend,cr_Ptr<float> vol,int ring,r_Ptr<float> FP,int dzcut,float valcut)
{
	// This version uses thread linar addressing to allow tuning experiments
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	uint tstride = gridDim.x*blockDim.x;
	int nphi = (smend-smstart)*cryNum;
	while(id < nphi) {
		int phi =id%cryNum;                // these two lines added for
		int smpos = smstart+ id/cryNum;    // thread linear addressing
		smLor tl = key2lor(sm[smpos].key);
		tl.c1 = cyc_add(tl.c1,phi);        // rotate by phi		
		tl.c2 = cyc_add(tl.c2,phi);        // rotate by phi
		int dc = c2_to_dc2(tl.c1,tl.c2);   // sm has actual c2 not delta	
		int dz = tl.zsm1+tl.zsm2;
	
		float val= sm[smpos].val;  // system matrix value
		if(dz > dzcut || val <valcut) { smpos += gridDim.x; continue; }

		uint lor_index = zdz_slice(dz)*cryCdCNum + dc*cryNum + tl.c1;
		uint vol_index = (ring*zNum + tl.zsm1)*cryNum + phi;      // z+z1 here as voxel index
		for(int zs1 = 0; zs1 < zNum-dz; zs1++) {  // zs1 is sliding posn of lh end of lor
			float element = vol[vol_index] * val;
			atomicAdd(&FP[lor_index],element);
			lor_index += cryCdCNum;  // for zs1++
			vol_index += cryNum;     // for zs1++
		}
		id += tstride;  // replaces smpos+=gridDim.x for thread linear addressing case
	}
}

// Example 8.8 (GPU version)
// Uses thread linear addressing for flexible thread bock sizes.
__global__  void backward_project(cr_Ptr<smPart> sm,uint smstart,uint smend,cr_Ptr<uint> meas,int ring,cr_Ptr<float> FP,r_Ptr<float> BP,int dzcut,float valcut)
{
	// This version uses thread linar addressing to allow tuning experiments
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	uint tstride = gridDim.x*blockDim.x;
	int nphi = (smend-smstart)*cryNum;
	while(id < nphi) {
		int phi =id%cryNum;                 // these two lines added for
		int smpos = smstart+id/cryNum;      // thread linear addressing
		smLor tl = key2lor(sm[smpos].key);
		tl.c1 = cyc_add(tl.c1,phi);         // rotate by phi		
		tl.c2 = cyc_add(tl.c2,phi);         // rotate by phi
		int dc = c2_to_dc2(tl.c1,tl.c2);    // sm has actual c2 not delta		
		int dz = tl.zsm1+tl.zsm2;  
	
		float val= sm[smpos].val;   // system matrix value
		if(dz > dzcut || val < valcut) { smpos += gridDim.x; continue; }

		uint lor_index = zdz_slice(dz)*cryCdCNum + dc*cryNum + tl.c1;  // new july 6 
		uint vol_index = (ring*zNum + tl.zsm1)*cryNum + phi;       // z1+zs1 here as voxel index
		for(int zs1 = 0; zs1 < zNum-dz; zs1++) {  // zs1 is sliding posn of lh end of lor 	
			float FPdiv = max(1.0f,FP[lor_index]);
			float element = val * meas[lor_index] / FPdiv;
			atomicAdd(&BP[vol_index],element);
			lor_index += cryCdCNum;  // for zs1++
			vol_index += cryNum;     // for zs1++
		}
		id += tstride;  // replaces smpos+=gridDim.x for thread linear addressing case
	}
}

__global__ void rescale(r_Ptr<float> vol,cr_Ptr<float> BP,cr_Ptr<float> norm)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	while(id < zNum*radNum*cryNum){

		vol[id] *= BP[id] / norm[id/cryNum];
		id += blockDim.x*gridDim.x;
	}
}

__global__ void calc_chisd(r_Ptr<float> vol,cr_Ptr<float> gold,r_Ptr<float> chisd)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	while(id < zNum*radNum*cryNum){
		chisd[id] = (vol[id]-gold[id])*(vol[id]-gold[id]);
		id += blockDim.x*gridDim.x;
	}
}

template <typename T> __global__ void clear_vector(r_Ptr<float> a,uint len)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	while(id < len){
		a[id] = (T)0;
		id += blockDim.x*gridDim.x;
	}
}

//int normalise_sm(thrustHvec<smPart> &sm,thrustHvec<float> &norm,thrustHvec<uint> &smhits,thrustHvec<uint> &smstart, int dzcut, float valcut)
int normalise_sm(thrustHvec<smPart> &sm,thrustHvec<float> &norm,thrustHvec<smTab> &systab,int dzcut,float valcut)
{
	// NB all dZ and dC cuts assumed to have already been made in readspot
	uint norm_size = radNum*zNum;
	// normalise allowing for voxel volume Router^2 - Rinner^2
	for(int r = 0;r<radNum;r++){
		//uint sm_start = smstart[r];
		//uint smnum = smhits[r+2];
		//for(uint k=sm_start;k<sm_start+smnum;k++){
		for(uint k=systab[r].start;k<systab[r].end;k++){
			smLor tl = key2lor(sm[k].key);
			float val = sm[k].val;
			int dz = tl.zsm2+tl.zsm1;  // changed 9/7/19
			if(dz > dzcut || val < valcut) continue;
			for(int z = tl.zsm1; z < zNum - tl.zsm2; z++)  norm[r*zNum+z] += val;	  //vertex posn here
		}
	}
	printf("normalization done for %d rings and %d slices\n",radNum,zNum);
	for(uint i=0;i<norm_size;i++) norm[i] /= sysMatScale;  // assume 10^10 generations per voxel
	cx::write_raw("norm_new.raw",norm.data(),norm_size);
	//cx::write_raw("norm_recip.raw",norm.data(),norm_size);
	return 0;
}

int list_sm(thrustHvec<smPart> &sm,thrustHvec<uint> &smhits,thrustHvec<uint> &smstart)
{
	printf("list sm called\n");
	for(int r=0;r<radNum;r++){
		printf("list sm called r=%d\n",r);
		char name[256];
		sprintf(name,"smlist_r%3.3d.txt",r);
		FILE * flog = fopen(name,"w");
		uint sm_start = smstart[r];
		uint smnum = smhits[r+2];
		for(uint k=sm_start;k<sm_start+smnum;k++){
			smLor tl = key2lor(sm[k].key);
			float val = sm[k].val;
			fprintf(flog,"smpos %6u lor (%2d %3d)-(%2d %3d) val %.0f\n",k,tl.zsm1,tl.c1,tl.zsm2,tl.c2,val);
		}
		fclose(flog);
	}
	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage reco <pet file (phantom)> <result file> <sm file> <sm tab file> <iterations> [ dzcut|63] valcut|0 usehost|0] blocks|5000 threads|400 rmin|0 rmax|100\n");
		return 0;
	}

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	int blscale = 1024;
	int thscale = 256;
	int niter    = (argc > 5) ? atoi(argv[5]) : 10;
	int dzcut    = (argc > 6) ? atoi(argv[6]) : 63;
	float valcut = (argc > 7) ? atof(argv[7]) : 0.0;
	int usehost  = (argc > 8) ? atoi(argv[8]) : 0; 
	if(usehost != 0) printf("WARNING HOST code being used for this run\n");
	
	// set up system matix
	char name[256];

	int blocks  = (argc> 9) ? atoi(argv[9])*36 : 5000*36;  // 36 =  number of sm units on gpu
	int threads = (argc > 10) ? atoi(argv[10]) : cryNum;
	printf("using blocks = %d, threads = %d valcut %.1f dzcut %d\n",blocks,threads,valcut,dzcut);

	int rmin  = (argc > 11) ? atoi(argv[11]) : 0;
	int rmax = (argc > 12) ? atoi(argv[12]) : 100;

	// this for sysmat_tab
	thrustHvec<smTab> systab(radNum);  // start and end indicies for individual rings within sysmat file
	if(cx::read_raw(argv[4],systab.data(),radNum)){ printf("bad read %s\n",argv[4]); return 1; }

	uint sm_size = systab[radNum-1].end;
	uint lor_size = cryCdCNum*detZdZNum;
	uint vol_size = cryNum*radNum*zNum;
	uint norm_size = radNum*zNum;
	uint zphi_size = cryNum*zNum;
	printf("sm_size = %u, lor_size %u vol_size %u\n",sm_size,lor_size,vol_size);

	thrustHvec<smPart>      sm(sm_size);  // the system matrix (sysmat) file generated by fullsim + readspot
	thrustDvec<smPart>  dev_sm(sm_size);
	if(cx::read_raw(argv[3],sm.data(),sm_size)) { printf("bad read on sysmat file %s\n",argv[3]); return 1; }
	dev_sm = sm;

	thrustHvec<uint>      meas(lor_size);  // the PET measured lor file, either real mesurements 
	thrustDvec<uint>  dev_meas(lor_size);  // or simulated phantom data
	if(cx::read_raw(argv[1],meas.data(),lor_size)) { printf("bad read on PET measurment file %s\n",argv[1]); return 1; }
	dev_meas = meas;

	thrustHvec<float>     FP(lor_size); // working space for forward projection (voxels => lors)
	thrustDvec<float> dev_FP(lor_size);

	thrustHvec<float>     BP(vol_size); // working space for backward projection  (lors => voxels)
	thrustDvec<float> dev_BP(vol_size);

	thrustHvec<float>     vol(vol_size); // The PET voxels to be calculated
	thrustDvec<float> dev_vol(vol_size);

	int usegold = 1;
	thrustHvec<float> gold(vol_size);                     // gold standard answer from simuation
	thrustDvec<float> dev_gold(vol_size);                 // a chisqd will be calculated if the
	thrustDvec<float> dev_chisd(vol_size);                // file gold.raw is succefully read
	if( cx::read_raw("gold.raw",gold.data(),vol_size,0) ) usegold = 0;
	if(usegold) dev_gold = gold;
	thrustHvec<float>     norm(norm_size); // voxel normaliztions depends on both ring and z values
	thrustDvec<float> dev_norm(norm_size);


	cx::timer ntim;
	ntim.start();

	// due to z-sliding of sm elements normaisation of sm elements requires care
	// the required factors are calculated here every time but could be read from
	// the saved file to save time.
	normalise_sm(sm,norm,systab,dzcut,valcut);
	cx::write_raw("smnorm.raw",norm.data(),norm_size);
	dev_norm = norm;
	ntim.add();
	printf("Host normalize call %.3f ms\n",ntim.time());

	double tot_activity = 0.0;
	for(uint k = 0; k < lor_size; k++) tot_activity += meas[k];

	//float mean_activity = tot_activity / vol_size;
	//for (uint k = 0; k < vol_size; k++) vol[k] = mean_activity;

	// new initialisation accounting for voxel volumes (makes little difference)
	float roi_volume = cx::pi<float>*roiRadius*roiRadius;
	float act_density = tot_activity/roi_volume;

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
	//cx::write_raw("reco_start_vol.raw",vol.data(),vol_size); // debug

	cx::timer tim1;
	cx::timer tim2;
	cx::timer tim3;
	cx::timer tim4;
	cx::timer all;

	all.reset();
	if(usehost)for(int iter = 0;iter< niter;iter++){
		if(iter>0){
			std::fill(FP.begin(),FP.end(),0);
			std::fill(BP.begin(),BP.end(),0);
		}
		tim1.reset();
		for(int r = rmin; r < rmax; r++) {
			forward_project_host(sm.data(),systab[r].start,systab[r].end,meas.data(),vol.data(),r,FP.data(),dzcut,valcut);
		}
		tim1.add();
		tim2.reset();
		for(int r = rmin; r < rmax; r++) {
			backward_project_host(sm.data(),systab[r].start,systab[r].end,meas.data(),r,FP.data(),BP.data(),dzcut,valcut);
		}
		tim2.add();
		tim3.reset();
		rescale_host(vol.data(),BP.data(),norm.data());
		tim3.add();

		all.add();
		printf("host iteration %3d times fwd %.3f bwd %.3f rsc %.3f all %.3f ms\n",iter+1,tim1.time(),tim2.time(),tim3.time(),all.time());
	}

	else for(int iter = 0;iter< niter;iter++){
		if(iter>0){
			clear_vector<float><<<blscale,thscale>>>(dev_FP.data().get(),lor_size);
			clear_vector<float><<<blscale,thscale>>>(dev_BP.data().get(),vol_size);
		}
		tim1.start();

		for(int r = rmin; r < rmax; r++) {
			forward_project<<<blocks,threads>>>(dev_sm.data().get(),systab[r].start,systab[r].end,dev_vol.data().get(),r,dev_FP.data().get(),dzcut,valcut);
		}
		cx::ok(cudaDeviceSynchronize());
		tim1.add();
		tim2.start();

		for(int r = rmin; r < rmax; r++) {
			backward_project<<<blocks,threads>>>(dev_sm.data().get(),systab[r].start,systab[r].end,dev_meas.data().get(),r,dev_FP.data().get(),dev_BP.data().get(),dzcut,valcut);
		}
		cx::ok(cudaDeviceSynchronize());
		tim2.add();
		tim3.start();
		rescale<<<blscale,thscale>>>(dev_vol.data().get(),dev_BP.data().get(),dev_norm.data().get());
		cx::ok(cudaDeviceSynchronize());
		tim3.add();
		// debug detailed progess check
		//vol = dev_vol;  
		//sprintf(name,"%s%3.3d.raw",argv[2],iter+1);
		//cx::write_raw(name, vol.data(), vol_size);
		cx::ok(cudaDeviceSynchronize());

		tim4.start();
		float xhisd  = 0.0f;
		if(usegold){
			calc_chisd<<<blscale,thscale>>>(dev_vol.data().get(),dev_gold.data().get(),dev_chisd.data().get());
			cx::ok(cudaDeviceSynchronize());
			xhisd = thrust::reduce(dev_chisd.begin(),dev_chisd.end());
		}
		tim4.add();

		all.add();
		printf("iteration %3d chi %5e times fwd %.3f bwd %.3f rsc %.3f chi %.3f all %.3f ms\n",iter+1,xhisd,tim1.time(),tim2.time(),tim3.time(),tim4.time(),all.time());
		int iout = iter+1;
		//if(iout<=5 || (iout<=10 && iout%2==0) || (iout<=50 && iout%10==0) || iout%50==0 || iout==niter ){  // long runs
		if(iout==niter){
			sprintf(name,"%s_mlem%3.3d.raw",argv[2],iout);
			vol = dev_vol;
			cx::write_raw(name,vol.data(),vol_size);
		}
	}

	all.add();
	printf("All time %.3f ms\n",all.time());

	// these for debug 
	//if(!usehost)vol = dev_vol;
	//sprintf(name,"%s_%d_mlem.raw",argv[2],niter);
	//cx::write_raw(name, vol.data(), vol_size);

	//sprintf(name,"FPbug%3.3d.raw",niter);
	//FP = dev_FP;
	//cx::write_raw(name, FP.data(), lor_size);

	//BP = dev_BP;
	//sprintf(name,"BPbug%3.3d.raw",niter);
	//cx::write_raw(name, BP.data(), vol_size);

	return 0;
}
