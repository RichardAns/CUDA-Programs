// recosem program from chapter 8
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

	thrustHvec<float> gold(vol_size);
	thrustDvec<float> dev_gold(vol_size);
	thrustDvec<float> dev_chisd(vol_size);
	cx::read_raw("gold.raw",gold.data(),vol_size);
	dev_gold = gold;

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
		calc_chisd<<<blscale,thscale>>>(dev_vol.data().get(),dev_gold.data().get(),dev_chisd.data().get());
		cx::ok(cudaDeviceSynchronize());
		float xhisd = thrust::reduce(dev_chisd.begin(),dev_chisd.end());
		tim4.add();
		all.add();
		printf("iteration %3d chi %5e times times fwd %.3f bwd %.3f rsc %.3f chi %.3f all %.3f ms\n",iter+1,xhisd,tim1.time(),tim2.time(),tim3.time(),tim4.time(),all.time());
		fprintf(flog," %3d  %3d %5e %.3f\n",osteps,iter,xhisd,all.time());
		int iout = iter+1;
		//if(iout<=5 || (iout<=10 && iout%2==0) || iout==15 || (iout<=40 && iout%10==0) || iout%50==0 || iout==niter ){  // this for osem2
		//if(iout==1 || iout==2 || iout==4|| iout==8 || iout==16 || iout==32 || iout==64 || iout==128 || iout==niter ){  // this for osem2
		if(iout%8==0 || iout == niter){  // optional progress monitor
			//if(iter==niter-1){
			sprintf(name,"%s_osem%d_%3.3d.raw",argv[2],osteps,iout);
			vol = dev_vol;
			cx::write_raw(name,vol.data(),vol_size);
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
