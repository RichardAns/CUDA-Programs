// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// examples 8.1-8.4 The fullsim program
// 
// 8.1 - 8.4 (fullsim)
// 
// RTX 2070
// C:\bin\fullsim.exe 1 256 1024  100000 123456 50 51 0 1
// Detector: len 508.0 radius 254.648 rings 127 crytals/ring 400 zstep 4.000 phistep 0.900 (degrees)
// Roi  phi (  0.000   0.016) r (  100.0   102.0) z (  252.0   256.0) a (    0.79   101.00   254.00)
// file spot_map050.raw written
// ngen 100007936000 good 46433964145 eff 46.430% tries 3815 passes 100 gen time 19719.253 io time // 87.248 total 19810.513 ms
// done
// 
// RTX 3080
// C:\bin\fullsim.exe 1 256 1024  100000 123456 50 51 0 1
// Detector: len 508.0 radius 254.648 rings 127 crytals/ring 400 zstep 4.000 phistep 0.900 (degrees)
// Roi  phi (  0.000   0.016) r (  100.0   102.0) z (  252.0   256.0) a (    0.79   101.00   254.00)
// file spot_map050.raw written
// ngen 100007936000 good 46433964145 eff 46.430% tries 3815 passes 100 gen time 6247.259 io time // 58.241 total 6306.808 ms
// done

// Fullsim is a complete MC program with the modes of operation selected with first parameter
// 
// mode 1:
//   Generate polar voxel spot files needed to build system matrix as discussed in chapter 8.
//
//                1      2       3       4    5     6    7      8        9   
//   Arguments: mode, threads, blocks, ngen, seed, vx1, vx2, savelors, savemap 
//      ngen:  number of generation in millions
//      seed: starting seed for MC
//      vx1 and vx2: range of radial voxel, typically 0 100 for 100 radial voxels
//      savelors: if > 0 save the complete lor data. These are big files > 2GB for each voxel
//      savemap:  if > 0 save compressed spot maps for each voxel
//
//   e.g > fullsim.exe 1 256 1024 10000 123456 0 100 0 1
//
//   Note the output files have fixed names are are overwritten if fullsim is rerun.
//
// mode 2:
//   Generate phantom data set as lor file. Currently only cylindrical voulumes are
//   supported by the code can easily be extended to in include cubes and spheres as required
//   The output file is appended to if it already exists. This allows a phantom with multiple 
//   cylinders to be built and as an example associated Derenzo program builds a script file to do this.
//
//                1      2       3       4    5        6        7 - 8    9 - 10     11 -12  13 - 15    16       17
//   Arguments:  mode, threads, blocks, ngen, seed, outputname, r-range, phi-range, z-range, offsets, savevol, savelors
//     r-range: r-min and r-max for cylinder, if r-min is non zero a hollow cylinder is generated
//     phi-range:  range of phi vales, normally 0 and 360 for a full cylinder
//     z-range: z-min and z-max
//     offsets: displacement of x and y from centre and additional z displacement from zero.
//     savevol: save generated point in 3D cartesian volume, used for debugging.
//     savelors: save lors to file names in argument 6. The file is updated if it already exits.
//
//     e.g. fullsim.exe 2 256 1024 93 1493 derenzo_vol.raw   0 3.06   0 360   64 192   -6.12 41.59 0   0 1
//
// mode 3:
//   A host version of mode 1 generation for timing comparisons. Spot file compressions is not implemented.
//
//                1      2     3    4    5     6           
//   Arguments: mode,  ngen, seed, vx1, vx2, savelors
// 
//   e.g. fullsim.exe 3 1 123456  30 31 0
//
//  Note the host version only works correctly for vx2 = vx1+1


#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxconfun.h"
#include "scanner.h"
#include "helper_math.h"     // for lerp
#include "curand_kernel.h"   // for cuRand
#include <random>

// modified ray_to_cyl accumulates singles map in short detector for debug
__device__ int ray_to_cyl_singles(r_Ptr<uint> singles, Ray& g)
{
	//swim to cyl: solve quadratic Ax^2+2Bx+C = 0
	float A = g.n.x*g.n.x + g.n.y*g.n.y;
	float B = g.a.x*g.n.x + g.a.y*g.n.y;  // factors of 2 ommited as they cancel
	float C = g.a.x*g.a.x + g.a.y*g.a.y - detRadius*detRadius;
	float D = B*B-A*C;
	float rad = sqrtf(D);
	g.lam1 = (-B+rad)/A;  // gamma1
	float z1 = g.a.z+g.lam1*g.n.z;
	g.lam2 = (-B-rad)/A;  // gamma2
	float z2 = g.a.z+g.lam2*g.n.z;

	if (z1 >= 0.0f && z1 < detLen) {
		float x1  = g.a.x+g.lam1*g.n.x;
		float y1  = g.a.y+g.lam1*g.n.y;
		float phi = myatan2(x1, y1);
		int iz1 =  (int)(z1*detStep);
		int ic1 =  phi2cry(phi);
		int index=(iz1*cryNum)+ ic1;
		atomicAdd(&singles[index], 1);
	}
	if (z2 >= 0.0f && z2 < detLen) {
		float x2  = g.a.x+g.lam2*g.n.x;
		float y2  = g.a.y+g.lam2*g.n.y;
		float phi = myatan2(x2,y2);
		int iz2 = (int)(z2*detStep);  
		int ic2 = phi2cry(phi);                 
		int index=(iz2*cryNum)+ ic2;
		atomicAdd(&singles[index], 1);
	}
	return 0; 
}


// this is example 8.3 in text
__host__ __device__ int ray_to_cyl(Ray &g,Lor &l,float length)  //15/6/19 added length argument  
{
	//swim to cyl: solve quadratic Ax^2+2Bx+C = 0
	float A = g.n.x*g.n.x + g.n.y*g.n.y;
	float B = g.a.x*g.n.x + g.a.y*g.n.y;  // factors of 2 ommited as they cancel
	float C = g.a.x*g.a.x + g.a.y*g.a.y - detRadius*detRadius;
	float D = B*B-A*C;
	float rad = sqrtf(D);
	g.lam1 = (-B+rad)/A;  // gamma1
	float z1 = g.a.z+g.lam1*g.n.z;
	g.lam2 = (-B-rad)/A;  // gamma2
	float z2 = g.a.z+g.lam2*g.n.z;

	if(z1 >= 0.0f && z1 < length && z2 >= 0.0f && z2 < length && abs(z2-z1) < detLen){ // same zdiff short and long detectors
		float x1  = g.a.x+g.lam1*g.n.x;
		float y1  = g.a.y+g.lam1*g.n.y;
		float phi = myatan2(x1,y1);
		l.z1 =  (int)(z1*detStep); 
		l.c1 =  phi2cry(phi);       
		float x2  = g.a.x+g.lam2*g.n.x;
		float y2  = g.a.y+g.lam2*g.n.y;
		phi = myatan2(x2,y2);
		l.z2 = (int)(z2*detStep);  
		l.c2 = phi2cry(phi);                 

		if(l.z1 > l.z2){          // here z1 & z2 measured from LH end of scanner
			cx::swap(l.z1,l.z2);
			cx::swap(l.c1,l.c2);
		}
		return 1;  // success
	}

	return 0;  // failure
}

template <typename T> __global__ void clear_vec(T *vec,size_t count)
{
	size_t id = threadIdx.x + blockIdx.x*blockDim.x;
	while(id < count) {
		vec[id] = 0;
		id += gridDim.x*blockDim.x;
	}
}

template <typename S> __global__ void init_generator(long long seed,S *states)
{
	// minimal version
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed + id,0,0,&states[id]);
	//curand_init(seed, id , 0, &states[id]);  
}

__device__ __host__ float3 roi2xyz(Roi &v)
{
	float3 a;
	float phi = 0.5f*(v.phi.y+v.phi.x);   // averge phi
	float rxy = 0.5f*(v.r.y+v.r.x);       // distance from z axis
	a.x = rxy*sinf(phi);                  // centre of voxel
	a.y = rxy*cosf(phi);
	a.z = 0.5f*(v.z.y+v.z.x);           // average z
	printf("Roi  phi (%7.3f %7.3f) r (%7.1f %7.1f) z (%7.1f %7.1f) a (%8.2f %8.2f %8.2f)\n",v.phi.x,v.phi.y,v.r.x,v.r.y,v.z.x,v.z.y,a.x,a.y,a.z);
	return a;
}

// called with zNum blocks of cryNum threads
// this is example 8.4 in text
__global__ void find_spot(r_Ptr<uint> map,r_Ptr<uint> spot,Roi vox)
{
	float c1 = threadIdx.x;   // use floats for (z1,c1) as
	float z1 = blockIdx.x;    // they are used in calcuations
	if(c1 >= cryNum || z1 >= zNum) return;

	float3 a; // LH hit on scanner
	float phi = (cx::pi2<> / cryNum)*(c1 + 0.5f);
	a.x = detRadius * sinf(phi);  // phi = 0 along y axis
	a.y = detRadius * cosf(phi);  // and increases clockwise
	a.z = crySize * (z1 + 0.5f);  // use slice centre

	float3 b;  // decay point 
	phi = 0.5f*(vox.phi.y + vox.phi.x);   // averge phi
	float rxy = 0.5f*(vox.r.y + vox.r.x); // radial distance from z axis
	b.x = rxy * sinf(phi);                // centre of voxel
	b.y = rxy * cosf(phi);
	b.z = 0.5f*(vox.z.y + vox.z.x);     // average z

	// find lam such that a+ lam*b hits cyl
	float lam2 = -2.0f*(a.x*b.x + a.y*b.y - detRadius*detRadius) / ((b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y));
	float3 c; // centre of (z2,c2) cluster for (z1,c1) lor family
	c = lerp(a,b,lam2);
	phi = myatan2(c.x,c.y);
	int c2 = phi2cry(phi);  // This is prediced end point (z2,c2)
	int z2 = c.z / crySize; // for LOR with start (z1,c1)

	// extract 24x24 "spot" from map file 
	int zsm1 = zNum-1 - (int)z1;
	int zsm2 = z2 - zNum+1;
	zsm2 = clamp(zsm2,0,zNum-1); // zsm2 can be out of range

	// copy hits to spot map
	size_t m_slice = (zsm1*cryNum + c1)*mapSlice;           // (z1,c1) slice in map
	size_t s_slice = (zsm1*cryNum + c1)*spotNphi*spotNz;    // (z1,c1) slice in spotmap

	int sz = max(0,zsm2 - spotNz/2); // z offset
	for(int iz = 0; iz < spotNz; iz++) {
		int sc = cyc_sub(c2,spotNphi/2); // c offset
		for(int ic = 0; ic < spotNphi; ic++) {
			uint val = map[m_slice + sz * cryNum + sc];
			spot[s_slice + iz * spotNphi + ic] = val;
			sc = cyc_inc(sc);  // TODO error to wrap here
		}
		sz++;
		if(sz >= zNum) break;
	}

	// store offsets 
	spot[s_slice] = max(0,zsm2-spotNz/2);
	spot[s_slice + spotNphi] = cyc_sub(c2,spotNphi/2);

}

// this is example 8.2 in text
template <typename S> __global__ void voxgen(r_Ptr<uint> map,r_Ptr<double> ngood,Roi roi,S *states,uint tries)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];
	Ray g;
	Lor lor;
	uint good = 0;
	float r1sq = roi.r.x*roi.r.x;      // r1^2
	float r2sq = roi.r.y*roi.r.y-r1sq; // r2^2 - r1^2
	float dphi = roi.phi.y-roi.phi.x;  // phi range
	float dz =   roi.z.y-roi.z.x;      // z range
	for(uint k=0;k<tries;k++){
		// generate point in roi
		float phi = roi.phi.x + dphi*curand_uniform(&state);  // uniform in phi
		float r =   sqrtf(r1sq +r2sq*curand_uniform(&state)); // uniforam in annulus		
		g.a.x = r*sinf(phi);
		g.a.y = r*cosf(phi);
		g.a.z = roi.z.x + dz*curand_uniform(&state); // uniform in z

		// generate isotropic back to back gammas
		float phi_gam = cx::pi2<float>*curand_uniform(&state);     // use uniform for phi & acos for theta
		float theta_gam = acosf(1.0f-2.0f*curand_uniform(&state)); // for random 3D isotropic direction
		g.n.x = sinf(phi_gam)*sinf(theta_gam);
		g.n.y = cosf(phi_gam)*sinf(theta_gam);
		g.n.z = cosf(theta_gam);

		// find & save hits in scanner detectors
		if(ray_to_cyl(g,lor,detLongLen)){
			good++;
			uint zsm2 = max(0,lor.z2-zNum+1);
			uint zsm1 = max(0,zNum-lor.z1-1);
			uint index = (zsm1*cryNum+lor.c1)*mapSlice+(zsm2*cryNum+lor.c2);
			atomicAdd(&map[index],1);
		}
	}
	ngood[id] += good;
	states[id] = state;
}

//NB this can be called with either z1 or (z2-z1) as argument
//   steps in the other variable will then be adjacent in memory
//   Using (z2-z1) as argument turns out to be a bit faster.
__device__ int zdz_slice(int z)
{
	return detZdZNum - (zNum-z)*(zNum-z+1)/2;
}

// very similar to voxgen but used for large volume phantoms. The set of lors produced simulates data from
// scanning a real ojbect. The code can be extended to include other shapes and the addition of noise.
// It would also be possible to simualate scatter from a realistic patient volume.
template <typename S> __global__ void phantom(r_Ptr<uint> map,r_Ptr<uint> vfill,r_Ptr<uint> vfill2,r_Ptr<uint> singles,PRoi roi,r_Ptr<double> ngood,S *states,uint tries,int savevol,int dosing)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];

	Ray g;
	Lor lor;
	int good = 0;
	float r1sq = roi.r.x*roi.r.x;
	float r2sq = roi.r.y*roi.r.y-r1sq;
	float dphi = roi.phi.y-roi.phi.x;
	float dz =   roi.z.y-roi.z.x;
	int3 p;
	int ir;
	int iz;
	int iphi;
	for(uint k=0;k<tries;k++){
		// generate decay point at a
		float phi = roi.phi.x + dphi*curand_uniform(&state);
		float r =   sqrtf(r1sq +r2sq*curand_uniform(&state));
		g.a.x = r*sinf(phi) +roi.o.x;
		g.a.y = r*cosf(phi) +roi.o.y;
		g.a.z = roi.z.x + dz*curand_uniform(&state) + roi.o.z; // sum of z.x and o.z is z start value
		if(savevol){  // save generated decay points debug only
			p.x = (int)((g.a.x+roiRadius)*voxStep);
			p.y = (int)((g.a.y+roiRadius)*voxStep);
			p.z = (int)(g.a.z*detStep);
			p.z = min(zNum-1,p.z);
			if(p.x <0 || p.x >= voxNum || p.y < 0 || p.y >= voxNum || p.z < 0 || p.z >= zNum) printf("bad roi %d %d %d\n",p.x,p.y,p.z);
			else {
				ir = (int)(voxStep*sqrtf(g.a.x*g.a.x + g.a.y*g.a.y));
				iz = (int)(detStep*g.a.z);
				float phi2 = myatan2(g.a.x,g.a.y);
				iphi = (int)((cryNum*phi2)/cx::pi2<>);
				atomicAdd(&vfill[(ir*zNum+iz)*cryNum+iphi],1);  // this for polar
			}
		}
		// generate isotropic back to back gammas int +/- n direction
		phi = cx::pi2<float>*curand_uniform(&state);
		float theta = acosf(1.0f-2.0f*curand_uniform(&state));
		g.n.x = sinf(phi)*sinf(theta);
		g.n.y = cosf(phi)*sinf(theta);
		g.n.z = cosf(theta);
		if (dosing) ray_to_cyl_singles(singles, g); // generate singles map
		if(ray_to_cyl(g,lor,detLen)){
			int cdiff = abs(lor.c1-lor.c2);
			if(cdiff >= cryDiffMin && cdiff <=cryDiffMax){
				good++;
				if(savevol){ 
					atomicAdd(&vfill2[(ir*zNum+iz)*cryNum+iphi],1);  // this for polar
				}
				uint lor2fix = max(0,lor.z2);  lor2fix = min(lor2fix,63);
				uint lor1fix = max(0,lor.z1);  lor1fix = min(lor1fix,63);
				uint zdz =    zdz_slice(lor.z2-lor.z1) + lor.z1;     // zdz_slice(dz) + zs1 in reco (from July 6th)
				uint cdc =    cyc_sub(lor.c2,lor.c1) - cryDiffMin;
				uint index =  zdz*cryCdCNum+cdc*cryNum+lor.c1;

				atomicAdd(&map[index],1);
			}
		}
	} // done
	ngood[id] += good;
	states[id] = state;
}

int do_phantom(int argc,char *argv[])
{
	if(atoi(argv[1]) != 2) return 1;
	uint threads = atoi(argv[2]);
	uint blocks =  atoi(argv[3]);

	uint size = blocks*threads;
	int passes = 1;
	long long  ngen = 1000000;
	int ndo = atoi(argv[4]);
	if(ndo <1000) ngen *= (long long)ndo;
	else {
		double genall = ndo;
		genall *= ngen;
		while(genall/passes > 1.0e+09)passes++;
		ngen = (long long)(genall)/(double)passes;
	}

	uint tries = (ngen+size-1)/size;
	ngen = (long long)tries*(long long)size;
	long long ngen_all = ngen*(long long)passes;
	std::random_device rd;
	long long seed = rd(); if(atoi(argv[5]) > 0) seed = atoi(argv[5]);


	PRoi roi;
	// outfile in argv[6]
	roi.r.x      = (argc > 7) ? atof(argv[ 7]) : 0.0f;
	roi.r.y      = (argc > 8) ? atof(argv[ 8]) : 1.0f;
	roi.phi.x    = (argc > 9) ? atof(argv[ 9])*cx::pi<float>/180.0f : 0.0f;
	roi.phi.y    = (argc >10) ? atof(argv[10])*cx::pi<float>/180.0f : 2.0f*cx::pi<float>;
	roi.z.x      = (argc >11) ? atof(argv[11]) : 0.0f;
	roi.z.y      = (argc >12) ? atof(argv[12]) : 1.0f;
	roi.o.x      = (argc >13) ? atof(argv[13]) : 0.0f;
	roi.o.y      = (argc >14) ? atof(argv[14]) : 0.0f;
	roi.o.z      = (argc >15) ? atof(argv[15]) : 0.0f;
	int savevol  = (argc >16) ? atof(argv[16]) : 0;
	int savelors = (argc >17) ? atoi(argv[17]) : 1; 
	int dosing   = (argc >18) ? atoi(argv[18]) : 0; 

	printf("Phantom r (%.1f %.1f) p (%.3f %.3f) z (%.1f %.1f) o (%.1f %.1f %.1f) saves %d %d %d\n",
		roi.r.x,roi.r.y,roi.phi.x,roi.phi.y,roi.z.x,roi.z.y,roi.o.x,roi.o.y,roi.o.z,savevol,savelors,dosing);

	// use XORWOW
	thrustDvec<curandState> state(size);  // this for curand_states

	uint vsize = zNum*voxNum*voxNum;
	uint msize = mapSlice*mapSlice;
	uint zsize = cryNum*cryDiffNum*detZdZNum;
	uint ssize = cryNum*zNum;
	thrustHvec<uint>       vfill(vsize);
	thrustDvec<uint>   dev_vfill(vsize);
	thrustHvec<uint>       vfill2(vsize);
	thrustDvec<uint>   dev_vfill2(vsize);
	thrustHvec<uint>       map(msize);
	thrustDvec<uint>   dev_map(msize);
	thrustDvec<double> dev_good(size);
	thrustHvec<uint>       zdzmap(zsize);
	thrustDvec<uint>   dev_zdzmap(zsize);
	thrustHvec<uint>       singles(zsize);
	thrustDvec<uint>   dev_singles(zsize);


	// if file exists we append
	if(savelors != 0 && cx::can_be_opened(argv[6])){
		if(cx::read_raw(argv[6],zdzmap.data(),zsize) != 0) for(uint k=0;k<zsize;k++) zdzmap[k] = 0;
		else dev_zdzmap = zdzmap;
	}
	if(savevol != 0 && cx::can_be_opened("phant_roi_all.raw")){
		if(cx::read_raw("phant_roi_all.raw",vfill.data(),vsize) == 0) dev_vfill = vfill;
	}
	if(savevol != 0 && cx::can_be_opened("phant_roi_good.raw")){
		if(cx::read_raw("phant_roi_good.raw",vfill.data(),vsize) == 0) dev_vfill2 = vfill;
	}

	if(dosing != 0 && cx::can_be_opened("singles_roi_all.raw")){
		if (cx::read_raw("singles_roi_all.raw", singles.data(), ssize) == 0) dev_singles = singles;
	}

	cx::timer tim;
	init_generator<<<blocks,threads>>>(seed,state.data().get());
	for(int k=0;k<passes;k++){
		phantom<<<blocks,threads >>>(dev_zdzmap.data().get(),dev_vfill.data().get(),dev_vfill2.data().get(),
			dev_singles.data().get(),roi,dev_good.data().get(),state.data().get(),tries,savevol,dosing);
	}
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	tim.add();
	double all_good = thrust::reduce(dev_good.begin(),dev_good.end());
	double eff = 100.0*all_good/(double)ngen_all;

	printf("Phantom ngen %lld good %.0f eff %.3f%% time %.3f ms\n",ngen_all,all_good,eff,tim.time());

	if(savevol){
		vfill = dev_vfill;
		cx::write_raw("phant_roi_all.raw",vfill.data(),vsize);
		vfill = dev_vfill2; // NB
		cx::write_raw("phant_roi_good.raw",vfill.data(),vsize);
	}
	if(savelors){
		zdzmap = dev_zdzmap;
		cx::write_raw(argv[6],zdzmap.data(),zsize); // always append
	}
	if (dosing) {
		singles = dev_singles;
		cx::write_raw("singles_roi_all.raw", singles.data(), ssize);
	}

	return 0;
}

// simple host only version for timing comparisons
void hostgen(r_Ptr<uint> map,double &ngood,Roi roi,uint seed,uint ngen)
{
	std::default_random_engine gen(seed);
	std::uniform_int_distribution<int>  idist(0,2147483647); // uniform ints
	float idist_scale = 1.0/2147483648.0;

	Ray g;
	Lor lor;
	uint good = 0;
	float r1sq = roi.r.x*roi.r.x;      // r1^2
	float r2sq = roi.r.y*roi.r.y-r1sq; // r2^2 - r1^2
	float dphi = roi.phi.y-roi.phi.x;  // phi range
	float dz =   roi.z.y-roi.z.x;      // z range
	for(uint k=0;k<ngen;k++){
		// generate point in roi
		float phi = roi.phi.x + dphi*idist_scale*idist(gen);  // uniform in phi
		float r =   sqrtf(r1sq +r2sq*idist_scale*idist(gen)); // uniforam in annulus		
		g.a.x = r*sinf(phi);
		g.a.y = r*cosf(phi);
		g.a.z = roi.z.x + dz*idist_scale*idist(gen);; // uniform in z

		// generate isotropic back to back gammas
		float phi_gam = cx::pi2<float>*idist_scale*idist(gen);      // uniform phi & use acos for theta
		float theta_gam = acosf(1.0f-2.0f*idist_scale*idist(gen)); // for 3D isotropic directions
		g.n.x = sinf(phi_gam)*sinf(theta_gam);
		g.n.y = cosf(phi_gam)*sinf(theta_gam);
		g.n.z = cosf(theta_gam);

		// find & save hits in scanner detectors
		if(ray_to_cyl(g,lor,detLongLen)){
			good++;
			uint zsm2 = max(0,lor.z2-zNum+1);
			uint zsm1 = max(0,zNum-lor.z1-1);
			uint index = (zsm1*cryNum+lor.c1)*mapSlice+(zsm2*cryNum+lor.c2);
			//atomicAdd(&map[index],1);
			map[index]++;
		}
	}
	ngood += good;
}

int do_hostver(int argc,char *argv[])
{
	int ngen       = (argc> 2) ? 1000000*atoi(argv[2]) : 1000000;
	std::random_device rd;
	long long seed = (argc > 3) ? atoi(argv[3]) : rd();
	int vx1      =   (argc > 4) ? atoi(argv[4]) : 0;     // integer voxel start r rspan setis preset as voxSize
	int vx2      =   (argc > 5) ? atoi(argv[5]) : vx1+1;
	int savelors =   (argc > 6) ? atoi(argv[6]) : 0;


	printf("Detector: len %.1f radius %.3f rings %d crytals/ring %d zstep %.3f phistep %.3f (degrees)\n",
		detLongLen,detRadius,zLongNum,cryNum,crySize,360.0/cryNum);

	Roi roi;
	roi.r.x =  voxSize*vx1;
	roi.r.y =  roi.r.x+voxSize; 

	roi.z.x = crySize*(zNum-1);    // calib voxel spans z in range 63-64;
	roi.z.y =  roi.z.x+crySize;    // z span one detector ring;

	roi.phi.x = 0.0;
	roi.phi.y = roi.phi.x+phiStep;

	thrustHvec<uint>       hits(zLongNum*cryNum);
	thrustHvec<uint>       vfill(zLongNum*voxNum*voxNum);
	thrustHvec<uint>       map(mapSlice*mapSlice);
	thrustHvec<uint>       spot(spotNphi*spotNz*mapSlice);
	roi2xyz(roi); // just for printing
	cx::timer tot;
	cx::timer tim;
	double ngood  =0.0;
	for(int vx=vx1;vx<vx2;vx++){
		hostgen(map.data(),ngood,roi,seed,ngen); // NB roi is not changed
	}
	double t1 = tim.lap_ms();

	//printf("ngen %d good %.0f time %.3f ms\n",ngen,ngood,t1);
	tim.reset();
	if(savelors){
		char name[256];
		sprintf(name,"host_big_map%3.3d.raw",vx1);
		cx::write_raw(name,map.data(),mapSlice*mapSlice);
	}
	double t2 = tim.lap_ms();
	double t3 = tot.lap_ms();
	double eff = 100.0*ngood/ngen;
	printf("ngen %d good %.0f eff %.3f%%  gen time %.3f io time %.3f total %.3f ms\n",
		ngen,ngood,eff,t1,t2,t3);


	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("The first fullsim parameter is 1, 2 or 3 for operation mode\n\n");
		printf("mode 1 generate spot files for the system matrix\n");	
		printf("mode 2 genrate cylindrical phantom datasets for testing reconstuction\n");
		printf("mode 3 is a simple host verison of mode1 for timing comparisons\n");
		printf("usage: fullsim 1 threads blocks ngen seed vx1 vx2 savelors savemap\n");
		printf("or:    fullsim 2 threads blocks ngen seed <outfile name> r1 r2 phi1 phi2 z1 z2 ox oy oz savevol savelors singles\n");
		printf("or:    fullsim 3 ngen seed vx1 vx2 savelors\n");
		return 0;
	}

	FILE * flog = fopen("fullsim.log","a");
	for(int k=0;k<argc;k++) fprintf(flog," %s",argv[k]); fprintf(flog,"\n"); fclose(flog);


	int mode = (argc >1) ? atoi(argv[1]) : 1;
	if(mode==2){
		do_phantom(argc,argv);
		return 0;
	}
	else if(mode==3){
		do_hostver(argc,argv);
		return 0;
	}
	else if(mode != 1) {printf("bad mode=%d\n",mode); return 1; }

	printf("Detector: len %.1f radius %.3f rings %d crytals/ring %d zstep %.3f phistep %.3f (degrees)\n",
		detLongLen,detRadius,zLongNum,cryNum,crySize,360.0/cryNum);

	uint threads =   (argc> 2) ? atoi(argv[2]) : 256;
	uint blocks  =   (argc> 3) ? atoi(argv[3]) : 1024;
	int ndo      =   (argc> 4) ? atoi(argv[4]) : 100;
	std::random_device rd;
	long long seed = (argc > 5) ? atoi(argv[5]) : rd();
	int vx1      =   (argc > 6) ? atoi(argv[6]) : 0;      // integer voxel start r rspan setis preset as voxSize
	int vx2      =   (argc > 7) ? atoi(argv[7]) : vx1+1;
	int savelors =   (argc > 8) ? atoi(argv[8]) : 0;
	int savemap  =   (argc > 9) ? atoi(argv[9]) : 1;

	uint size = blocks*threads;
	int passes = 1;
	long long  ngen = 1000000;

	if(ndo <= 1000) ngen *= (long long)ndo;
	else {
		passes = (ndo + 999) / 1000;
		ngen *= 1000ll;  // ll long long
	}
	uint tries = (ngen+size-1)/size;
	ngen = (long long)tries*(long long)size;
	long long ngen_all = ngen*(long long)passes;


	Roi roi;
	roi.r.x =  voxSize*vx1;        // voxel r range voxSize*[vx1,vx2]
	roi.r.y =  roi.r.x+voxSize;    //         

	roi.z.x = crySize*(zNum-1);    // voxel z range crySize*[63,64]
	roi.z.y =  roi.z.x+crySize;    // 

	roi.phi.x = 0.0;               // voxel phi range phiStep*[0,1]
	roi.phi.y = roi.phi.x+phiStep; //

	roi2xyz(roi);

	// use XORWOW
	thrustDvec<curandState> state(size);  // this for curand_states

	thrustHvec<uint>       hits(zLongNum*cryNum);
	thrustDvec<uint>   dev_hits(zLongNum*cryNum);
	thrustHvec<uint>       vfill(zLongNum*voxNum*voxNum);
	thrustDvec<uint>   dev_vfill(zLongNum*voxNum*voxNum);
	//thrustHvec<ushort>     sfill(zLongNum*voxNum*voxNum);
	//thrustDvec<ushort> dev_sfill(zLongNum*voxNum*voxNum);
	thrustHvec<uint>       map(mapSlice*mapSlice);
	thrustDvec<uint>   dev_map(mapSlice*mapSlice);
	thrustHvec<uint>       spot(spotNphi*spotNz*mapSlice);
	thrustDvec<uint>   dev_spot(spotNphi*spotNz*mapSlice);

	thrustDvec<double> dev_good(size);
	cx::timer tim;
	cx::timer io;
	cx::timer tot;
	init_generator<<<blocks,threads>>>(seed,state.data().get());

	for(int vx=vx1;vx<vx2;vx++){  // loop over set of radial voxels from vx1 to vx2-1
		tim.start();
		for(int k=0;k<passes;k++){
			voxgen<<< blocks,threads >>>(dev_map.data().get(),dev_good.data().get(),roi,state.data().get(),tries);
			checkCudaErrors(cudaPeekAtLastError()); // these need to be inside loop for FERMI CUDA Driver Bug??
			checkCudaErrors(cudaDeviceSynchronize());
		}
		tim.add();
		find_spot<<<zNum,cryNum >>>(dev_map.data().get(),dev_spot.data().get(),roi);

		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		double all_good = thrust::reduce(dev_good.begin(),dev_good.end());
		double eff = 100.0*all_good/(double)ngen_all;

		io.reset();
		if(savelors){
			char name[256];
			sprintf(name,"lor_map%3.3d.raw",vx);
			map = dev_map;
			cx::write_raw(name,map.data(),mapSlice*mapSlice);
		}
		if(savemap){
			char name[256];
			sprintf(name,"spot_map%3.3d.raw",vx);
			spot = dev_spot;
			cx::write_raw(name,spot.data(),spotNphi*spotNz*mapSlice);
		}
		io.add();

		if(vx < vx2-1){   // step voxel r range and clear buffers for all but last voxel
			roi.r.x = roi.r.y;
			roi.r.y = roi.r.x+voxSize;
			clear_vec<<<blocks,threads>>>(dev_hits.data().get(),zLongNum*cryNum);
			clear_vec<<<blocks,threads>>>(dev_map.data().get(),mapSlice*mapSlice);
			clear_vec<<<blocks,threads>>>(dev_spot.data().get(),spotNphi*spotNz*mapSlice);
			clear_vec<<<blocks,threads>>>(dev_good.data().get(),size);
		}
		tot.add();
		printf("ngen %lld good %.0f eff %.3f%% tries %d passes %d gen time %.3f io time %.3f total %.3f ms\n",ngen_all,all_good,eff,tries,passes,tim.time(),io.time(),tot.time());
	}

	printf("done\n");
	return 0;
}