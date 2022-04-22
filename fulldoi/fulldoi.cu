// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 8.9 The fulldoi program
// This is the same os the fullsim program except that depth of interation is added to ray_to_cyl
// The spot sizes are bigger for oblique rays are find_spot is alos modified.
// Host versions of the code have been remved

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxconfun.h"
#include "scanner.h"
#include "helper_math.h"     // for lerp
#include "curand_kernel.h"   // for cuRand
#include <random>

// this is example 8.9
template <typename S> __device__ int ray_to_cyl_doi(Ray &g,Lor &l,float length,S &state)
{
  //swim to cyl: solve quadratic Ax^2+2Bx+C = 0
	float A = g.n.x*g.n.x + g.n.y*g.n.y;
	float B = g.a.x*g.n.x + g.a.y*g.n.y;  // factors of 2 ommited as they cancel
	float C = g.a.x*g.a.x + g.a.y*g.a.y - detRadius * detRadius;
	float D = B * B - A * C;
	float rad = sqrtf(D);

	g.lam1 = (-B + rad) / A;  // gamma1
	float path1 = -BGO_atlen * logf(1.0f - curand_uniform(&state));
	g.lam1 += (g.lam1 >= 0.0f) ? path1 : -path1;
	float x1 = g.a.x + g.lam1*g.n.x;
	float y1 = g.a.y + g.lam1*g.n.y;
	if(x1*x1+y1*y1 > doiR2) return 0;  // ray escapes 
	float z1 = g.a.z + g.lam1*g.n.z;

	g.lam2 = (-B - rad) / A;  // gamma2
	float path2 = -BGO_atlen * logf(1.0f - curand_uniform(&state));
	g.lam2 += (g.lam2 > 0.0f) ? path2 : -path2;
	float x2 = g.a.x + g.lam2*g.n.x;
	float y2 = g.a.y + g.lam2*g.n.y;
	if(x2*x2 + y2*y2 > doiR2) return 0;  // ray escapes

	float z2 = g.a.z + g.lam2*g.n.z;

	if(z1 >= 0.0f && z1 < length && z2 >= 0.0f && z2 < length && abs(z2 - z1) < detLen) { // same zdiff short and long detectors
		float phi = myatan2(x1,y1);
		l.z1 = (int)(z1*detStep);  
		l.c1 = phi2cry(phi);       
		phi = myatan2(x2,y2);
		l.z2 = (int)(z2*detStep);  
		l.c2 = phi2cry(phi);                

		if(l.z1 > l.z2) {          
			cx::swap(l.z1,l.z2);
			cx::swap(l.c1,l.c2);
		}
		return 1; // good ray
	}
	return 0;  // bad ray
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

__global__ void find_spot_doi(r_Ptr<uint> map, r_Ptr<uint> spot, Roi vox)
{
	float c1 = threadIdx.x;   // use floats for (z1,c1) as
	float z1 = blockIdx.x;    // they are used in calcuations
	if(c1 >= cryNum || z1 >= zNum) return;

	float3 a; // LH hit on scanner
	float phi = (cx::pi2<> / cryNum)*(c1 + 0.5f);
	a.x = detRadius * sinf(phi);  // phi = 0 along y axis
	a.y = detRadius * cosf(phi);  // and increases clockwise
	a.z = crySize * (z1 + 0.5f);

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
	int c2 = phi2cry(phi);  // This is prediced end point
	int z2 = c.z / crySize; // of lor with start (z1,c1)

	int zsm1 = zNum-1 - (int)z1;
	int zsm2 = z2 - zNum+1;
	zsm2 = clamp(zsm2,0,zNum-1); // spot z in valid range (does not have to be)

	size_t m_slice = (zsm1*cryNum + c1)*mapSlice;           // (z1,c1) slice in map
	size_t s_slice = (zsm1*cryNum + c1)*spotNphi*spotNz;    // (z1,c1) slice in spotmap

	// This added for doi refine position to include actual cluster
	// NB using zsm coords for both arrays
	int cl = cyc_sub(c2,spotNphi);
	int zl = max(0,zsm2-spotNphi/2);
	int cstart = 2*cryNum;
	int zstart = 2*zNum;
	int cend = -1;
	int zend = -1;
	for(int zp=0;zp<zNum;zp++) {
		int c = cl;
		for(int cp=0;cp<spotNphi*2;cp++){
			if(map[m_slice + zp*cryNum +c] > 10){
				cstart = min(cstart,c);
				cend   = max(cend,c);
				zstart = min(zstart,zp);
				zend =   max(zend,zp);
			}
			c = cyc_inc(c);
		}
	}
	if(cend > cstart+cryNum/4){
		int t = cend;
		cend = cstart;
		cstart = t;
	}
	zstart = clamp(zstart-1,0,zNum-spotNz);
	int zrange = zend-zstart+1;
	int zadj = max(0,(spotNz-zrange)/2);
	zstart  = max(0,zstart-zadj);
	int crange = cyc_sub(cend,cstart)+1;
	int cadj = max(0,(spotNphi-crange)/2);
	cstart = cyc_sub(cstart,cadj);

	// extract 24x24 "spot" from map file 	
	int sz = zstart; // z offset in map
	for(int iz = 0; iz < spotNz; iz++) {
		int sc = cstart; // c offset in map
		for(int ic = 0; ic < spotNphi; ic++) {
			uint val = map[m_slice + sz * cryNum + sc];
			spot[s_slice + iz * spotNphi + ic] = val;
			sc = cyc_inc(sc);  // might wrap to zero
		}
		sz++;
		if(sz >= zNum) break;
	}

	// store offsets in col 0 rows 0-1
	spot[s_slice] = zstart;
	spot[s_slice + spotNphi] = cstart;

}


// this is example 8.2 in text with one line changed
template <typename S> __global__ void voxgen_doi(r_Ptr<uint> map, r_Ptr<double> ngood, Roi roi, S *states, uint tries)
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
		if(ray_to_cyl_doi(g, lor, detLongLen, state)){  // example 8.9 modification
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
template <typename S> __global__ void phantom(r_Ptr<uint> map,r_Ptr<uint> vfill,r_Ptr<uint> vfill2,PRoi roi,r_Ptr<double> ngood,S *states,uint tries,int savevol)
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
		if(ray_to_cyl_doi(g,lor,detLen,state)){ // changed for DOI
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
	roi.r.x      = (argc > 7) ? atof(argv[7]) : 0.0f;
	roi.r.y      = (argc > 8) ? atof(argv[8]) : 1.0f;
	roi.phi.x    = (argc > 9) ? atof(argv[9])*cx::pi<float>/180.0f : 0.0f;
	roi.phi.y    = (argc >10) ? atof(argv[10])*cx::pi<float>/180.0f : 2.0f*cx::pi<float>;
	roi.z.x      = (argc >11) ? atof(argv[11]) : 0.0f;
	roi.z.y      = (argc >12) ? atof(argv[12]) : 1.0f;
	roi.o.x      = (argc >13) ? atof(argv[13]) : 0.0f;
	roi.o.y      = (argc >14) ? atof(argv[14]) : 0.0f;
	roi.o.z      = (argc >15) ? atof(argv[15]) : 0.0f;
	int savevol    = (argc >16) ? atof(argv[16]) : 0;
	int savelors = (argc >17) ? atoi(argv[17]) : 1;

	printf("Phantom r (%.1f %.1f) p (%.3f %.3f) z (%.1f %.1f) o (%.1f %.1f %.1f)\n",
		roi.r.x,roi.r.y,roi.phi.x,roi.phi.y,roi.z.x,roi.z.y,roi.o.x,roi.o.y,roi.o.z);

	// use XORWOW
	thrustDvec<curandState> state(size);  // this for curand_states

	uint vsize = zNum*voxNum*voxNum;
	uint msize = mapSlice*mapSlice;
	uint zsize = cryNum*cryDiffNum*detZdZNum;
	thrustHvec<uint>       vfill(vsize);
	thrustDvec<uint>   dev_vfill(vsize);
	thrustHvec<uint>       vfill2(vsize);
	thrustDvec<uint>   dev_vfill2(vsize);
	thrustHvec<uint>       map(msize);
	thrustDvec<uint>   dev_map(msize);
	thrustDvec<double> dev_good(size);
	thrustHvec<uint>       zdzmap(zsize);
	thrustDvec<uint>   dev_zdzmap(zsize);

	// if file exists we append
	if(cx::can_be_opened(argv[6])){
		if(cx::read_raw(argv[6],zdzmap.data(),zsize) != 0) for(uint k=0;k<zsize;k++) zdzmap[k] = 0;
		else dev_zdzmap = zdzmap;
	}
	if(cx::can_be_opened("phant_roi_all.raw")){
		if(cx::read_raw("phant_roi_all.raw",vfill.data(),vsize) == 0) dev_vfill = vfill;
	}
	if(cx::can_be_opened("phant_roi_good.raw")){
		if(cx::read_raw("phant_roi_good.raw",vfill.data(),vsize) == 0) dev_vfill2 = vfill;
	}

	cx::timer tim;
	init_generator<<<blocks,threads>>>(seed,state.data().get());
	for(int k=0;k<passes;k++){
		phantom<<<blocks,threads >>>(dev_zdzmap.data().get(),dev_vfill.data().get(),dev_vfill2.data().get(),
			roi,dev_good.data().get(),state.data().get(),tries,savevol);
	}
	cx::ok(cudaPeekAtLastError());
	cx::ok(cudaDeviceSynchronize());
	tim.add();
	double all_good = thrust::reduce(dev_good.begin(),dev_good.end());
	double eff = 100.0*all_good/(double)ngen_all;

	printf("Phantom ngen %lld good %.0f eff %.3f%% time %.3f ms\n",ngen_all,all_good,eff,tim.time());

	if(savevol){
		vfill = dev_vfill;
		cx::write_raw("phant_roi_all.raw",vfill.data(),zNum*voxNum*voxNum);
		vfill = dev_vfill2; // NB
		cx::write_raw("phant_roi_good.raw",vfill.data(),zNum*voxNum*voxNum);
	}
	if(savelors){
		zdzmap = dev_zdzmap;
		cx::write_raw(argv[6],zdzmap.data(),cryNum*cryDiffNum*detZdZNum); // always append
	}

	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("The first fulldoi parameter is 1, 2  for operation mode\n\n");
		printf("mode 1 generate spot files with doi for the system matrix\n");
		printf("mode 2 genrate cylindrical phantom datasets for testing reconstuction\n");
		printf("usage: fulldoi 1 threads blocks ngen seed vx1 vx2 savelors savemap\n");
		printf("or:    fulldoi 2 threads blocks ngen seed <outfile name> r-range phi-range z-range offsets savelors\n");
		return 0;
	}

	FILE * flog = fopen("fullsim.log","a");
	for(int k=0;k<argc;k++) fprintf(flog," %s",argv[k]); fprintf(flog,"\n"); fclose(flog);


	int mode = (argc >1) ? atoi(argv[1]) : 1;
	if(mode==2){
		do_phantom(argc,argv);
		return 0;
	}
	else if(mode != 1) { printf("bad mode=%d\n",mode); return 1; }

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
		ngen *= 1000ll;
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
			voxgen_doi<<< blocks,threads >>>(dev_map.data().get(), dev_good.data().get(), roi, state.data().get(), tries);
			cx::ok(cudaPeekAtLastError()); 
			cx::ok(cudaDeviceSynchronize());
		}
		tim.add();
		find_spot_doi<<<zNum,cryNum >>>(dev_map.data().get(), dev_spot.data().get(), roi);

		cx::ok(cudaPeekAtLastError());
		cx::ok(cudaDeviceSynchronize());

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