// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// fullblk program Chaper 8 examples 8.10 - 8.13

// Note a blocked detector breaks the 400-fold phi symmetry of the previous PET examples.
// There remains a 50-fold symmetry at the block level with additional mirror symmetry
// w.r.t the centres of blocks.Thus,the system matrix is potentially increased in size
// by a factor of 8 (or 4 if we use the mirror symmetry).
// 
// In practice this means generating a separate system matrix for each different voxel
// symmetry (4 or 8) in this case.This:
// 
// 	(1) Increases the fullsim generation time proportionally.
// 		Not really an issue as only done once.
// 	(2) Increases the memory requirements of the system matrix proportionally.
// 		This IS a potential problem for the memory bound reco program.
// 	(3) Using mirror symmetry halves the memory requirement but adds complexity to the code.
// 
// 	Code for blocked reconstuction is being developed and will be added to the repository when ready.
//
// Richard Ansorge June 2021
//
#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxconfun.h"
#include "scanner.h"
#include "helper_math.h"     // for lerp
#include "curand_kernel.h"   // for cuRand
#include <random>

__device__ Ray rot_ray(Ray &g,float phi)
{
	float c = cos(phi);
	float s = sin(phi);
	Ray rg;
	//__sincosf(phi,&s,&c);
	rg.a.x = g.a.x*c - g.a.y*s;
	rg.a.y = g.a.x*s + g.a.y*c;
	rg.a.z = g.a.z;

	rg.n.x = g.n.x*c - g.n.y*s;
	rg.n.y = g.n.x*s + g.n.y*c;
	rg.n.z = g.n.z;

	return rg;
}

// this is example 8.11 ray_to_block2
__device__ int ray_to_block2(Ray &g,const float3 &p,const float3 &q)
{
	// get the 6 points were ray meets planes that define the block
	float3 lam_p = (p - g.a) / g.n; // NB all operations on
	float3 lam_q = (q - g.a) / g.n; // corresponding components

	float lmin = 1.0e+06f;   // this is lam-in
	float lmax = -1.0e+06f;  // this is lam out
	int exit_plane = 0;      // return code, zero means missed

	// lambda function (C++11)
	auto lam_check = [&lmin,&lmax,&exit_plane,g,p,q](float3 &lam,int side)
	{
		float3 b = g.a + lam.x*g.n;  // y-z left side plane
		if(b.y >= p.y && b.y <= q.y && b.z >= p.z && b.z <= q.z) {
			lmin = fminf(lmin,lam.x);
			if(lam.x > lmax) { exit_plane = 1 + side; lmax = lam.x; }
		}
		b = g.a + lam.y * g.n;  // x-z  bottom plane
		if(b.x >= p.x && b.x <= q.x && b.z >= p.z && b.z <= q.z) {
			lmin = fminf(lmin,lam.y);
			if(lam.y > lmax) { exit_plane = 3 + side; lmax = lam.y; }
		}
		b = g.a + lam.z * g.n;  // x-y front plane
		if(b.x >= p.x && b.x <= q.x && b.y >= p.y && b.y <= q.y) {
			lmin = fminf(lmin,lam.z);
			if(lam.z > lmax) { exit_plane = 5 + side; lmax = lam.z; }
		}
	};

	lam_check(lam_p,0); // Faces 1, 3 and 5
	lam_check(lam_q,1); // Faces 2, 4 and 6

	g.lam1 = lmin;  // set hit points
	g.lam2 = lmax;  // for caller

	return exit_plane;  // zero is failure
}

// this is ray_to_block example 8.10
__device__ int ray_to_block_book(Ray &g,const float3 &p,const float3 &q)
{
	// get 6 points were ray meets planes that define the block
	float3 lam_p = (p-g.a)/g.n; // NB all operations on
	float3 lam_q = (q-g.a)/g.n; // corresponding components

	float lmin =  1.0e+06f;  // this is lam-in
	float lmax = -1.0e+06f;  // this is lam out
	int exit_plane = 0;   // return code, zero means missed

	float3 b = g.a+lam_p.x*g.n;  // y-z left side plane
	if(b.y >= p.y && b.y <= q.y && b.z >= p.z && b.z <= q.z) {
		lmin = fminf(lmin,lam_p.x);
		if(lam_p.x > lmax) { exit_plane = 1; lmax = lam_p.x; }
	}
	b = g.a + lam_p.y * g.n;  // x-z  bottom plane
	if(b.x >= p.x && b.x <= q.x && b.z >= p.z && b.z <= q.z) {
		lmin = fminf(lmin,lam_p.y);
		if(lam_p.y > lmax) { exit_plane = 3; lmax = lam_p.y; }
	}
	b = g.a + lam_p.z * g.n;  // x-y front plane
	if(b.x >= p.x && b.x <= q.x && b.y >= p.y && b.y <= q.y) {
		lmin = fminf(lmin,lam_p.z);
		if(lam_p.z > lmax) { exit_plane = 5; lmax = lam_p.z; }
	}
	b = g.a + lam_q.x * g.n;  // y-z right side plane
	if(b.y >= p.y && b.y <= q.y && b.z >= p.z && b.z <= q.z) {
		lmin = fminf(lmin,lam_q.x);
		if(lam_q.x > lmax) { exit_plane = 2; lmax = lam_q.x; }
	}
	b = g.a + lam_q.y * g.n;  // x-z top plane
	if(b.x >= p.x && b.x <= q.x && b.z >= p.z && b.z <= q.z) {
		lmin = fminf(lmin,lam_q.y);
		if(lam_q.y > lmax) { exit_plane = 4; lmax = lam_q.y; }
	}
	b = g.a + lam_q.z * g.n;  // x-y back plane
	if(b.x >= p.x && b.x <= q.x && b.y >= p.y && b.y <= q.y) {
		lmin = fminf(lmin,lam_q.z);
		if(lam_q.z > lmax) { exit_plane = 6; lmax = lam_q.z; }
	}

	g.lam1 = lmin;  // set hit points
	g.lam2 = lmax;  // for caller

	return exit_plane;  // zero is failure
}

// this is track_ray example 8.12
__device__ int track_ray_book(Ray &g,const float length,float path)
{
	//swim to cyl through block face centres: solve quadratic Ax^2+2Bx+C = 0
	float A = g.n.x*g.n.x + g.n.y*g.n.y;
	float B = g.a.x*g.n.x + g.a.y*g.n.y;  // factors of 2 ommited as they cancel
	float C = g.a.x*g.a.x + g.a.y*g.a.y - bRadius * bRadius;
	float D = B * B - A * C;
	float rad = sqrtf(D);

	float lam_base = (-B + rad) / A;  // this is +ve root
	float3 b = g.a + lam_base * g.n;  // b on cylinder touching inner surfce of blocks
	float phi = myatan2(b.x,b.y);
	int block = (int)(phi*bStep);
	block = block % bNum;            // nearest block to b 

	// rotated ray rg meets AABB aligned block
	float rotback = rbStep * ((float)block + 0.5f);
	Ray rg = rot_ray(g,rotback);
	float3 p{-bcHalf,bRadius,0.0f};               // block p corner
	float3 q{bcHalf,bRadius + cryDepth,length};  // block q corner

	// find rg intersection with block
	int exit_plane = ray_to_block2(rg,p,q);
	if(exit_plane > 0 && rg.lam2-rg.lam1 >= path){  //ray detected
		g.lam1 = rg.lam1;
		g.lam2 = rg.lam1+path; // interaction point defines LOR
		return exit_plane;
	}
	if(exit_plane > 2) return 0;  // gamma not detected

	// try adjcent block if not absorbed 
	if(exit_plane > 0)  path -= rg.lam2-rg.lam1;
	if(rg.n.x < 0.0f)  rotback -= rbStep;  // enter on RH side of block
	else                rotback += rbStep;  // enter on LH side of block
	rg = rot_ray(g,rotback);
	exit_plane = ray_to_block2(rg,p,q);
	if(exit_plane > 0 && rg.lam2 - rg.lam1 >= path) {
		g.lam1 = rg.lam1;
		g.lam2 = rg.lam1+path; // interaction point defines LOR
		return exit_plane;
	}
	return 0;  // not detected
}

//  this is example 8.13 voxgen_block
template <typename S> __global__ void voxgen_block(r_Ptr<uint> map,r_Ptr<double> ngood,Roi roi,S *states,uint tries,r_Ptr<uint> view,r_Ptr<uint> angle)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];
	Ray g1;  // change now use seperate rays for
	Ray g2;  // the two gammas in block version
	Lor lor;
	uint good = 0;
	float r1sq = roi.r.x*roi.r.x;         // r1^2
	float r2sq = roi.r.y*roi.r.y - r1sq;  // r2^2 - r1^2
	float dphi = roi.phi.y - roi.phi.x;   // phi range
	float dz = roi.z.y - roi.z.x;         // z range

	for(uint k = 0; k < tries; k++) {
		// generate point in roi
		float phi = roi.phi.x + dphi * curand_uniform(&state);  // uniform in phi
		float r = sqrtf(r1sq + r2sq * curand_uniform(&state)); // uniforam in annulus		
		g1.a.x = r * sinf(phi);
		g1.a.y = r * cosf(phi);
		g1.a.z = roi.z.x + dz * curand_uniform(&state); // uniform in z

		// generate isotropic back to back gammas
		phi = cx::pi2<float>*curand_uniform(&state);           // uniform phi & use acos for theta
		float theta = acosf(1.0f - 2.0f*curand_uniform(&state)); // for 3D isotropic directions
		g1.n.x = sinf(phi)*sinf(theta);
		g1.n.y = cosf(phi)*sinf(theta);
		g1.n.z = cosf(theta);

		g2.a =  g1.a;
		g2.n = -g1.n; // direction vectors now in opposite directions

		// find & save hits in scanner detectors
		float path1 = -BGO_atlen * logf(1.0f - curand_uniform(&state));
		float path2 = -BGO_atlen * logf(1.0f - curand_uniform(&state));
		int ex1 = track_ray_book(g1,detLongLen,path1);
		int ex2 = track_ray_book(g2,detLongLen,path2);
		if(ex1 > 0 && ex2 > 0) {   // was single &
			float3 p1 = g1.a + g1.lam2*g1.n;  //(g1.lam1 + path1)*g1.n; 
			float phi = myatan2(p1.x,p1.y);
			lor.z1 = (int)(p1.z*detStep);
			lor.c1 = phi2cry(phi);

			float3 p2 = g2.a + g2.lam2*g2.n;  //(g2.lam1 + path2)*g2.n; 
			phi = myatan2(p2.x,p2.y);
			lor.z2 = (int)(p2.z*detStep);
			lor.c2 = phi2cry(phi);
			if(abs(lor.z2 - lor.z1) < zNum) {
				if(lor.z1 > lor.z2) {          // here z1 & z2 measured from LH end of long scanner
					cx::swap(lor.z1,lor.z2);
					cx::swap(lor.c1,lor.c2);
				}
				uint zsm2 = max(0,lor.z2 - zNum + 1);
				uint zsm1 = max(0,zNum - lor.z1 - 1);
				uint index = (zsm1*cryNum + lor.c1)*mapSlice + (zsm2*cryNum + lor.c2);
				atomicAdd(&map[index],1);
				good++;
			}
		}
	}
	ngood[id] += good;
	states[id] = state;
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
	//printf("Roi  phi (%7.3f %7.3f) r (%7.1f %7.1f) z (%7.1f %7.1f) a (%8.2f %8.2f %8.2f)\n",v.phi.x,v.phi.y,v.r.x,v.r.y,v.z.x,v.z.y,a.x,a.y,a.z);
	return a;
}

__global__ void find_spot_book2_doi(r_Ptr<uint> map,r_Ptr<uint> spot,Roi vox)
{
	float c1 = threadIdx.x;   // use floats for (z1,c1) as
	float z1 = blockIdx.x;    // they are used in calcuations
	if(c1 >= cryNum || z1 >= zNum) return;
	int zsm1 = zNum - 1 - (int)z1;
	size_t m_slice = (zsm1*cryNum + c1)*mapSlice;           // (z1,c1) slice in map
	size_t s_slice = (zsm1*cryNum + c1)*spotNphi*spotNz;    // (z1,c1) slice in spotmap

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
	float lam2 = -2.0f*(a.x*b.x + a.y*b.y - detRadius * detRadius) / ((b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y));
	float3 c; // centre of (z2,c2) cluster for (z1,c1) lor family
	c = lerp(a,b,lam2);
	phi = myatan2(c.x,c.y);
	int c2 = phi2cry(phi);  // This is prediced end point
	int z2 = c.z / crySize; // of lor with start (z1,c1)
	int zsm2 = z2 - zNum + 1;
	zsm2 = clamp(zsm2,0,zNum - 1); // spot z in valid range (does not have to be)


	// added for doi  refine position to include actual cluster
	// NB using zsm coords for both arrays
	int cl = cyc_sub(c2,spotNphi);
	int zl = max(0,zsm2 - spotNphi / 2);
	int cstart = 2 * cryNum;
	int zstart = 2 * zNum;
	int cend = -1;
	int zend = -1;
	for(int zp = 0; zp < zNum; zp++) {
		int c = cl;
		for(int cp = 0; cp < spotNphi * 2; cp++) {
			if(map[m_slice + zp * cryNum + c] > 10) {
				cstart = min(cstart,c);
				cend = max(cend,c);
				zstart = min(zstart,zp);
				zend = max(zend,zp);
			}
			c = cyc_inc(c);
		}
	}
	if(cend > cstart + cryNum / 4) {
		int t = cend;
		cend = cstart;
		cstart = t;
	}
	zstart = clamp(zstart - 1,0,zNum - spotNz);
	int zrange = zend - zstart + 1;
	int zadj = max(0,(spotNz - zrange) / 2);
	zstart = max(0,zstart - zadj);
	int crange = cyc_sub(cend,cstart) + 1;
	int cadj = max(0,(spotNphi - crange) / 2);
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

//NB this can be called with either z1 or (z2-z1) as argument
//   steps in the other variable will then be adjacent in memory
//   Using (z2-z1) are argument turns out to be a bit faster.
__device__ int zdz_slice(int z)
{
	return detZdZNum - (zNum-z)*(zNum-z+1)/2;
}

template <typename S> __global__ void phantom_blk(r_Ptr<uint> map,r_Ptr<uint> vfill,r_Ptr<uint> vfill2,PRoi roi,r_Ptr<double> ngood,S *states,uint tries,int dovol)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];
	Ray g1;  // change now use seperate rays for
	Ray g2;  // the two gammas in block version
	Lor lor;
	uint good = 0;
	float r1sq = roi.r.x*roi.r.x;      // r1^2
	float r2sq = roi.r.y*roi.r.y - r1sq; // r2^2 - r1^2
	float dphi = roi.phi.y - roi.phi.x;  // phi range
	float dz = roi.z.y - roi.z.x;      // z range
	int3 p;
	int ir;
	int iz;
	int iphi;
	for(uint k = 0; k < tries; k++) {
		// generate point in roi
		float phi = roi.phi.x + dphi * curand_uniform(&state);  // uniform in phi
		float r = sqrtf(r1sq + r2sq * curand_uniform(&state)); // uniforam in annulus		
		g1.a.x = r * sinf(phi) + roi.o.x;
		g1.a.y = r * cosf(phi) + roi.o.y;
		g1.a.z = roi.z.x + roi.o.z + dz * curand_uniform(&state); // uniform in z
		if(dovol) {  // save generated decay points debug only
			p.x = (int)((g1.a.x + roiRadius)*voxStep);
			p.y = (int)((g1.a.y + roiRadius)*voxStep);
			p.z = (int)(g1.a.z*detStep);
			p.z = min(zNum - 1,p.z);
			//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
			if(p.x < 0 || p.x >= voxNum || p.y < 0 || p.y >= voxNum || p.z < 0 || p.z >= zNum) printf("bad roi %d %d %d\n",p.x,p.y,p.z);
			else {
				//atomicAdd(&vfill[(p.z*voxNum+p.y)*voxNum+p.x],1);  //this for cartesian
				ir = (int)(voxStep*sqrtf(g1.a.x*g1.a.x + g1.a.y*g1.a.y));
				iz = (int)(detStep*g1.a.z);
				float phi2 = myatan2(g1.a.x,g1.a.y);
				iphi = (int)((cryNum*phi2) / cx::pi2<>);
				atomicAdd(&vfill[(ir*zNum + iz)*cryNum + iphi],1);  // this for polar
			}
		}


		// generate isotropic back to back gammas
		phi = cx::pi2<float>*curand_uniform(&state);           // uniform phi & use acos for theta
		float theta = acosf(1.0f - 2.0f*curand_uniform(&state)); // for 3D isotropic directions
		//float theta = acosf(maxCosTheta*(1.0f - 2.0f*curand_uniform(&state))); // for 3D isotropic directions 
		g1.n.x = sinf(phi)*sinf(theta);
		g1.n.y = cosf(phi)*sinf(theta);
		g1.n.z = cosf(theta);

		g2.a = g1.a;
		g2.n = -g1.n; // direction vectors now in opposite directions

		// find & save hits in scanner detectors
		float path1 = -BGO_atlen * logf(1.0f - curand_uniform(&state));
		float path2 = -BGO_atlen * logf(1.0f - curand_uniform(&state));
		int ex1 = track_ray_book(g1,detLen,path1);  // use short detector of phantoms
		int ex2 = track_ray_book(g2,detLen,path2);
		if(ex1 > 0 & ex2 > 0) {
			float3 p1 = g1.a + g1.lam2*g1.n;  //(g1.lam1 + path1)*g1.n; 
			float phi = myatan2(p1.x,p1.y);
			lor.z1 = (int)(p1.z*detStep);
			lor.c1 = phi2cry(phi);

			float3 p2 = g2.a + g2.lam2*g2.n;  //(g2.lam1 + path2)*g2.n; 
			phi = myatan2(p2.x,p2.y);
			lor.z2 = (int)(p2.z*detStep);
			lor.c2 = phi2cry(phi);

			if(lor.z1 > lor.z2) {          // here z1 & z2 measured from LH end of scanner
				cx::swap(lor.z1,lor.z2);   // must have z1 <= z2 for output packing
				cx::swap(lor.c1,lor.c2);
			}

			int cdiff = abs(lor.c1 - lor.c2);
			if(cdiff >= cryDiffMin && cdiff <= cryDiffMax) {
				good++;
				if(dovol) {
					//atomicAdd(&vfill2[(p.z*voxNum+p.y)*voxNum+p.x],1); 
					atomicAdd(&vfill2[(ir*zNum + iz)*cryNum + iphi],1);  // this for polar volume
				}

				int zdz = zdz_slice(lor.z2 - lor.z1) + lor.z1;     // zdz_slice(dz) + zs1 in reco (from July 6th)
				zdz = min(zdz,detZdZNum - 1);                     // kill occasional dz = 64
				int cdc = cyc_sub(lor.c2,lor.c1) - cryDiffMin;
				int index = (zdz*cdcNum + cdc)*cryNum + lor.c1;
				if(zdz < 0 || zdz >= detZdZNum || cdc < 0 || cdc >= cdcNum || lor.c1 < 0 || lor.c1 >= cryNum || index < 0 || index >= detZdZNum*cdcNum*cryNum) {
					printf("bad lor zdz %d cdc %d c1 %d index %d\n",zdz,cdc,lor.c1,index);
				}
				else atomicAdd(&map[index],1);

				//atomicAdd(&map[index], 1);
			}
		}
	}
	ngood[id] += good;
	states[id] = state;
	//if( id < 100) printf("ngood[%d] = %.0f",id,ngood[id]);
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
	roi.r.x =  atof(argv[7]);
	roi.r.y =  atof(argv[8]);
	roi.phi.x = atof(argv[9])*cx::pi<float>/180.0f;
	roi.phi.y = atof(argv[10])*cx::pi<float>/180.0f;
	roi.z.x = atof(argv[11]);
	roi.z.y = atof(argv[12]);


	roi.o.x = 0.0f;  if(argc >13) roi.o.x = atof(argv[13]);
	roi.o.y = 0.0f;  if(argc >14) roi.o.y = atof(argv[14]);
	roi.o.z = 0.0f;  if(argc >15) roi.o.z = atof(argv[15]);

	int dovol = 0;   if(argc >16) dovol= atof(argv[16]);
	int savemap = 1; if(argc >17) savemap = atof(argv[17]);

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
		phantom_blk<<<blocks,threads >>> (dev_zdzmap.data().get(),dev_vfill.data().get(),dev_vfill2.data().get(),roi,dev_good.data().get(),state.data().get(),tries,dovol);
	}
	cx::ok(cudaPeekAtLastError());
	cx::ok(cudaDeviceSynchronize());
	tim.add();
	double all_good = thrust::reduce(dev_good.begin(),dev_good.end());
	double eff = 100.0*all_good/(double)ngen_all;

	printf("Phantom ngen %lld good %.0f eff %.3f%% time %.3f ms\n",ngen_all,all_good,eff,tim.time());

	if(dovol){
		vfill = dev_vfill;
		cx::write_raw("phant_roi_all.raw",vfill.data(),zNum*voxNum*voxNum);
		vfill = dev_vfill2; // NB
		cx::write_raw("phant_roi_good.raw",vfill.data(),zNum*voxNum*voxNum);
	}
	if(savemap){
		//map = dev_map;
		//cx::write_raw("phant_map_int.raw",map.data(),mapSlice*mapSlice);
		zdzmap = dev_zdzmap;
		cx::write_raw(argv[6],zdzmap.data(),cryNum*cryDiffNum*detZdZNum);
	}

	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage: fullsim <mode> <threads> <blocks> <ngen> <seed> ....\n");
		printf(" mode = 1 <r1> <r2> [savelors|1 savebigmap|0] => generate smat files radius from <vr1>  to <vr2>-1 \n");
		printf(" mode = 2 <outfile append or create> <r1> <r2> <ph1> <ph2> <z1> z2> <x> <y> <z> <dovol|0> <savemap|1> => generate cylindrical phantom withe specified origin\n");
		return 0;
	}

	FILE * flog = fopen("fullsim.log","a");
	for(int k=0;k<argc;k++) fprintf(flog," %s",argv[k]); fprintf(flog,"\n"); fclose(flog);

	int argshift = 1;
	int mode = 1; mode= atoi(argv[1]);
	if(mode==2){
		do_phantom(argc,argv);
		return 0;
	}
	if(mode != 1) argshift = 0; // retro fit mode to old code 

	printf("Detector has %d blocks: len %.1f radius %.3f rings %d crytals/ring %d zstep %.3f phistep %.3f (degrees)\n",
		bNum,detLongLen,bRadius,zLongNum,cryNum,crySize,360.0/cryNum);

	uint threads = 256; if(argc>1) threads = atoi(argv[1+argshift]);
	uint blocks = 1024; if(argc>2) blocks  = atoi(argv[2+argshift]);

	uint size = blocks*threads;
	int passes = 1;
	long long  ngen = 1000000;
	int ndo = atoi(argv[3+argshift]);

	if(ndo <= 1000) ngen *= (long long)ndo;
	else {
		passes = (ndo + 999) / 1000;
		ngen *= 1000ll;
	}
	uint tries = (ngen+size-1)/size;
	ngen = (long long)tries*(long long)size;
	long long ngen_all = ngen*(long long)passes;

	std::random_device rd;
	long long seed = rd(); if(argc > 4) seed = atoi(argv[4+argshift]);

	int vx1 = 0;      if(argc > 5+argshift) vx1 = atoi(argv[5+argshift]);
	int vx2 = vx1+1;  if(argc > 6+argshift) vx2 = atoi(argv[6+argshift]);
	int savelors = 1; if(argc > 7+argshift) savelors = atoi(argv[7+argshift]);
	int savemap = 0;  if(argc > 8+argshift) savemap = atoi(argv[8+argshift]);
	Roi roi;
	roi.r.x =  voxSize*vx1;
	roi.r.y =  roi.r.x+voxSize;
	roi.z.x = crySize*(zNum-1);    // calib voxel spans z in range 63-64;
	roi.z.y =  roi.z.x+crySize;    // z span one detector ring;
	roi.phi.x = 0.0;
	roi.phi.y = roi.phi.x+phiStep;


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

	thrustDvec<uint>   dev_view(vSize*vSize);
	thrustHvec<uint>       view(vSize*vSize);
	thrustDvec<uint>   dev_angle(aSize*aSize);
	thrustHvec<uint>       angle(aSize*aSize);


	thrustDvec<double> dev_good(size);
	cx::timer tim;
	cx::timer io;
	cx::timer tot;
	init_generator<<<blocks,threads>>>(seed,state.data().get());

	//sadly cos(theta cut) does not save ant time
	// Max cos(theta) for efficiency cut in MC
	//float minChord = 2.0f*sqrt(bRadius*bRadius - roiRadius * roiRadius);
	//float maxCosTheta = detLen / sqrt(minChord*minChord + detLen * detLen);
	float maxCosTheta = 1.0f;
	printf("Using maxCosTheta cut at %.3f\n",maxCosTheta);

	for(int vx=vx1;vx<vx2;vx++){
		int vx_offset = vx/radNum;
		int vx_rad = vx%radNum;
		roi.r.x = voxSize * vx_rad;  // set this inside loop now
		roi.r.y = roi.r.x + voxSize;
		roi.phi.x = phiStep*vx_offset;  // offset phi 1-3 steps for vx2 = 400
		roi.phi.y = roi.phi.x + phiStep;
		if(vx==vx1)roi2xyz(roi);  // print roi
		for(int k = 0; k < passes; k++) {
			voxgen_block<<< blocks,threads >>>(dev_map.data().get(),dev_good.data().get(),roi,state.data().get(),tries,dev_view.data().get(),dev_angle.data().get());
			cx::ok(cudaPeekAtLastError()); // these need to be inside loop for FERMI CUDA Driver Bug??
			cx::ok(cudaDeviceSynchronize());
		}
		tim.add();
		find_spot_book2_doi<<<zNum,cryNum >>>(dev_map.data().get(),dev_spot.data().get(),roi);

		cx::ok(cudaPeekAtLastError());
		cx::ok(cudaDeviceSynchronize());

		double all_good = thrust::reduce(dev_good.begin(),dev_good.end());

		double eff = 100.0*all_good/(double)ngen_all;

		// these for debug
		//hits = dev_hits;
		//cx::write_raw("fsim_hits_int.raw",hits.data(),zLongNum*cryNum);

		//vfill = dev_vfill;	
		//cx::write_raw("fsim_roi_int.raw",vfill.data(),zLongNum*voxNum*voxNum);

		////sfill = dev_sfill;
		////cx::write_raw("fsim_roi_short.raw",sfill.data(),zLongNum*voxNum*voxNum);
		io.reset();
		if(savemap){
			char name[256];
			sprintf(name,"big_map_blk%3.3d.raw",vx);
			map = dev_map;
			cx::write_raw(name,map.data(),mapSlice*mapSlice);
		}
		if(savelors){
			char name[256];
			sprintf(name,"spot_map_blk%3.3d.raw",vx);
			spot = dev_spot;
			cx::write_raw(name,spot.data(),spotNphi*spotNz*mapSlice);
			//view = dev_view;  // add view save here
			//sprintf(name, "view_%3.3d.raw", vx);
			//cx::write_raw(name, view.data(), vSize*vSize);
			//angle = dev_angle;  // add view save here
			//sprintf(name, "angle_%3.3d.raw", vx);
			//cx::write_raw(name, angle.data(), aSize*aSize);
		}
		io.add();

		if(vx < vx2-1){
			//roi.r.x = roi.r.y;  // set at start of loop now
			//roi.r.y = roi.r.x+voxSize;
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