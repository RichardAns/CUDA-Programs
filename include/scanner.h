// scanner.h
#pragma once
#include "vector_types.h"  // This is OK in host code - defines float3 etc. etc.
#include "cxconfun.h"

template <typename T> __host__ __device__ T myatan2(T x,T y){
	T angle = atan2(x,y);
	if(angle <0) angle += cx::pi2<T>;
	return angle;
}

//use for real lors
struct Lor {  // Lor definition
	int z1;   // displacement from left edge of scanner
	int c1;
	int z2;   // displacement from left edge of scanner
	int c2;
};

// use for translation invarient base lors
struct smLor {  // Same as Lor except names of z variables changed
	int zsm1;   // displacement to left of decay voxel
	int c1;
	int zsm2;   // displacement of right of decay voxel
	int c2;
};

struct smPart { // 8-byte sytem matrix element Svl
	uint key;
	float val;
};

// cannot be member functions with present CUDA
#if __CUDACC__
__device__ __host__ inline smLor key2lor(uint key)  { 
	smLor sml;
	sml.c2 = key & 0x000001ff;         // bits  8-0
	sml.zsm2 = (key>>9) & 0x0000007f;  // bits 15-9
	sml.c1 = (key>>16) & 0x000001ff;   // bits 24-16
	sml.zsm1 = (key>>25);              // bits 31-25
	return sml;
}

__device__ __host__ inline uint lor2key(smLor &sml) {  // currently not used
	uint key = 0;
	key = (sml.zsm1<<25) | (sml.c1<<16) | (sml.zsm2<<9) | (sml.c2);
	return key;
}
#else
smLor key2lor(uint key)  { 
	smLor sml;
	sml.c2 = key & 0x000001ff;         // bits  8-0
	sml.zsm2 = (key>>9) & 0x0000007f;  // bits 15-9
	sml.c1 = (key>>16) & 0x000001ff;   // bits 24-16
	sml.zsm1 = (key>>25);              // bits 31-25
	return sml;
}

uint lor2key(smLor &sml) {  // currently not used
	uint key = 0;
	key = (sml.zsm1<<25) | (sml.c1<<16) | (sml.zsm2<<9) | (sml.c2);
	return key;
}
#endif
 
struct sm_full_part {
	uint key;
	uint key2;
	float val;

};

struct smTab {  // to complement monolithic sm table
	int ring;
	uint start;                                            // was sm_start
	uint end;   // like iterator this is one past the end  // was sm_end
	int  phi_steps;
};


// pack lor into 32-bit uint key, 
// | zsm1 | C1 | zsm2 | c2 |
// |  7   | 9  |  7   |  9 |
class Hit {
private:
	uint key;
	float v;
public:
	inline  void key_from(Lor &l) {  // currently not used
		key = 0;
		key = (l.z1<<25) | (l.c1<<16) | (l.z2<<9) | (l.c2);
	}
	inline Lor key_to(void) const {  //used by readspot
		Lor l;
		l.c2 = key & 0x000001ff;
		l.z2 = (key>>9) & 0x0000007f;
		l.c1 = (key>>16) & 0x000001ff;
		l.z1 = (key>>25);
		return l;
	}
	inline  uint getkey(void) const { return key;}
	inline  float   val(void) const { return v;}
	Hit(){ key = 0; v = 0.0f; }  // necessary for std::vector allocation
	Hit(Lor &l,float val){
		key_from(l);
		v = val;
	}
};

#if  __CUDACC__
// used by reco_kernels.h
__device__ void lor_from_key(uint &key, Lor &l)
{
	l.c2 = key & 0x000001ff;
	l.z2 = (key>>9) & 0x0000007f;
	l.c1 = (key>>16) & 0x000001ff;
	l.z1 = (key>>25);
}

// currently not used
__device__ void key_from_lor(uint &key, Lor &l)
{
	key = 0;
	key = (l.z1<<25) | (l.c1<<16) | (l.z2<<9) | (l.c2);
}
#endif

struct Ray {
	float3 a;   // point on ray
	float3 n;   // direction vector
	float lam1; // user point 1 on ray
	float lam2; // user point 2 on ray
};

struct Roi {
	float2 z;
	float2 r;
	float2 phi;
};

struct PRoi {   // Like Roi but adds origin, for phantom use
	float2 z;   // z range
	float2 r;   // radial range
	float2 phi; // phi range
	float3 o;   // orign
};

// TODO clarify detector and scanner parameters scnRadius nor detRadius etc
// Parameterise Scanner NB lengths in mm
// Basic Parameters
constexpr int    cryNum     = 400;   // number of crystals in one ring
constexpr float  crySize    = 4.0f;  // size of square face 4 x 4 mm
constexpr float  cryDepth   = 20.0f; // depth of crystal 20 mm
constexpr int    zNum = 64;          // Number of rings in detector
constexpr float  LSO_atlen  = 11.4f;  // mean path in mm
constexpr float  BGO_atlen  = 10.4f;  // mean path in mm
// Derived Parameters
constexpr float detRadius = cryNum*crySize/cx::pi2<>; // inner circumference / 2pi
constexpr float detLen    = crySize*zNum;
constexpr float doiR2     = (detRadius + cryDepth)*(detRadius + cryDepth); // square of outer radius

// These for Block Detectors
constexpr int   bcNum =  8;
constexpr int   bzNum =  8;
constexpr float bcSize = bcNum*crySize;  // square blocks 32 x 32 mm
constexpr float bcHalf = bcSize*0.5f;
constexpr float bzSize = bzNum*crySize;  
constexpr int   bNum =   cryNum/bcNum;   // total block 400/8 = 50
constexpr float bGap =   0.1f;           // extend base by 0.1 mm on both sides in transverse plane
constexpr float bRadius = (bcSize/2+bGap)/cx::tan_cx(cx::pi<>/bNum);  // perp dist to face center
constexpr float bStep =  (float)bNum / cx::pi2<>; // map phi to block
constexpr float rbStep = (float)cx::pi2<> / bNum; // map block to phi

// These for debug plots in fullblk
constexpr int vSize = 600*4;
constexpr int aSize = 720;

// Long detector used in SM simulation
constexpr int   zLongNum   = 2*zNum-1;         // double length simulation detector
constexpr float detLongLen = crySize*zLongNum; // long axial length (max allowed z in mm)

// Limits used in reconstuction
constexpr int      zDiffMax = zNum-1;
constexpr int    cryDiffMin = cryNum/4;            // min tranverse length of lor
constexpr int    cryDiffMax = cryNum-cryDiffMin;   // max transverse length of lor
constexpr int    cryDiffNum = (cryDiffMax-cryDiffMin+1);

// for polar voxel coordinate grid
constexpr float  roiRadius = 200.0f;               // max radius for roi
constexpr int    radNum  = 100;                    // voxel circles
constexpr float  radStep = roiRadius/radNum;       // radial voxel size (2 mm)

// Max cos(theta) for efficiency cut in MC
//constexpr float minChord = 2.0f*cx::sqrt_cx(bRadius*bRadius - roiRadius * roiRadius);
//constexpr float maxCosTheta = detLen / cx::sqrt_cx(minChord*minChord+detLen*detLen);

// for SM constuction
constexpr int    spotNphi = 24;                  // sinogram spot max phi size
constexpr int    spotNz = 24;                    // sinogram spot max z size
constexpr int    spotSlice   = spotNz*spotNphi;   // slice size in spot maps
constexpr float  sysMatScale = 1.0e+11f;

// for polar to cartesian voxel mapping.
constexpr int    voxBox   = 5;                   // nxn polar to cartesian box
constexpr int    voxBoxOffset = voxBox/2;        //  box offset
constexpr int    voxNum  =  radNum*2;              // voxel transverse number
constexpr float  voxSize  = 2.0f*roiRadius/voxNum; // voxel transverse size
constexpr float  voxStep  = 1.0f/voxSize;          // map tranverse distance to voxel

// convert detector element to/from angle 
constexpr float  cryStep    = (float)cryNum/cx::pi2<>; // map phi to crystal
constexpr float  phiStep    = cx::pi2<>/(float)cryNum; // map crystal to phi 

// convert axial distante to rdetector ring number
constexpr float  detStep =   1.0f/crySize;             // map z to axial-ring number

// for indexing
constexpr int    mapSlice   = cryNum*zNum;           // for sinogram maps
constexpr int    detZdZNum = zNum*(zNum+1)/2;        // Max z1 & dz combinations
constexpr int    cdcNum = cryDiffMax - cryDiffMin+1; // number of dc = c2-c1  combinations
constexpr int    cryCdCNum = cryNum*cdcNum;          //  c1*dc combinations

// convert between angular positions and crystals
//#if  __CUDACC__
__host__ __device__ int phi2cry(float phi)
{
	while(phi < 0.0f) phi += cx::pi2<float>;
	while(phi >= cx::pi2<float>) phi -= cx::pi2<float>;
	return (int)( phi*cryStep );
}

__host__ __device__ float cry2phi(int cry)  // not used
{
	while(cry < 0) cry += cryNum;
	while(cry >= cryNum) cry -= cryNum;
	return ( (float)cry+0.5f )*phiStep;  // phi at crystal centre
}
//#else
//int phi2cry(float phi)
//{
//	while(phi < 0.0f) phi += cx::pi2<float>;
//	while(phi >= cx::pi2<float>) phi -= cx::pi2<float>;
//	return (int)( phi*cryStep );
//}

//float cry2phi(int cry)  // not used
//{
//	while(cry < 0) cry += cryNum;
//	while(cry >= cryNum) cry -= cryNum;
//	return ( (float)cry+0.5f )*phiStep;  // phi at crystal centre
//}
//#endif

// indexing modulo cryNum = 400, assumes i is within interval
#if  __CUDACC__
__host__ __device__ int cyc_sub(cint i,cint step) { return i >= step ? i-step : i-step+cryNum; }
__host__ __device__ int cyc_dec(cint i) { return i >= 1 ? i-1 : cryNum-1; }  // not used
__host__ __device__ int cyc_add(cint i,cint step) { return i+step < cryNum  ? i+step : i+step-cryNum; }
__host__ __device__ int cyc_inc(cint i) { return i+1 < cryNum  ? i+1 : 0; }
#else
int cyc_sub(cint i,cint step) { return i >= step ? i-step : i-step+cryNum; }
int cyc_dec(cint i) { return i >= 1 ? i-1 : cryNum-1; }  // not used
int cyc_add(cint i,cint step) { return i+step < cryNum  ? i+step : i+step-cryNum; }
int cyc_inc(cint i) { return i+1 < cryNum  ? i+1 : 0; }
#endif
// end