// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 5.7 affine3D
// This extended code contains several kernel and host versions. The type user arguemnt 
// controls which kernel or host versions runs as follows:
//
// Type 1: Run affine3D using texture lookup with 2D thread blocks and looping over z
//         This version give best performance of ~2.9 ms per 256^3 volume.
// Type 2: Run affine3D_B using interpe3D function instead of texture lookup.
//         This version takes ~4.4 ms per 256^3 volume.
// Type 3: Run affine3D_C, similar to type 1 but using 1 thread per voxel.
//         This version takes ~4.3 ms per 256^3 volume.
// Type 4: Run affine3D_D a dummy version of addine3D without texture lookup. 
//         Non interpolation time measured as ~1.1ms per 256^3 volume.
// Type 5: Host version of affine3D using interp3D function.
//         Host takes about 5.6 seconds per 256^3 volume.
// Type 6: Host dummy version without call to interp3D, takes ~1.1 seconds per 256^3 volume
//
// Above times measured on an RTX 2070 GPU with a 32 x 8 x 1 thread block size.
//  
// RTX 2070
// C:\bin\affine3D.exe data\register\vol2.raw data\cav1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 1000 1 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) // iter 1000 type 1 outflag 0
// file data\register\vol2.raw read
// iter 1000 type 1 rot_time 306.428 ms
// 
// C:\bin\affine3D.exe data\register\vol2.raw data\cav1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 1000 2 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) iter 1000 type 2 outflag 0
// file data\register\vol2.raw read
// iter 1000 type 2 rot_time 624.514 ms
// 
// C:\bin\affine3D.exe data\register\vol2.raw data\cav1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 1000 3 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) iter 1000 type 3 outflag 0
// file data\register\vol2.raw read
// iter 1000 type 3 rot_time 420.546 ms
// 
// C:\bin\affine3D.exe data\register\vol2.raw data\cav1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 10 5 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) iter 10 type 5 outflag 0
// file data\register\vol2.raw read
// iter 10 type 5 rot_time 5133.006 ms
// RTX 3080
// C:\bin\affine3D.exe data\register\vol2.raw data\test1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 1000 1 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) iter 1000 type 1 outflag 0
// file data\register\vol2.raw read
// iter 1000 type 1 rot_time 147.507 ms
// 
// C:\bin\affine3D.exe data\register\vol2.raw data\test1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 1000 2 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) iter 1000 type 2 outflag 0
// file data\register\vol2.raw read
// iter 1000 type 2 rot_time 217.647 ms
// 
// C:\bin\affine3D.exe data\register\vol2.raw data\test1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 1000 3 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) iter 1000 type 3 outflag 0
// file data\register\vol2.raw read
// iter 1000 type 3 rot_time 152.804 ms
// 
// C:\bin\affine3D.exe data\register\vol2.raw data\test1.raw 256 256 256   256 256 256  0.0 10.0 0.0  0.0 0.0 0.0  1.0 32 8 10 5 0
// input 256x256x256 out 256x256x256 rots 0.0 0.2 0.0 trans 0.0 0.0 0.0 scale 1.000 threads (32,8) iter 10 type 5 outflag 0
// file data\register\vol2.raw read
// iter 10 type 5 rot_time 4236.734 ms

#include "cx.h"
#include "helper_math.h" 
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxtextures.h"

// This struct holds affine matritix buti from rotations and translations etc.
struct affparams {
	float4 A0;   // row 1:  Three top rows of the affine matrix
	float4 A1;   // row 2:  The translations are in the 4th column
	float4 A2;   // row3:
	float scale; // embedded scale factor (not used)
};

// Host 3 x 3 matrix support
template <typename T> int matmul3(T c[3][3],double b[3][3],double a[3][3])
{
	double temp[3][3];
	for(int i=0;i<3;i++) for(int j=0;j<3;j++){
		double sum = 0.0;
		for(int k= 0;k<3;k++) sum += b[i][k]*a[k][j];
		temp[i][j] = sum;
	}
	for(int i=0;i<3;i++) for(int j=0;j<3;j++) c[i][j] = (T)temp[i][j];
	return 0;
}

// printf a 3 x 3 matrix
int show(const char *tag, double m[3][3])
{
	printf("%s ",tag);
	for(int i=0;i<3;i++) for(int j=0;j<3;j++) printf(" %6.2f",m[i][j]);
	printf("\n");
	return 0;
}

// 3D interpolation built from lerps.
template <typename T> __host__ __device__ T interp3D(cr_Ptr<T> a,cfloat x,cfloat y,cfloat z,cint3 n)
{
	auto idx = [&n](int z,int y,int x){ return (z*n.y+y)*n.x+x; };
	if(x < -1.0f ||  x >= n.x || y < -1.0f || y >= n.y || z < -1.0f || z >= n.z) return (T)0;


	float x1 = floorf(x-0.5f);
	float y1 = floorf(y-0.5f);
	float z1 = floorf(z-0.5f);

	float ax = x - x1;
	float ay = y - y1;
	float az = z - z1;

	int kx1 = max(0,(int)x1); int kx2 = min(n.x-1,kx1+1); // in [0,n.x-1]
	int ky1 = max(0,(int)y1); int ky2 = min(n.y-1,ky1+1); // in [0,n.y-1]
	int kz1 = max(0,(int)z1); int kz2 = min(n.z-1,kz1+1); // in [0,n.z-1]

	float ly1 = lerp(a[idx(kz1,ky1,kx1)],a[idx(kz1,ky1,kx2)],ax);
	float ly2 = lerp(a[idx(kz1,ky2,kx1)],a[idx(kz1,ky2,kx2)],ax);
	float lz1 = lerp(ly1,ly2,ay); // bilinear x-y interp at z1

	float ly3 = lerp(a[idx(kz2,ky1,kx1)],a[idx(kz2,ky1,kx2)],ax);
	float ly4 = lerp(a[idx(kz2,ky2,kx1)],a[idx(kz2,ky2,kx2)],ax);
	float lz2 = lerp(ly3,ly4,ay); // bilinear x-y interp at z2

	float val = lerp(lz1,lz2,az); // trilinear interp x-y-z
	return (T)val;
}


// This version is for ushort data types. With noisy data the lerps can
// return negative values leading to false bright spots on an image
// this problem is fixed by using explicit casts to float in all the lerps
// and clamping the final float results to positve vlaues for output.
__host__ __device__ ushort interp3D_noise(cr_Ptr<ushort> a,cfloat x,cfloat y,cfloat z,cint3 n)
{
	auto idx = [&n](int z,int y,int x){ return (z*n.y+y)*n.x+x; };
	if(x < -1.0f ||  x >= n.x || y < -1.0f || y >= n.y || z < -1.0f || z >= n.z) return 0;

	float x1 = floorf(x-0.5f);
	float y1 = floorf(y-0.5f);
	float z1 = floorf(z-0.5f);

	float ax = x - x1;
	float ay = y - y1;
	float az = z - z1;

	int kx1 = max(0,(int)x1); int kx2 = min(n.x-1,kx1+1); // in [0,n.x-1]
	int ky1 = max(0,(int)y1); int ky2 = min(n.y-1,ky1+1); // in [0,n.y-1]
	int kz1 = max(0,(int)z1); int kz2 = min(n.z-1,kz1+1); // in [0,n.z-1]

	float ly1 = lerp((float)a[idx(kz1,ky1,kx1)],(float)a[idx(kz1,ky1,kx2)],ax);
	float ly2 = lerp((float)a[idx(kz1,ky2,kx1)],(float)a[idx(kz1,ky2,kx2)],ax);
	float lz1 = lerp(ly1,ly2,ay); // bilinear x-y interp at z1

	float ly3 = lerp((float)a[idx(kz2,ky1,kx1)],(float)a[idx(kz2,ky1,kx2)],ax);
	float ly4 = lerp((float)a[idx(kz2,ky2,kx1)],(float)a[idx(kz2,ky2,kx2)],ax);
	float lz2 = lerp(ly3,ly4,ay); // bilinear x-y interp at z2

	float val = lerp(lz1,lz2,az);   // trilinear interp
	val = clamp(val,0.0f,65535.0f); // this for unsigned shorts
	return (ushort)val;
}

template <typename T> int host_affine3D(r_Ptr<T> b,cr_Ptr<T> a,affparams &aff,cint3 &n,cint3 &m)
{
	auto mdx = [&m](int z,int y,int x){ return (z*m.y+y)*m.x+x; };

	float dx = (float)n.x/(float)m.x;  // need regualar
	float dy = (float)n.y/(float)m.y;  // coordinatesfor(int iz=0;iz<m.z;iz++) {
	float dz = (float)n.z/(float)m.z;  // with interp2D	float z = iz*dz-aff.scale*0.5f;

	for(int iz=0;iz<m.z;iz++) {
		float z = iz*dz - 0.5f*m.z;
		for(int iy=0;iy<m.y;iy++) {
			float y = iy*dy - 0.5f*m.y;
			for(int ix=0;ix<m.x;ix++){
				float x = ix*dx - 0.5f*m.x;
				float xr = aff.A0.x*x+aff.A0.y*y+aff.A0.z*z + dx*aff.A0.w + 0.5f*m.x;
				float yr = aff.A1.x*x+aff.A1.y*y+aff.A1.z*z + dy*aff.A1.w + 0.5f*m.y;
				float zr = aff.A2.x*x+aff.A2.y*y+aff.A2.z*z + dz*aff.A2.w + 0.5f*m.z;
				// perform trilinear interpolation using host lerp code
				b[mdx(iz,iy,ix)] = interp3D(a,xr,yr,zr,n);
			}
		}
	}
	return 0;
}

template <typename T> int host_dummy3D(r_Ptr<T> b,cr_Ptr<T> a,affparams &aff,cint3 &n,cint3 &m)
{
	auto mdx = [&m](int z,int y,int x){ return (z*m.y+y)*m.x+x; };

	float dx = (float)n.x/(float)m.x;  // need regualar
	float dy = (float)n.y/(float)m.y;  // coordinatesfor(int iz=0;iz<m.z;iz++) {
	float dz = (float)n.z/(float)m.z;  // with interp2D	float z = iz*dz-aff.scale*0.5f;

	for(int iz=0;iz<m.z;iz++) {
		float z = iz*dz - 0.5f*m.z;
		for(int iy=0;iy<m.y;iy++) {
			float y = iy*dy - 0.5f*m.y;
			for(int ix=0;ix<m.x;ix++){
				float x = ix*dx - 0.5f*m.x;
				float xr = aff.A0.x*x+aff.A0.y*y+aff.A0.z*z + dx*aff.A0.w + 0.5f*m.x;
				float yr = aff.A1.x*x+aff.A1.y*y+aff.A1.z*z + dy*aff.A1.w + 0.5f*m.y;
				float zr = aff.A2.x*x+aff.A2.y*y+aff.A2.z*z + dz*aff.A2.w + 0.5f*m.z;
				b[mdx(iz,iy,ix)] = xr+yr+zr;  // two adds instead of interpolation
			}
		}
	}
	return 0;
}

// Best performing version
__global__ void affine3D(r_Ptr<ushort> b,cudaTextureObject_t atex,affparams aff,cint3 n,cint3 m)
{
	cint ix = blockIdx.x*blockDim.x + threadIdx.x;
	cint iy = blockIdx.y*blockDim.y + threadIdx.y;
	if(ix >= m.x || iy >= m.y) return; // Check if within image bounds
	auto mdx = [&m](int z,int y,int x){ return (z*m.y+y)*m.x+x; };

	float dx = 1.0f/(float)m.x;  // using normalized coords 
	float dy = 1.0f/(float)m.y;  // for texture lookup
	float dz = 1.0f/(float)m.z;  // in a

	float x = ix*dx - 0.5f;  // move origin to
	float y = iy*dy - 0.5f;  // volume centre

	for(int iz=0;iz<m.z;iz++){  // thread loops over z
		float z = iz*dz - 0.5f;
		//         <---- affine 3 x 3 matix ------>  translation   texture    restore
		//                                            in pixels     offset    origin
		float xr = aff.A0.x*x+aff.A0.y*y+aff.A0.z*z + dx*aff.A0.w + 0.5f/n.x + 0.5f;
		float yr = aff.A1.x*x+aff.A1.y*y+aff.A1.z*z + dy*aff.A1.w + 0.5f/n.y + 0.5f;
		float zr = aff.A2.x*x+aff.A2.y*y+aff.A2.z*z + dz*aff.A2.w + 0.5f/n.z + 0.5f;
		b[mdx(iz,iy,ix)] = (ushort)(65535.0f*tex3D<float>(atex,xr,yr,zr));
	}
}

// version using interp3D
__global__ void affine3D_B(r_Ptr<ushort> b,cr_Ptr<ushort> a,cudaTextureObject_t atex,affparams aff,cint3 n,cint3 m)
{
	cint ix = blockIdx.x*blockDim.x + threadIdx.x;
	cint iy = blockIdx.y*blockDim.y + threadIdx.y;
	if(ix >= m.x || iy >= m.y) return; // Check if within image bounds
	auto mdx = [&m](int z,int y,int x){ return (z*m.y+y)*m.x+x; };

	float dx = (float)n.x/(float)m.x;  // need regualar
	float dy = (float)n.y/(float)m.y;  // coordinates
	float dz = (float)n.z/(float)m.z;  // with interp2D

	float x = ix*dx - 0.5f*m.x;
	float y = iy*dy - 0.5f*m.y;

	for(int iz=0;iz<m.z;iz++){
		float z = iz*dz - 0.5f*m.z; //*aff.scale; //- (aff.scale-1.0f)*0.5f;
		float xr = aff.A0.x*x+aff.A0.y*y+aff.A0.z*z + dx*aff.A0.w  + 0.5f*m.x;
		float yr = aff.A1.x*x+aff.A1.y*y+aff.A1.z*z + dy*aff.A1.w  + 0.5f*m.y;
		float zr = aff.A2.x*x+aff.A2.y*y+aff.A2.z*z + dz*aff.A2.w  + 0.5f*m.z;

		b[mdx(iz,iy,ix)] = interp3D(a,xr,yr,zr,n);
	}
}

// dummy version with no interpolation
__global__ void affine3D_D(r_Ptr<ushort> b,cudaTextureObject_t atex,affparams aff,cint3 n,cint3 m)
{
	cint ix = blockIdx.x*blockDim.x + threadIdx.x;
	cint iy = blockIdx.y*blockDim.y + threadIdx.y;
	if(ix >= m.x || iy >= m.y) return; // Check if within image bounds
	auto mdx = [&m](int z,int y,int x){ return (z*m.y+y)*m.x+x; };

	float dx = 1.0f/(float)m.x;  // using normalized coords 
	float dy = 1.0f/(float)m.y;  // for texture lookup
	float dz = 1.0f/(float)m.z;  // in a

	float x = ix*dx - 0.5f;  //*aff.scale; // (aff.scale-1.0f)*0.5f;
	float y = iy*dy - 0.5f;  //*aff.scale; // (aff.scale-1.0f)*0.5f;

	for(int iz=0;iz<m.z;iz++){
		float z = iz*dz - 0.5f; //*aff.scale; //- (aff.scale-1.0f)*0.5f;
		float xr = aff.A0.x*x+aff.A0.y*y+aff.A0.z*z + dx*aff.A0.w + 0.5f/n.x + 0.5f; //*aff.scale;
		float yr = aff.A1.x*x+aff.A1.y*y+aff.A1.z*z + dy*aff.A1.w + 0.5f/n.y + 0.5f; //*aff.scale;
		float zr = aff.A2.x*x+aff.A2.y*y+aff.A2.z*z + dz*aff.A2.w + 0.5f/n.z + 0.5f; //*aff.scale;

		b[mdx(iz,iy,ix)] = (ushort)(xr+yr+zr);  // Just 2 adds here
	}
}

// Similar to affine3D but using one thread per voxel.
__global__ void affine3D_C(r_Ptr<ushort> b,cudaTextureObject_t atex,affparams aff,cint3 n,cint3 m)
{
	cint ix = blockIdx.x*blockDim.x + threadIdx.x;
	cint iy = blockIdx.y*blockDim.y + threadIdx.y;
	cint iz = blockIdx.z*blockDim.z + threadIdx.z;
	if(ix >= m.x || iy >= m.y || iz >= m.z) return; // Check if within image bounds
	auto mdx = [&m](int z,int y,int x){ return (z*m.y+y)*m.x+x; };

	float dx = 1.0f/(float)m.x;  // using normalized coords 
	float dy = 1.0f/(float)m.y;  // for texture lookup
	float dz = 1.0f/(float)m.z;  // in a

	float x = ix*dx - 0.5f;
	float y = iy*dy - 0.5f;
	float z = iz*dz - 0.5f;

	float xr = aff.A0.x*x+aff.A0.y*y+aff.A0.z*z + dx*aff.A0.w + 0.5f/n.x + 0.5f;
	float yr = aff.A1.x*x+aff.A1.y*y+aff.A1.z*z + dy*aff.A1.w + 0.5f/n.y + 0.5f;
	float zr = aff.A2.x*x+aff.A2.y*y+aff.A2.z*z + dz*aff.A2.w + 0.5f/n.z + 0.5f;

	b[mdx(iz,iy,ix)] = (ushort)(65535.0f*tex3D<float>(atex,xr,yr,zr));

}

// build affparams object from supplied transfromations
int make_affine(affparams &aff,cfloat3 &rot,cfloat3 &T,float scale)
{
	scale = 1.0/scale;   // need to zoom 2.0 needs scale = 0.5;
	double A[3][3] ={{scale,0,0},{0,scale,0},{0,0,scale}};
	float  B[3][3]; // for final product

	double cz = cos(rot.z);  double sz = sin(rot.z);
	double cx = cos(rot.x);  double sx = sin(rot.x);
	double cy = cos(rot.y);  double sy = sin(rot.y);

	double RZ[3][3] ={{cz,sz,0},{-sz,cz,0},{0,0,1}};
	double RX[3][3] ={{1,0,0},{0,cx,sx},{0,-sx,cx}};
	double RY[3][3] ={{cy,0,sy},{0,1,0},{-sy,0,cy}};

	matmul3(A,RZ,A); // NB rotations do not commute
	matmul3(A,RY,A); // therefore order of these mauiplications
	matmul3(B,RX,A); // affects the result.

	aff.A0.x = B[0][0]; aff.A0.y = B[0][1]; aff.A0.z = B[0][2]; aff.A0.w = T.x;
	aff.A1.x = B[1][0]; aff.A1.y = B[1][1]; aff.A1.z = B[1][2]; aff.A1.w = T.y;
	aff.A2.x = B[2][0]; aff.A2.y = B[2][1]; aff.A2.z = B[2][2]; aff.A2.w = T.z;
	aff.scale = scale; // this value not explicily used
	return 0;
}

int main(int argc,char *argv[])
{
	//return bugs(argc,argv);

	if(argc < 3){
		printf("usage affine3D <infile> <outfile> [nx,ny,nz in] [mx,my,mz out] [rx,ry,rz rotn] [ tx,ty,tz trans] scale thread x|32 tready|8 iter|1 type|2 outflag\n");
		printf("type 1: 2D thread grid with 3D texture\n");
		printf("type 2: 2D thread grid and interp3D not texture lookup\n");
		printf("type 3: 3D thread grid with 3D texture\n");
		printf("type 4: dummy version for timing overheads\n");
		printf("type 5: host version using interp3D\n");
		printf("type 6: host dummy version\n");
		return 0;
	}
	int3 n;
	n.x = (argc >3) ? atoi(argv[3]) : 256;
	n.y = (argc >4) ? atoi(argv[4]) : 256;
	n.z = (argc >5) ? atoi(argv[5]) : 256;
	int size = n.x*n.y*n.z;

	int3 m;
	m.x = (argc > 6) ? atoi(argv[6]) : 256;
	m.y = (argc > 7) ? atoi(argv[7]) : 256;
	m.z = (argc > 8) ? atoi(argv[8]) : 256;
	int msize = m.x*m.y*m.z;

	float3 rot ={0.0f,0.0f,0.0f};
	rot.x = (argc >  9) ? atof(argv[9])  : 0.0f;
	rot.y = (argc > 10) ? atof(argv[10]) : 0.0f;
	rot.z = (argc > 11) ? atof(argv[11]) : 0.0f;
	rot *= cx::pi<float>/180.0f; // degrees to radians using overloaded operator

	float3 T ={0.0f,0.0f,0.0f}; //for translate
	T.x = (argc > 12) ? atof(argv[12]) : 0.0f;
	T.y = (argc > 13) ? atof(argv[13]) : 0.0f;
	T.z = (argc > 14) ? atof(argv[14]) : 0.0f;

	float scale = (argc > 15) ? atof(argv[15]) : (float)sqrt(3.0);
	affparams aff;
	make_affine(aff,rot,T,scale);

	int tx =      (argc > 16) ? atoi(argv[16]) : 32;
	int ty =      (argc > 17) ? atoi(argv[17]) : 8;
	int iter    = (argc > 18) ? atoi(argv[18]) : 1;
	int type    = (argc > 19) ? atoi(argv[19]) : 2;
	int doout =   (argc > 20) ? atoi(argv[20]) : 0;


	printf("input %dx%dx%d out %dx%dx%d rots %.1f %.1f %.1f trans %.1f %.1f %.1f scale %.3f threads (%d,%d) iter %d type %d outflag %d\n",n.x,n.y,n.z,m.x,m.y,m.z,rot.x,rot.y,rot.z,T.x,T.y,T.z,scale,tx,ty,iter,type,doout);

	thrust::host_vector<ushort> a(size);  // image has type ushort
	if(cx::read_raw(argv[1],a.data(),size)) return 1;
	thrust::device_vector<ushort> dev_a(size);
	dev_a = a;  // this for type 2

	cx::txs3D<ushort> atex(n,a.data(),cudaFilterModeLinear,cudaAddressModeBorder,cudaReadModeNormalizedFloat,cudaCoordNormalized);

	thrust::host_vector<ushort>   b(msize);     // for result on host
	thrust::device_vector<ushort> dev_b(msize); // for result on gpu

	thrust::host_vector<float>       r(9); // rotation

	make_affine(aff,rot,T,scale);  // build aff object 

	dim3 threads(tx,ty,1); 
	dim3 blocks((m.x+threads.x-1)/threads.x,(m.y+threads.y-1)/threads.y,1);
	dim3 blocks2((m.x+threads.x-1)/threads.x,(m.y+threads.y-1)/threads.y,m.z);  // for type 3

	float3 aspect ={(float)n.x,(float)n.y,(float)n.z};
	cx::timer tim;
	if(type==1){  // 2D thread grid with 3D texture
		for(int k=0;k<iter;k++){
			affine3D<<< blocks,threads>>>(dev_b.data().get(),atex.tex,aff,n,m);
		}
	}
	else if(type==2){ // 2D grid and interp3D not texture lookup
		for(int k=0;k<iter;k++){
			affine3D_B<<<blocks,threads>>>(dev_b.data().get(),dev_a.data().get(),atex.tex,aff,n,m);
		}
	}
	else if(type==3){  // 3D thread grid
		for(int k=0;k<iter;k++){
			affine3D_C<<< blocks2,threads>>>(dev_b.data().get(),atex.tex,aff,n,m);
		}
	}
	else if(type==4){  // dummy version for timing overheads
		for(int k=0;k<iter;k++){
			affine3D_D<<< blocks,threads>>>(dev_b.data().get(),atex.tex,aff,n,m);
		}
	}
	else if(type==5){  // host version using interp3D
		for(int k=0;k<iter;k++){
			host_affine3D(b.data(),a.data(),aff,n,m);
		}
	}
	else if(type==6){  // host dummy version
		for(int k=0;k<iter;k++){
			host_dummy3D(b.data(),a.data(),aff,n,m);
		}
	}

	cudaDeviceSynchronize();
	double rot_time = tim.lap_ms();
	printf("iter %d type %d rot_time %.3f ms\n",iter,type,rot_time);

	if(doout){
		if(type < 5) b = dev_b;
		cx::write_raw(argv[2],b.data(),msize);
	}

	return 0;
}


