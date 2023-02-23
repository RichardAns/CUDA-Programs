
// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// register chapter5 contans the code from examples 5.9 - 5.13
// 
// RTX 2070
// C:\bin\register.exe data\register\vol1.raw data\register\vol2.raw test1to2cav.raw 256
// file data\register\vol1.raw read
// file data\register\vol2.raw read
// starting cf  1.48153e+14
// pscale 2.000 iterations 31 cf calls 712 cf  7.18262e+12
// pscale 1.333 iterations 10 cf calls 942 cf  4.28740e+12
// pscale 0.889 iterations  7 cf calls 1103 cf  3.59204e+12
// pscale 0.593 iterations  4 cf calls 1195 cf  3.38810e+12
// pscale 0.395 iterations  2 cf calls 1241 cf  3.33894e+12
// pscale 0.263 iterations  2 cf calls 1287 cf  3.30197e+12
// pscale 0.176 iterations  1 cf calls 1310 cf  3.29348e+12
// pscale 0.117 iterations  1 cf calls 1333 cf  3.28947e+12
// pscale 0.078 iterations  1 cf calls 1356 cf  3.28689e+12
// file test1to2cav.raw written
// final cf  3.28689e+12 total cf calls 1357 reg time 637.800 ms job time 896.241 ms
// 
// C:\bin\register.exe data\register\vol2.raw data\register\vol1.raw test2to1cav.raw 256
// file data\register\vol2.raw read
// file data\register\vol1.raw read
// starting cf  1.48153e+14
// pscale 2.000 iterations 21 cf calls 482 cf  1.77929e+13
// pscale 1.333 iterations  3 cf calls 551 cf  1.71427e+13
// pscale 0.889 iterations  3 cf calls 620 cf  1.64995e+13
// pscale 0.593 iterations 41 cf calls 1563 cf  9.59296e+12
// pscale 0.395 iterations 16 cf calls 1931 cf  7.91667e+12
// pscale 0.263 iterations  4 cf calls 2023 cf  7.41033e+12
// pscale 0.176 iterations  6 cf calls 2161 cf  6.92232e+12
// pscale 0.117 iterations  3 cf calls 2230 cf  6.68465e+12
// pscale 0.078 iterations  2 cf calls 2276 cf  6.55695e+12
// file test2to1cav.raw written
// final cf  6.55695e+12 total cf calls 2277 reg time 1139.858 ms job time 1407.141 ms
// 
// RTX 3080
// C:\bin\register.exe data\register\vol1.raw data\register\vol2.raw test1to2.raw 256
// file data\register\vol1.raw read
// file data\register\vol2.raw read
// starting cf  1.48153e+14
// pscale 2.000 iterations 33 cf calls 758 cf  6.93737e+12
// pscale 1.333 iterations 10 cf calls 988 cf  4.29380e+12
// pscale 0.889 iterations  7 cf calls 1149 cf  3.58948e+12
// pscale 0.593 iterations  4 cf calls 1241 cf  3.39283e+12
// pscale 0.395 iterations  2 cf calls 1287 cf  3.34142e+12
// pscale 0.263 iterations  2 cf calls 1333 cf  3.30281e+12
// pscale 0.176 iterations  1 cf calls 1356 cf  3.29395e+12
// pscale 0.117 iterations  1 cf calls 1379 cf  3.28978e+12
// pscale 0.078 iterations  1 cf calls 1402 cf  3.28703e+12
// file test1to2.raw written
// final cf  3.28703e+12 total cf calls 1403 reg time 334.068 ms job time 526.643 ms
// 
// C:\bin\register.exe data\register\vol2.raw data\register\vol2.raw test2to1.raw 256
// file data\register\vol2.raw read
// file data\register\vol2.raw read
// starting cf  2.59529e-01
// pscale 2.000 iterations  1 cf calls 24 cf  2.59530e-01
// pscale 1.333 iterations  1 cf calls 47 cf  2.59529e-01
// pscale 0.889 iterations  1 cf calls 70 cf  2.59529e-01
// pscale 0.593 iterations  1 cf calls 93 cf  2.59530e-01
// pscale 0.395 iterations  1 cf calls 116 cf  2.59529e-01
// pscale 0.263 iterations  1 cf calls 139 cf  2.59529e-01
// pscale 0.176 iterations  1 cf calls 162 cf  2.59530e-01
// pscale 0.117 iterations  1 cf calls 185 cf  2.59529e-01
// pscale 0.078 iterations  1 cf calls 208 cf  2.59529e-01
// file test2to1.raw written
// final cf  2.59529e-01 total cf calls 209 reg time 51.201 ms job time 235.172 ms

#include "cooperative_groups.h"
#include "cx.h"
#include "helper_math.h" 
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxtextures.h"

namespace cg = cooperative_groups;

// used by GPU code
struct affparams {
	float4 A0;   // three rows of affine
	float4 A1;   // matrix, tranlations 
	float4 A2;   // in 4th column
	float scale;
};

// used by optimation code
struct paramset {
	affparams p;
	float a[7];  // theta[3], trans[3], scale
	int calls;
	int verbose;
	paramset(float sc = 1.0f,float ax = 0.0f,float ay = 0.0f,float az = 0.0f,float tx =0.0f,float ty = 0.0f,float tz = 0.0f,int cl=0,int vb=0)
	{
		a[0] = ax; a[1] = ay; a[2] = az;
		a[3] = tx; a[4] = ty; a[5] = tz;
		a[6] = sc;
		calls = cl;
		verbose = vb;
	}
};

void pshow(paramset &s)
{
	printf("A0 %8.3f %8.3f %8.3f  %8.3f\n",s.p.A0.x,s.p.A0.y,s.p.A0.z,s.p.A0.w);
	printf("A1 %8.3f %8.3f %8.3f  %8.3f\n",s.p.A1.x,s.p.A1.y,s.p.A1.z,s.p.A1.w);
	printf("A2 %8.3f %8.3f %8.3f  %8.3f\n",s.p.A2.x,s.p.A2.y,s.p.A2.z,s.p.A2.w);
	printf("s.p.scale %8.3f\n",s.p.scale);
	printf("theta %8.3f  %8.3f  %8.3f\n",s.a[0],s.a[1],s.a[2]);
	printf("trans %8.3f  %8.3f  %8.3f\n",s.a[3],s.a[4],s.a[5]);
	printf("s.scale %8.3f, calls %d vrbose %d\n",s.a[6],s.calls,s.verbose);

}

// modified from affine3D
__global__ void costfun_sumsq(r_Ptr<float> cost,r_Ptr<ushort> b,cudaTextureObject_t atex,affparams aff,cint3 n,cint3 m)
{
	cint ix = blockIdx.x*blockDim.x + threadIdx.x;
	cint iy = blockIdx.y*blockDim.y + threadIdx.y;
	if(ix >= m.x || iy >= m.y) return; // Check if within image bounds
	auto mdx = [&m](int z,int y,int x){ return (z*m.y+y)*m.x+x; };
	auto cdx = [&m](int y,int x){ return y*m.x+x; };

	float dx = 1.0f/(float)m.x;  // using normalized coords 
	float dy = 1.0f/(float)m.y;  // for texture lookup
	float dz = 1.0f/(float)m.z;  // in a

	float x = ix*dx - 0.5f;  // move origin to
	float y = iy*dy - 0.5f;  // volume centre

	if(cost != nullptr) cost[cdx(iy,ix)] = 0.0f;
	for(int iz=0;iz<m.z;iz++){
		float z = iz*dz - 0.5f;
		//         <---- affine 3 x 3 matix ------>  translation   texture    restore
		//                                            in pixels     offset    origin
		float xr = aff.A0.x*x+aff.A0.y*y+aff.A0.z*z + dx*aff.A0.w + 0.5f/m.x + 0.5f;
		float yr = aff.A1.x*x+aff.A1.y*y+aff.A1.z*z + dy*aff.A1.w + 0.5f/m.y + 0.5f;
		float zr = aff.A2.x*x+aff.A2.y*y+aff.A2.z*z + dz*aff.A2.w + 0.5f/m.z + 0.5f;
		
		if(cost != nullptr) { // normal case compute cost function (b is vol2)
			float aval = 65535.0f*tex3D<float>(atex,xr,yr,zr);
			float bval = (float)b[mdx(iz,iy,ix)];
			cost[cdx(iy,ix)] += (aval-bval)*(aval-bval);
		}    // special case get transformed volume  (b is buffer array)           
		else b[mdx(iz,iy,ix)] = (ushort)(65535.0f*tex3D<float>(atex,xr,yr,zr));
	}
}

template <typename T> __global__ void warp_reduce(T *sums,T *data,int N)
{
	auto g = cg::this_thread_block();
	auto w = cg::tiled_partition<32>(g); // explicit 32 thread warp
	int tid = g.size()*g.group_index().x + g.thread_rank(); // thread rank in grid

	T v = 0;
	int stride = gridDim.x*g.size();
	for(int k=tid;k<N;k+=stride) v += data[k];

	int id = w.thread_rank();  // thread rank in warp
	v += w.shfl_down(v,16);
	v += w.shfl_down(v,8);
	v += w.shfl_down(v,4);
	v += w.shfl_down(v,2);
	v += w.shfl_down(v,1);
	if(id == 0) atomicAdd(&sums[g.group_index().x],v);
}


// 3 x 3 matrix support
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

// build affine matrix from physical paramters
int make_params(paramset &s)
{
	double d2r = cx::pi<double>/180.0;
	double scale = 1.0/s.a[6];   // need to zoom 2.0 needs scale = 0.5;
	double A[3][3] ={{scale,0,0},{0,scale,0},{0,0,scale}};


	double cz = cos(s.a[2]*d2r);  double sz = sin(s.a[2]*d2r);
	double cx = cos(s.a[0]*d2r);  double sx = sin(s.a[0]*d2r);
	double cy = cos(s.a[1]*d2r);  double sy = sin(s.a[1]*d2r);

	double RZ[3][3] ={{cz,sz,0},{-sz,cz,0},{0,0,1}};
	double RX[3][3] ={{1,0,0},{0,cx,sx},{0,-sx,cx}};
	double RY[3][3] ={{cy,0,sy},{0,1,0},{-sy,0,cy}};

	matmul3(A,RZ,A);
	matmul3(A,RY,A);
	matmul3(A,RX,A);
	s.p.A0.x = A[0][0]; s.p.A0.y = A[0][1]; s.p.A0.z = A[0][2]; s.p.A0.w = s.a[3];
	s.p.A1.x = A[1][0]; s.p.A1.y = A[1][1]; s.p.A1.z = A[1][2]; s.p.A1.w = s.a[4];
	s.p.A2.x = A[2][0]; s.p.A2.y = A[2][1]; s.p.A2.z = A[2][2]; s.p.A2.w = s.a[5];
	s.p.scale = scale;  // NB this is 1/scale
	return 0;
}

// does all the work, behaves like a function
struct cost_functor {
	ushort *b;                           // moving image
	float *d;                            // buffer size m.x x m.y
	cudaTextureObject_t atex;            //fixed image in texture
	int3 n,m;                           // sizes of fixed and moving images
	int calls;
	cost_functor() {};
	cost_functor(ushort *bs, float *ds, cudaTextureObject_t at1, int3 n1, int3 m1) {
	   b = bs; d = ds; atex = at1; n = n1; m = m1; calls = 0;
	};
	float operator()(paramset &s) {
		make_params(s); // put this here to simplify calling code
		dim3 threads(32,16,1);
		dim3 blocks((m.x+threads.x-1)/threads.x,(m.y+threads.y-1)/threads.y,1);
		thrust::device_vector<float> dsum(1);
		costfun_sumsq<<< blocks,threads>>>(d,b,atex,s.p,n,m);
		//dsum[0] = 0.0f; now  done by the allocation
		warp_reduce<<<1,256>>>(dsum.data().get(),d,m.x*m.y);
		cudaDeviceSynchronize();
		float sum = dsum[0];
		calls++;
		return sum;
	}
};


float optimise(paramset &s,cost_functor &cf,float scale)
{
	//               <--- angles -->  < translations >  scale
	float step[7] ={2.0f,2.0f,2.0f,4.0f,4.0f,4.0f,0.05f};  // step sizes for parameters
	paramset sl = s;     // step down
	paramset sh = s;     // step up
	paramset sb = s;     // best so far
	paramset sopt = s;   // trial set

	float cost1 = cf(s);

	for(int k=0;k<7;k++){  // reduce delta on each pass
		float delta = step[k]*scale;
		sl.a[k] = s.a[k] - delta;
		sh.a[k] = s.a[k] + delta;
		float cost0 = cf(sl); float cost2 = cf(sh);
		if(cost0 > cost1 || cost2 > cost1){  //potential min
			float div = cost2+cost0-2.0f*cost1;
			if(abs(div) > 0.1) {
				float leap = delta*(cost0 - cost2)/div+s.a[k];  // optimal if parabloic
				leap = (leap < 0.0f) ? std::max(leap,s.a[k]-2.0f*delta) : std::min(leap,s.a[k]+2.0f*delta);
				sopt = s;
				sopt.a[k] = leap;
				float cnew = cf(sopt);
				if(cnew < cost1) sb.a[k] = leap;

			}
		}
		// here if parabolic maximum, so go to smallest
		else sb.a[k] = (cost0 < cost2) ? sl.a[k] : sh.a[k];
	}
	float cost3 = cf(sb);
	if(cost3 < cost1) s = sb; // only if imporved
	return cost3;
}

float optimise2(paramset &s,cost_functor &cf,float scale)
{
	//              <-- angles -->  <translations>  scale
	float step[7] ={2.0f,2.0f,2.0f, 4.0f,4.0f,4.0f, 0.005f};  // step sizes for parameters
	paramset sl = s;     // step down
	paramset sh = s;     // step up
	paramset sb = s;     // best so far
	paramset sopt = s;   // trial set

	float cost1 = cf(s);

	for(int k=0;k<7;k++){  // reduce delta on each pass
		float delta = step[k]*scale;
		sl.a[k] = s.a[k] - delta;
		sh.a[k] = s.a[k] + delta;
		float cost0 = cf(sl); float cost2 = cf(sh);
		if(cost0 > cost1 || cost2 > cost1){  //potential min
			float div = cost2+cost0-2.0f*cost1;
			if(abs(div) > 0.1) {
				float leap = delta*(cost0 - cost2)/div+s.a[k];  // optimal if parabloic
				leap = (leap < 0.0f) ? std::max(leap,s.a[k]-2.0f*delta) : std::min(leap,s.a[k]+2.0f*delta);
				sopt = s;
				sopt.a[k] = leap;
				float cnew = cf(sopt);
				if(cnew < cost1) { sb.a[k] = leap; continue; }
			}
		}
		// here if leap fails
		if(cost0 < cost1 || cost2 < cost1) sb.a[k] = (cost0 < cost2) ? sl.a[k] : sh.a[k];
	}
	float cost3 = cf(sb);
	if(cost3 < cost1) s = sb; // only if improved
	//printf("s.a: %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",s.a[0],s.a[1],s.a[2],s.a[3],s.a[4],s.a[5],s.a[6]);
	return cost3;
}


int main(int argc,char *argv[])
{
	if(argc < 4){
		printf("usage register <infile1> <infile2> <outfile> nx ny nz mx my mz\n");
		printf("file1 (size n) registered to file2 (size m) result has size m\n");
		return 0;
	}
	cx::timer job;  // start job timer right away
	int3 n; int3 m;
	n.x = (argc >4) ? atoi(argv[4]) : 256;
	n.y = (argc >5) ? atoi(argv[5]) : n.x;
	n.z = (argc >6) ? atoi(argv[6]) : n.x;

	m.x = (argc >7) ? atoi(argv[7]) : n.x;
	m.y = (argc >8) ? atoi(argv[8]) : m.x;
	m.z = (argc >9) ? atoi(argv[9]) : m.x;
	int asize = n.x*n.y*n.z;  // outfile same size as a.
	int bsize = m.x*m.y*m.z;

	thrust::host_vector<ushort> a(asize);          // source
	thrust::device_vector<ushort> dev_a(asize);

	thrust::host_vector<ushort> b(bsize);          // target
	thrust::device_vector<ushort> dev_b(bsize);

	thrust::host_vector<ushort> c(bsize);          // result (same size a b)
	thrust::device_vector<ushort> dev_c(bsize);

	int dsize = m.x*m.y;
	thrust::host_vector<float> d(dsize);         // buffer for partial cf sums
	thrust::device_vector<float> dev_d(dsize);

	thrust::device_vector<float> dev_dsum(1);    // buffer for sumover d

	if(cx::read_raw(argv[1],a.data(),asize)) return 1;
	if(cx::read_raw(argv[2],b.data(),bsize)) return 1;
	dev_b = b;

	// source image a on texture
	cx::txs3D<ushort>atex(n,a.data(),cudaFilterModeLinear,        cudaAddressModeBorder,
		                              cudaReadModeNormalizedFloat, cudaCoordNormalized);

	float scale = 1.0f; //(float)n.x/(float)m.x;
	paramset s(scale);
	//make_params(s);	
	dim3 threads(32,16,1);
	dim3 blocks((m.x+threads.x-1)/threads.x,(m.y+threads.y-1)/threads.y,1);


	cost_functor cf(dev_b.data().get(), dev_d.data().get(), atex.tex, n, m);


	float cf2 = cf(s);
	printf("starting cf %12.5e\n",cf2);

	cx::timer tim;
	float pscale = 2.0f;
	float cfold = cf2;
	float cfnew = cf2;
	int iter =0;
	for(int k=0;k<9;k++){
		while(iter <100){
			iter++;
			cfnew = optimise2(s,cf,pscale);
			if(cfnew > 0.99*cfold) break;
			cfold = cfnew;
		}
		printf("pscale %.3f iterations %2d cf calls %2d cf %12.5e\n",pscale,iter,cf.calls,cfnew);
		iter = 0;
		pscale /= 1.5f;  // 2.0 and 6 passes not as good	
	}
	double t1 = tim.lap_ms();
	costfun_sumsq<<< blocks,threads>>>(nullptr,dev_c.data().get(),atex.tex,s.p,n,m);
	cudaDeviceSynchronize();
	c = dev_c;
	cx::write_raw(argv[3],c.data(),bsize);
	float cffinal = cf(s);
	double t2 = job.lap_ms();  // end job time at end.
	printf("final cf %12.5e total cf calls %d reg time %.3f ms job time %.3f ms\n",cffinal,cf.calls,t1,t2);

	return 0;
}