// example 5.5 rotate4 

#include "cx.h"
#include "helper_math.h" 
#include "cxtimers.h"
#include "cxbinio.h"
#include "cxtextures.h" 

__global__ void rotate4(r_Ptr<uchar4> b, cudaTextureObject_t utex4, float angle, int mx, int my, float scale)
{
	cint x = blockIdx.x*blockDim.x + threadIdx.x;
	cint y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x >= mx || y >= my) return; // within image bounds?

	auto idx = [&mx](int y,int x){ return y*mx+x; };

	float xt = x - mx/2.0f;  // translate to make the centre of the 
	float yt = y - my/2.0f;  // image the centre of rotation
	float xr =  xt*cosf(angle)+ yt*sinf(angle) + mx/2.0f; // rotate and restore
	float yr = -xt*sinf(angle)+ yt*cosf(angle) + my/2.0f; // image origin 
	float xs = (xr+0.5f)*scale/mx-0.5f*scale+0.5f;        // preserve image centre
	float ys = (yr+0.5f)*scale/my-0.5f*scale+0.5f;

	float4 fb = tex2D<float4>(utex4,xs,ys);               // NB returns floats not uchars 
	b[idx(y,x)].x = (uchar)(255*fb.x);
	b[idx(y,x)].y = (uchar)(255*fb.y);
	b[idx(y,x)].z = (uchar)(255*fb.z);
	b[idx(y,x)].w = (uchar)(255*fb.w);
}



int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage rotate4 <infile> out<file> nx|512 ny|512 mx|nx my|ny angle|30.0 scale|1 iterations|1\n");
		return 0;
	}

	int nx      = (argc > 3) ? atoi(argv[3]) : 512;
	int ny      = (argc > 4) ? atoi(argv[4]) : nx;
	int mx      = (argc > 5) ? atoi(argv[5]) : nx;
	int my      = (argc > 6) ? atoi(argv[6]) : ny;
	float angle = (argc > 7) ? atof(argv[7]) : 30.0f;
	float scale = (argc > 8) ? atof(argv[8]) : 1.0f;
	int iter    = (argc > 9) ? atoi(argv[9]) : 1;
	angle *= cx::pi<float>/180.0f;

	scale = 1.0f/scale;  // For friendly user input (i.e. 2 gives zoom by 2 not shrink by 2) 
	printf("scale %f angle %f\n",scale,angle);
	int asize = nx*ny;
	int bsize = mx*my;
	printf("scale %f angle %f asize %d bsize %d\n",scale,angle,asize,bsize);
	thrustHvec<uchar3> abuf(asize); // NB here input raw image is RGB not RGBA
	thrustHvec<uchar4> a(asize);    // but process it as RGBA
	thrustHvec<uchar4> b(bsize);
	thrustDvec<uchar4> dev_a(asize);
	thrustDvec<uchar4> dev_b(bsize);

	if(cx::read_raw(argv[1],abuf.data(),asize)) return 1;
	for(int k=0;k<asize;k++){
		a[k].x = abuf[k].x;
		a[k].y = abuf[k].y;
		a[k].z = abuf[k].z;
		a[k].w = 255;       // no transparency
	}

	dev_a = a;  // copy to device

	int2 nxy ={nx,ny};
	cx::txs2D<uchar4> atex(nxy,a.data(),cudaFilterModeLinear,cudaAddressModeBorder,cudaReadModeNormalizedFloat,cudaCoordNormalized);

	dim3 threads ={16,16,1};
	dim3 blocks ={(uint)(mx+15)/16,(uint)(my+15)/16,1};

	cx::timer tim;
	for(int k=0;k<iter;k++){
		rotate4<<<blocks,threads>>>(dev_b.data().get(),atex.tex,angle,mx,my,scale);
	}
	cx::ok(cudaGetLastError());
	cx::ok(cudaDeviceSynchronize());
	double t1 = tim.lap_ms();

	b = dev_b; // get results
	cx::write_raw(argv[2],b.data(),bsize);  // NB save rotated image as RGBA for simplicity
	printf("rotate4 iterations %d time %.3f ms\n",iter,t1);

	std::atexit([]{cudaDeviceReset();});  // thrust safe reset
	return 0;
}
