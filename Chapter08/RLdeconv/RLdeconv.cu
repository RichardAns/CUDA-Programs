// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// RLdeconv program, includes examples 8.14 and 8.15
// 
// // RTX 2070
// C:\bin\RLdeconv.exe joriginal.raw 600 320 16 16 15 10.0 20 100000
// file joriginal.raw read
// kernel is 15 x 15 iter_host 20 iter_gpu 100000 sigma 10.00
//    0.326   0.348   0.368   0.384   0.398   0.408   0.414   0.416   0.414   0.408   0.398   0.384   0.368   0.348   0.326
//    0.348   0.371   0.392   0.410   0.425   0.436   0.442   0.444   0.442   0.436   0.425   0.410   0.392   0.371   0.348
//    0.368   0.392   0.414   0.433   0.449   0.460   0.467   0.470   0.467   0.460   0.449   0.433   0.414   0.392   0.368
//    0.384   0.410   0.433   0.453   0.470   0.481   0.489   0.491   0.489   0.481   0.470   0.453   0.433   0.410   0.384
//    0.398   0.425   0.449   0.470   0.486   0.499   0.506   0.509   0.506   0.499   0.486   0.470   0.449   0.425   0.398
//    0.408   0.436   0.460   0.481   0.499   0.511   0.519   0.521   0.519   0.511   0.499   0.481   0.460   0.436   0.408
//    0.414   0.442   0.467   0.489   0.506   0.519   0.527   0.529   0.527   0.519   0.506   0.489   0.467   0.442   0.414
//    0.416   0.444   0.470   0.491   0.509   0.521   0.529   0.532   0.529   0.521   0.509   0.491   0.470   0.444   0.416
//    0.414   0.442   0.467   0.489   0.506   0.519   0.527   0.529   0.527   0.519   0.506   0.489   0.467   0.442   0.414
//    0.408   0.436   0.460   0.481   0.499   0.511   0.519   0.521   0.519   0.511   0.499   0.481   0.460   0.436   0.408
//    0.398   0.425   0.449   0.470   0.486   0.499   0.506   0.509   0.506   0.499   0.486   0.470   0.449   0.425   0.398
//    0.384   0.410   0.433   0.453   0.470   0.481   0.489   0.491   0.489   0.481   0.470   0.453   0.433   0.410   0.384
//    0.368   0.392   0.414   0.433   0.449   0.460   0.467   0.470   0.467   0.460   0.449   0.433   0.414   0.392   0.368
//    0.348   0.371   0.392   0.410   0.425   0.436   0.442   0.444   0.442   0.436   0.425   0.410   0.392   0.371   0.348
//    0.326   0.348   0.368   0.384   0.398   0.408   0.414   0.416   0.414   0.408   0.398   0.384   0.368   0.348   0.326
// file blurred_image8.raw written
// file RL_Host8_15_1000_20.raw written
// file RL_GPU8_15_1000_100000.raw written
// times host 2389.938 gpu 29271.403
// 
// RTX 3080
// C:\bin\RLdeconv.exe joriginal.raw 600 320 16 16 15 10.0 20 100000
// file joriginal.raw read
// kernel is 15 x 15 iter_host 20 iter_gpu 100000 sigma 10.00
//    0.326   0.348   0.368   0.384   0.398   0.408   0.414   0.416   0.414   0.408   0.398   0.384   0.368   0.348   0.326
//    0.348   0.371   0.392   0.410   0.425   0.436   0.442   0.444   0.442   0.436   0.425   0.410   0.392   0.371   0.348
//    0.368   0.392   0.414   0.433   0.449   0.460   0.467   0.470   0.467   0.460   0.449   0.433   0.414   0.392   0.368
//    0.384   0.410   0.433   0.453   0.470   0.481   0.489   0.491   0.489   0.481   0.470   0.453   0.433   0.410   0.384
//    0.398   0.425   0.449   0.470   0.486   0.499   0.506   0.509   0.506   0.499   0.486   0.470   0.449   0.425   0.398
//    0.408   0.436   0.460   0.481   0.499   0.511   0.519   0.521   0.519   0.511   0.499   0.481   0.460   0.436   0.408
//    0.414   0.442   0.467   0.489   0.506   0.519   0.527   0.529   0.527   0.519   0.506   0.489   0.467   0.442   0.414
//    0.416   0.444   0.470   0.491   0.509   0.521   0.529   0.532   0.529   0.521   0.509   0.491   0.470   0.444   0.416
//    0.414   0.442   0.467   0.489   0.506   0.519   0.527   0.529   0.527   0.519   0.506   0.489   0.467   0.442   0.414
//    0.408   0.436   0.460   0.481   0.499   0.511   0.519   0.521   0.519   0.511   0.499   0.481   0.460   0.436   0.408
//    0.398   0.425   0.449   0.470   0.486   0.499   0.506   0.509   0.506   0.499   0.486   0.470   0.449   0.425   0.398
//    0.384   0.410   0.433   0.453   0.470   0.481   0.489   0.491   0.489   0.481   0.470   0.453   0.433   0.410   0.384
//    0.368   0.392   0.414   0.433   0.449   0.460   0.467   0.470   0.467   0.460   0.449   0.433   0.414   0.392   0.368
//    0.348   0.371   0.392   0.410   0.425   0.436   0.442   0.444   0.442   0.436   0.425   0.410   0.392   0.371   0.348
//    0.326   0.348   0.368   0.384   0.398   0.408   0.414   0.416   0.414   0.408   0.398   0.384   0.368   0.348   0.326
// file blurred_image8.raw written
// file RL_Host8_15_1000_20.raw written
// file RL_GPU8_15_1000_100000.raw written
// times host 1749.840 gpu 11661.283
//
// This is a complete demonstration program.  It includes tools to build a gaussian filter and
// (1) Apply this filter to an input image
// (2) Apply RL deconvolution to the blurred image using the same filter.
// (3) The image file is read and written as an 8-bit grey scale raw image
//
// It is straight forward to extend this code to add openCV support to:
// (1) Read any image file.
// (2) Apply a range of deburring filters.
// (3) Use some sharpness criterion to find the "best" filter.
// (4) Display progress in an openCV window.
// (5) Add a simple GUI for interactive tuning of the deconvolution kernel.
//

#include "cx.h"
#include "cxbinio.h"
#include "cxtimers.h"
#include "helper_math.h"  // for clamp

// Create 2D truncated Gaussian Filter
int kernel2D(float *kern,int nkern,float sigma)
{
	//  create 1D Guassian Kernel
	thrustHvec<float> k1d(nkern);
	float kstep = 1.0/(sqrtf(2.0f)*sigma);
	float kl = -kstep*float(nkern)/2.0f;
	float kh = kl+kstep;
	float norm = 0.0f;
	float erf_a = erf(kl);
	for(int k=0;k<nkern;k++) {
		float erf_b = erf(kh);
		k1d[k]= erf_b-erf_a;
		norm += k1d[k];
		kl += kstep;
		kh += kstep;
		erf_a = erf_b;
	}
	for(int k=0;k<nkern;k++)k1d[k] /= norm;  // normalise element sum to 1

	// create normalised 2D Gassian filter as outer product of normalised 1D filters
	for(int i=0;i<nkern;i++) for(int j=0;j<nkern;j++) kern[nkern*i+j] = k1d[i]*k1d[j];
	return 0;
}

// Host version apply 2D filter to image a with result in image b
int kapply(float *kern,int nkern,float* a,float* b,int nx,int ny)
{
	int offset = nkern/2;
	for(int y=0;y<ny;y++){
		for(int x=0;x<nx;x++){
			float sum = 0.0f;
			for(int ky=0;ky<nkern;ky++){
				int iy = y+offset-ky;
				iy = clamp(iy,0,ny-1);
				//iy = std::min(ny-1,std::max(0,iy));  //clamp 
				if(iy >=0 && iy <ny) for(int kx=0;kx<nkern;kx++){
					int ix = x+offset-kx;
					ix = clamp(ix,0,nx-1);
					//ix = std::min(nx-1,std::max(0,ix));  //clamp
					if(ix >=0 && ix <nx) sum += a[nx*iy+ix]*kern[ky*nkern+kx];
				}
			}
			b[nx*y+x] = sum;
		}
	}
	return 0;
}

// this is used for host version
int RLstep(float *kern,int nkern,float* u2,float* u1,float *im,int nx,int ny)
{
	int size = nx*ny;
	thrustHvec<float> c(size);
	kapply(kern,nkern,u1,c.data(),nx,ny); //project curent estimate onto c
	for(int k=0;k<size;k++) c[k] = (abs(c[k] > 1.0e-06)) ? im[k]/c[k] : im[k];
	thrustHvec<float> d(size);
	kapply(kern,nkern,c.data(),d.data(),nx,ny); //project current estimate onto c
	for(int k=0;k<size;k++) u2[k] = d[k]*u1[k];
	return 0;
}

// next 3 functions are example 8.14
__device__ void convolve(int &x,int &y,cr_Ptr<float> kern,int &nkern,cr_Ptr<float> p,float &q,int &nx,int &ny)
{
	int offset = nkern/2;
	float sum = 0.0f;
	for(int ky=0;ky<nkern;ky++){
		int iy = y+offset-ky;      
		iy = clamp(iy,0,ny-1);
		for(int kx=0;kx<nkern;kx++){
			int ix = x+offset-kx;
			ix = clamp(ix,0,nx-1);
			sum += p[nx*iy+ix]*kern[ky*nkern+kx];
		}
	}
	q = sum;
}

__global__ void rl_forward(cr_Ptr<float> kern,int nkern,cr_Ptr<float> p,r_Ptr<float> c,cr_Ptr<float> q,int nx,int ny)
{
	int x = blockDim.x*blockIdx.x +threadIdx.x;
	int y = blockDim.y*blockIdx.y +threadIdx.y;
	if(x >= nx || y >= ny) return;
	float f = 0.0f;  // single element of c
	convolve(x,y,kern,nkern,p,f,nx,ny);  // a => c
	c[y*nx+x] = (abs(f) > 1.0e-06) ? q[y*nx+x]/f : q[y*nx+x];
}

__global__ void rl_backward(cr_Ptr<float> kern,int nkern,cr_Ptr<float> p1,r_Ptr<float> p2,cr_Ptr<float> c,int nx,int ny)
{
	int x = blockDim.x*blockIdx.x +threadIdx.x;
	int y = blockDim.y*blockIdx.y +threadIdx.y;
	if(x >= nx || y >= ny) return;
	float f=0.0f;
	convolve(x,y,kern,nkern,c,f,nx,ny);  // c => b
	p2[y*nx+x] = p1[y*nx+x]*f;
}

// this is example 8.15
int RLdeconv(thrustHvec<float> &image,thrustHvec<float> &kern,int nx,int ny,int nkern,int iter,dim3 &blocks,dim3 &threads)
{
	int size = nx*ny;
	thrustDvec<float> p1(size,1.0f);   // p1 & p2 are ping-pong buffers
	thrustDvec<float> p2(size);        // for RL iterations
	thrustDvec<float> c(size);         // forward proj denominator
	thrustDvec<float> q(size);         // device copy of blurred image
	thrustDvec<float> kn(nkern*nkern); // kernel
	q = image;
	kn  = kern;

	for(int k=0;k<iter;k+=2){ // ping-pong pairs of iterations
		rl_forward<<<blocks,threads>>>(kn.data().get(),nkern,p1.data().get(),c.data().get(),q.data().get(),nx,ny);
		rl_backward<<<blocks,threads>>>(kn.data().get(),nkern,p1.data().get(),p2.data().get(),c.data().get(),nx,ny);

		rl_forward<<<blocks,threads>>>(kn.data().get(),nkern,p2.data().get(),c.data().get(),q.data().get(),nx,ny);
		rl_backward<<<blocks,threads>>>(kn.data().get(),nkern,p2.data().get(),p1.data().get(),c.data().get(),nx,ny);

	}
	image = p1;
	return 0;
}

int main(int argc,char *argv[])
{
	if(argc <2){
		printf("usage RLdeconv <input file> nx|512 ny |512 threadx|16 thready|16 nkern|15 sigma|10.0 iter_host|20 iter_gpu|10000\n");
		return 0;
	}

	int nx        = (argc >2) ? atoi(argv[2]) : 512;
	int ny        = (argc >3) ? atoi(argv[3]) : 512;
	uint threadx  = (argc >4) ? atoi(argv[4]) : 32;
	uint thready  = (argc >5) ? atoi(argv[5]) : 8;
	int  nkern    = (argc >6) ? atoi(argv[6]) : 15;
	float sigma   = (argc >7) ? atof(argv[7]) : 10.0f;  // sigma in pixels
	int iter_host = (argc >8) ? atoi(argv[8]) : 20;
	int iter_gpu  = (argc >9) ? atoi(argv[9]) : 10000;

	int size = nx*ny;
	// read 8-bit test image
	thrustHvec<uchar> im8(size);
	if(cx::read_raw(argv[1],im8.data(),size)) return 1;
	
	// convert to float for processing
	thrustHvec<float> im32(size);  
	for(int k=0;k<size;k++) im32[k] = (float)im8[k];

	// build and print gaussian kernel
	thrustHvec<float> kern(nkern*nkern);
	kernel2D(kern.data(),nkern,sigma);
	printf("kernel is %d x %d iter_host %d iter_gpu %d sigma %.2f\n",nkern,nkern,iter_host,iter_gpu,sigma);
	for(int y=0;y<nkern;y++) {
		for(int x=0;x<nkern;x++) printf(" %7.3f",100.0*kern[y*nkern+x]);
		printf("\n");
	}

	// blur image with kernel and round to integer values
	thrustHvec<float> im_bl(size);  //for blurred image	
	kapply(kern.data(),nkern,im32.data(),im_bl.data(),nx,ny);         // blur
	for(int k=0;k<size;k++) im8[k] = min(255,(int)(im_bl[k]+0.5f));   // round to integer
	cx::write_raw("blurred_image8.raw",im8.data(),size);              // save (debug)
	for(int k=0;k<size;k++) im_bl[k] = im8[k];                        // promote to truncated float

	// Host RL deconvolution for timing comparison
	cx::timer tim;
	thrustHvec<float> a(size,1.0f);  // ping pong buffers, note iiter_hostialisation to ones
	thrustHvec<float> b(size);       // for RL deconvolution
	tim.reset();
	for(int k=0;k<iter_host;k+=2){  // bug was iter_host fix 9/12/20
		RLstep(kern.data(),nkern,b.data(),a.data(),im_bl.data(),nx,ny);
		RLstep(kern.data(),nkern,a.data(),b.data(),im_bl.data(),nx,ny);

	}
	double t1 = tim.lap_ms();
	char name[256];
	sprintf(name,"RL_Host8_%d_%d_%d.raw",nkern,(int)(sigma*100.0),iter_host);
	for(int k=0;k<size;k++) im8[k] = min(255,(int)(a[k]+0.5f));
	if(iter_host>1)cx::write_raw(name,im8.data(),size);
	// end host section

	// GPU RL deconvloution
	dim3 threads{threadx,thready,1};
	dim3 blocks{(uint)(nx+threadx-1)/threadx,(uint)(ny+thready-1)/thready,1};

	tim.reset();
	RLdeconv(im_bl,kern,nx,ny,nkern,iter_gpu,blocks,threads);
	double t2 = tim.lap_ms();
	sprintf(name,"RL_GPU8_%d_%d_%d.raw",nkern,(int)(sigma*100.0),iter_gpu);
	for(int k=0;k<size;k++) im8[k] = min(255,(int)(im_bl[k]+0.5f));
	if(iter_gpu>1)cx::write_raw(name,im8.data(),size);

	printf("times host %.3f gpu %.3f\n",t1,t2);

	return 0;

}