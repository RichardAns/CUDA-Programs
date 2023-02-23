// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 5.6 rotate4cv
// see comments at for building with opencv 
#include "cx.h"
#include "helper_math.h" 
#include "cxbinio.h"
#include "cxtextures.h"
#include "opencv2/core.hpp"    // add openCV support
#include "opencv2/highgui.hpp"

using namespace cv;  // don't decorate Mat etc. with cv::

__global__ void rotate4(r_Ptr<uchar4> b,cudaTextureObject_t utex4,float angle,int mx,int my,float scale)
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

void opencv_to_uchar4(uchar4 *a,Mat &image)
{
	int size = image.rows*image.cols;
	for(int k=0; k<size; k++){
		a[k].z = image.data[3*k];    // BGR => RGB
		a[k].y = image.data[3*k+1];
		a[k].x = image.data[3*k+2];  // BGR => RGB
		a[k].w = 255;
	}
}

void uchar4_to_opencv(Mat &image,uchar4 *a)
{
	int size = image.rows*image.cols;
	for(int k=0; k<size; k++){
		image.data[3*k]   = a[k].z;  // RGB => BGR
		image.data[3*k+1] = a[k].y;
		image.data[3*k+2] = a[k].x;  // RGB => BGR
	}
}

int main(int argc,char *argv[])
{
    

	if(argc < 2){
		printf("usage rot4dcv <infile> out<file>  mx|nx my|ny angle|30.0 scale|1\n");
		return 0;
	}

	Mat image = imread(argv[1],IMREAD_COLOR); // read and decode image
	int nx = image.cols; // get image dimensions
	int ny = image.rows; // from image metadata
	int mx      = (argc > 3) ? atoi(argv[3]) : nx; 
	int my      = (argc > 4) ? atoi(argv[4]) : ny;
	float angle = (argc > 5) ? atof(argv[5]) : 30.0f;
	float scale = (argc > 6) ? atof(argv[6]) : 1.0f;
    // NB no timing loop used

	angle *= cx::pi<float>/180.0f;
	scale = 1.0f/scale;  // For friendly user input (i.e. 2 gives zoom by 2 not shrink by 2) 

	printf("scale %f angle %f %s size %d x %d out %d x %d\n",scale,angle,argv[1],nx,ny,mx,my);
	int asize = nx*ny;
	int bsize = mx*my;
	thrustHvec<uchar4> a(asize);
	opencv_to_uchar4(a.data(),image);
	thrustHvec<uchar4> b(bsize);
	thrustDvec<uchar4> dev_a(asize);
	thrustDvec<uchar4> dev_b(bsize);

	dev_a = a;  // copy to device

	int2 nxy ={nx,ny};
	cx::txs2D<uchar4> atex(nxy,a.data(),cudaFilterModeLinear,cudaAddressModeBorder,cudaReadModeNormalizedFloat,cudaCoordNormalized);

	dim3 threads ={16,16,1};
	dim3 blocks ={(uint)(mx+15)/16,(uint)(my+15)/16,1};
	rotate4<<<blocks,threads>>>(dev_b.data().get(),atex.tex,angle,mx,my,scale);
	cx::ok(cudaGetLastError());
	cx::ok(cudaDeviceSynchronize());

	b = dev_b; // get results

	Mat out_image(my,mx,CV_8UC3,Scalar(0));  // NB rows,cols for opencv Mat container
	uchar4_to_opencv(out_image,b.data());

	namedWindow(argv[2],WINDOW_NORMAL);     // Window for display.
	imshow(argv[2],out_image);   // show new image	
	if(waitKey(0) != ESC) imwrite(argv[2],out_image);  // wait, then write image unless ESC key detected

	return 0;
}
//--------------------------------------------------------------------------
// to build CUDA (or other) applications with open CV you need to
//
// (1) Add the opencv include directory to you compiler search path. We use
//     D:\all_code\opencv44\build\include
//
// (2) Add the opencv library directory to your linker search path. We use 
//     D:\all_code\opencv44\build\lib
//
// (3) Explicitly add required .lib files to the linker include list. We added
//
//  opencv_core440.lib
//  opencv_highgui440.lib
//  opencv_imgcodecs440.lib
//  opencv_imgproc440.lib
//  opencv_video440.lib
//  opencv_videoio440.lib
//
// (4) To run the program add the opencv binary path to your Windows system path. We use
//      D:\all_code\opencv44\build\bin\
//
// or (5) As an alternative to (4) copy the needed dll files to your working directory.
//        You will be told which file are missing when you attempt to run the .exe file.
//        We needed
//
//   opencv_core440.dll
//   opencv_highgui440.dll
//   opencv_imgcodecs440.dll
//   opencv_imgproc440.dll
//   opencv_video440.dll
//   opencv_videoio440.dll
//   tbb.dll
//-------------------------------------------------------------------------------------
