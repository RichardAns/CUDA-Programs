// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// examples 7.4 and 7.5 asyncDiskIO program

#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

#include <thread>       

__global__ void mashData(float *a,float *b,uint size,int ktime)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	for(int k = id; k < size; k += stride) {
		b[k] = 0.0f;
		for(int m = 0; m < ktime; m++) {
			b[k] += sqrtf(a[k] * a[k] + (float)(threadIdx.x % 32) + (float)m);
		}
	}
}

template <typename T> int read_block(FILE *fin,T *buf,uint len)
{
	uint check = fread(buf,sizeof(T),len,fin);   // read frame
	if(check != len) { printf("read error\n"); return 1; }
	return 0;
}

template <typename T> int write_block(FILE *fout,T *buf,uint len)
{
	uint check = fwrite(buf,sizeof(T),len,fout);  // write frame
	if(check != len) { printf("write error\n"); return 1; }
	return 0;
}

template <typename T> int swork(thrustHvecPin<T> &inbuf, thrustHvecPin<T> &outbuf,thrustDvec<T> &dev_in, 
	                                           thrustDvec<T> &dev_out, int blocks, int threads, uint fsize, int ktime)
{
	dev_in = inbuf;            // copy data to GPU
	mashData<<<blocks,threads>>>(dev_in.data().get(), dev_out.data().get(), fsize, ktime);

	cudaDeviceSynchronize();  // wait for kernel
	outbuf = dev_out;         // copy results to host

	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2) {
		printf("usage pipe_full <file in> <file out> <blocks> <threads> dataSize|2^28 frames|16 ktime|100  flush|1\n");
		return 0;
	}

	int blocks =  (argc > 3) ? atoi(argv[3]) : 256;
	int threads = (argc > 4) ? atoi(argv[4]) : 256;
	uint dsize =  (argc > 5) ? 1 << atoi(argv[5]) : 1 << 28; // data size
	int frames =  (argc > 6) ? atoi(argv[6]) : 16;
	int ktime =   (argc > 7) ? atoi(argv[7]) : 100;
	int flush =   (argc > 8) ? atoi(argv[8]) : 0;
	uint fsize = dsize / frames;  // frame size

	// Here we optionally flush OS disk cache which is ~10 GB on my Windows 10 system.
	// Note this is very slow and only defeats read caching not write caching. 
	// A better approching is to run muiltple jobs with a script file as dicussed in the text.
	// The extended version of this program that was used to constuct table 7.3.

	if(flush>0){               // flush OS disk cache
		uint flsize = 1<< 28;  // file size 1 GB needs flush=10 on test PC
		thrustHvec<float> flbuf(flsize);
		char name[256];
		for(int k=0;k<flush;k++){
			sprintf(name,"A%d.bin",k+1);  // use files A1.bin A2.bin ...
			cx::read_raw(name,flbuf.data(),flsize,0);
			printf("%s %.3f",name,flbuf[flsize/2]); // fool smart compliers
		}
	}
	thrustHvecPin<float> inbuf1(fsize);
	thrustHvecPin<float> inbuf2(fsize);
	thrustHvecPin<float> outbuf1(fsize);
	thrustHvecPin<float> outbuf2(fsize);
	thrustDvec<float>    dev_in(fsize);
	thrustDvec<float>    dev_out(fsize);

	FILE *fin =  fopen(argv[1],"rb"); // open input file
	FILE *fout = fopen(argv[2],"wb"); // open output file

	std::thread r1;    // read  thread for odd steps
	std::thread w1;    // write thread for odd steps
	std::thread r2;    // read  thread for even steps
	std::thread w2;    // write thread for even steps
	int fstep = 0;  // column counter

	cx::timer tim;
	while(fstep<frames+3) {

		// even fsteps here (= 0,2,4...)
		if(w2.joinable()) w2.join();  // wait for w2 & r2
		if(r2.joinable()) r2.join();  // to complete

		if(fstep>=2)       // async write blocks w1,w3,w5...
			w1 = std::thread(write_block<float>,fout,outbuf1.data(),fsize);
		if(fstep<frames)  // async read blocks r1,r3,r5...
			r1 = std::thread(read_block<float>,fin,inbuf1.data(),fsize);
		if(fstep >0 && fstep<=frames) // do work c2,c2,c4...
			swork<float>(inbuf2, outbuf2, dev_in, dev_out, blocks, threads, fsize, ktime);
		fstep++;

		// odd fsteps here (= 1,3,5...)
		if(w1.joinable()) w1.join();  // wait for w1 & r1
		if(r1.joinable()) r1.join();  // to complete

		if(fstep>=3)       // async write blocks w0,w2,w4...
			w2 = std::thread(write_block<float>,fout,outbuf2.data(),fsize);
		if(fstep < frames) // async read blocks r0,r2,r4...
			r2 = std::thread(read_block<float>,fin,inbuf2.data(),fsize);
		if(fstep >0 && fstep<=frames) // do work c1,c3,c5...
			swork<float>(inbuf1,outbuf1,dev_in,dev_out,blocks,threads,fsize,ktime);
		fstep++;
	}
	double t1 = tim.lap_ms(); printf("asyncDiskIO time %.3f ms\n",t1);

	fclose(fin);
	fclose(fout);

	std::atexit([]{ cudaDeviceReset(); });
	return 0;
}
