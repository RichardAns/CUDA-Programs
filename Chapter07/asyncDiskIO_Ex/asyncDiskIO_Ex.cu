// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 7.4 & 7.5 extended version (not shown in book)
// The associated run12_new.bat file can be used to run the program
// without disk caching which artificially reduces running times.
// The bat file can be found in Chapter07\asynDiskIO_Ex\data
// 
// =========================== 2070 test =============================
// 
// RTX 2070
// D:\temp\asyncDiskIO>run12_new.bat C:\Users\Richard\OneDrive\toGit2\bin . 64 12800 1
// 15:54:20.60
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 12441.884 ms gputime 11065.638 difference 1376.245, iotot 12437.022
// done
// 15:54:35.79
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 16546.677 ms gputime 11110.821 difference 5435.856, iotot 16542.042
// done
// 15:54:55.18
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 19432.707 ms gputime 11169.936 difference 8262.771, iotot 19158.694
// done
// 15:55:17.12
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 20950.266 ms gputime 11216.035 difference 9734.231, iotot 20911.181
// done
// 15:55:41.18
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 19154.099 ms gputime 11245.301 difference 7908.797, iotot 19071.200
// done
// 15:56:03.56
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 20355.519 ms gputime 11252.608 difference 9102.911, iotot 20348.180
// done
// 15:56:26.53
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 19934.922 ms gputime 11325.740 difference 8609.182, iotot 19840.964
// done
// 15:56:49.34
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 20770.608 ms gputime 11356.542 difference 9414.066, iotot 20612.708
// done
// 15:57:12.95
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 20447.162 ms gputime 11380.529 difference 9066.633, iotot 20365.528
// done
// 15:57:35.32
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 21786.833 ms gputime 11412.178 difference 10374.656, iotot 21628.433
// done
// 15:58:00.15
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 18649.454 ms gputime 11386.548 difference 7262.906, iotot 18436.460
// done
// 15:58:21.43
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 flush 0 check 0
// Total time 20903.031 ms gputime 11387.494 difference 9515.536, iotot 20752.146
// done
// 15:58:45.13
// 
// ======================================= 3080 test ==============================================
// 
// D:\temp\asyncDiskIO>run12_new.bat C:\Users\Richard\OneDrive\toGit2\bin . 64 12800 1
// 15:25:41.55
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 9372.409 ms gputime 5413.966 difference 3958.443, iotot 9367.582
// done
// 15:25:53.11
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 10931.311 ms gputime 5453.515 difference 5477.796, iotot 10926.865
// done
// 15:26:05.96
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 12652.068 ms gputime 5461.494 difference 7190.575, iotot 12647.636
// done
// 15:26:20.44
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 12878.954 ms gputime 5473.744 difference 7405.210, iotot 12874.350
// done
// 15:26:35.77
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 13921.287 ms gputime 5494.661 difference 8426.627, iotot 13916.344
// done
// 15:26:51.77
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 13248.364 ms gputime 5477.741 difference 7770.624, iotot 13243.814
// done
// 15:27:07.14
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 12989.967 ms gputime 5516.291 difference 7473.676, iotot 12985.355
// done
// 15:27:21.71
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 14345.649 ms gputime 5497.588 difference 8848.061, iotot 14284.304
// done
// 15:27:37.91
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 // gpu 1 flush 0 check 0
// Total time 14362.065 ms gputime 5494.562 difference 8867.503, iotot 14336.865
// done
// 15:27:54.34
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 13636.458 ms gputime 5496.577 difference 8139.881, iotot 13630.245
// done
// 15:28:11.11
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 13027.074 ms gputime 5494.062 difference 7533.011, iotot 13022.412
// done
// 15:28:25.71
// blocks 256 threads 256 dsize 268435456 subsets 64 dsize 268435456 fsize 4194304 ktime 12800 gpu 1 // flush 0 check 0
// Total time 13444.947 ms gputime 5495.771 difference 7949.176, iotot 13384.892
// done
// 15:28:41.69
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
	uint check = fread(buf,sizeof(T),len,fin);
	if(check != len) { printf("read_block error got %u expected %u\n",check,len); return 1; }
	return 0;
}

template <typename T> int write_block(FILE *fin,T *buf,uint len)
{
	uint check = fwrite(buf,sizeof(T),len,fin);
	if(check != len) { printf("write_block error gor %u expected %u\n",check,len); return 1; }
	return 0;
}

template <typename T> double swork(thrustHvecPin<T> &inbuf,
	thrustHvecPin<T> &outbuf,thrustDvec<T> &dev_in,thrustDvec<T> &dev_out,
	int blocks,int threads,uint fsize,int ktime,int gpu)
{
	cx::timer tim;
	if(gpu){
		dev_in = inbuf;
		if(ktime > 0)mashData<<<blocks,threads,0,0 >>>(dev_in.data().get(),dev_out.data().get(),fsize,ktime);
		checkCudaErrors(cudaGetLastError());
		outbuf = dev_out;
		//cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

	}
	double gtime = tim.lap_ms();
	return gtime;
}

template <typename T> double baseline(cchar *infile,cchar *outfile,int blocks,int threads,int frames,uint fsize,int ktime)
{
	FILE *fin = fopen(infile,"rb");  if(!fin) { printf("bad file %s\n",infile); return 1; }
	FILE *fout =fopen(outfile,"wb"); if(!fout) { printf("bad file %s\n",outfile); return 1; }
	thrustHvecPin<float> in(fsize);
	thrustHvecPin<float> out(fsize);
	thrustDvec<float> dev_in(fsize);
	thrustDvec<float> dev_out(fsize);

	cx::timer tim;
	for(int k=0;k<frames;k++){
		read_block(fin,in.data(),fsize);
		dev_in = in;
		if(ktime > 0)mashData<<<blocks,threads,0,0 >>>(dev_in.data().get(),dev_out.data().get(),fsize,ktime);
		out = dev_out;
		write_block(fout,out.data(),fsize);
	}
	double t1 = tim.lap_ms();
	return t1;
}

template <typename T> double onethread(cchar *infile,cchar *outfile,int blocks,int threads,int frames,uint fsize,int ktime,int flush)
{
	FILE *fin = fopen(infile,"rb");  if(!fin) { printf("bad file %s\n",infile); return 1; }
	FILE *fout =fopen(outfile,"wb"); if(!fout) { printf("bad file %s\n",outfile); return 1; }
	thrustHvecPin<float> in1(fsize);
	thrustHvecPin<float> out1(fsize);
	thrustHvecPin<float> in2(fsize);
	thrustHvecPin<float> out2(fsize);
	thrustDvec<float> dev_in1(fsize);
	thrustDvec<float> dev_out1(fsize);
	thrustDvec<float> dev_in2(fsize);
	thrustDvec<float> dev_out2(fsize);

	cx::timer tim;
	cx::timer rd;
	cx::timer wr;

	double tr=0.0;
	double tw = 0.0;

	read_block(fin,in1.data(),fsize); // preload first block
	tr += rd.lap_ms();

	for(int k=0;k<frames/2;k++){
		dev_in1 = in1;
		mashData<<<blocks,threads,0,0 >>>(dev_in1.data().get(),dev_out1.data().get(),fsize,ktime);

		rd.reset();
		read_block(fin,in2.data(),fsize);
		tr += rd.lap_ms();

		wr.reset();
		if(k>0)write_block(fout,out2.data(),fsize);
		tw += wr.lap_ms();

		out1 = dev_out1;  // implicit cudaDeviceSynchronize
		dev_in2 = in2;
		mashData<<<blocks,threads,0,0 >>>(dev_in2.data().get(),dev_out2.data().get(),fsize,ktime);

		rd.reset();
		if(k<frames/2-1) read_block(fin,in1.data(),fsize);
		tr += rd.lap_ms();

		wr.reset();
		write_block(fout,out1.data(),fsize);
		tw += wr.lap_ms();

		out2 = dev_out2;  // implicit cudaDeviceSynchronize		
	}

	wr.reset();
	write_block(fout,out2.data(),fsize); // write last block
	tw += wr.lap_ms();
	fclose(fin);
	fclose(fout);
	double tjob = tim.lap_ms();

	printf("onethread job %.3f read %.3f write %.3f flush %d ktime %d\n",tjob,tr,tw,flush,ktime);
	FILE *flog = fopen("onethread.log","a");
	fprintf(flog,"job %.3f read %.3f write %.3f flush %d ktime %d\n",tjob,tr,tw,flush,ktime);
	fclose(flog);

	return tjob;
}


template <typename T> double kernel_baseline(cchar *infile,cchar *outfile,int blocks,int threads,int frames,uint fsize,int ktime)
{
	FILE *fin = fopen(infile,"rb");  if(!fin) { printf("bad file %s\n",infile); return 1; }
	FILE *fout =fopen(outfile,"wb"); if(!fout) { printf("bad file %s\n",outfile); return 1; }
	thrustHvecPin<float> in(fsize);
	thrustHvecPin<float> out(fsize);
	thrustDvec<float> dev_in(fsize);
	thrustDvec<float> dev_out(fsize);

	double t1 = 0.0;
	cx::timer tim;
	for(int k=0;k<frames;k++){
		read_block(fin,in.data(),fsize);
		dev_in = in;
		tim.reset();
		mashData<<<blocks,threads,0,0 >>>(dev_in.data().get(),dev_out.data().get(),fsize,ktime);
		cudaDeviceSynchronize();
		t1 += tim.lap_ms();
		out = dev_out;
		write_block(fout,out.data(),fsize);
	}
	return t1;
}

template <typename T> double baselineX(cchar *infile,cchar *outfile,int blocks,int threads,uint size,int ktime,int flush)
{
	FILE *fin  = fopen(infile,"rb");  if(!fin) { printf("bad file %s\n",infile); return 1; }
	FILE *fout = fopen(outfile,"wb"); if(!fout) { printf("bad file %s\n",outfile); return 1; }
	thrustHvecPin<float> in(size);
	thrustHvecPin<float> out(size);
	thrustDvec<float> dev_in(size);
	thrustDvec<float> dev_out(size);

	cx::timer tim;
	cx::timer step;
	read_block(fin,in.data(),size);
	double tr = step.lap_ms();  // tr time read
	dev_in = in;
	cx::timer gpu;
	mashData<<<blocks,threads,0,0 >>>(dev_in.data().get(),dev_out.data().get(),size,ktime);
	cudaDeviceSynchronize();
	double tk = gpu.lap_ms();    // tk time kernel
	out = dev_out;
	double tc = step.lap_ms();  // tc time cuda
	write_block(fout,out.data(),size);
	double tw =step.lap_ms();   // tw time write
	double tj = tim.lap_ms();
	printf("baselineX total time %.3f kernel %.3f cuda %.3f read %.3f write %.3f\n",tj,tk,tc,tr,tw);
	FILE *flog = fopen("baselineX.log","a");
	fprintf(flog,"baselineX total time %.3f kernel %.3f cuda %.3f read %.3f write %.3f ms flush %d ktime %d\n",tj,tk,tc,tr,tw,flush,ktime);
	fclose(flog);
	return tj;
}

template <typename T> double kernel_baseline2(cchar *infile,cchar *outfile,int blocks,int threads,int frames,uint fsize,int ktime)
{
	FILE *fin = fopen(infile,"rb");  if(!fin) { printf("bad file %s\n",infile); return 1; }
	FILE *fout =fopen(outfile,"wb"); if(!fout) { printf("bad file %s\n",outfile); return 1; }
	thrustHvecPin<float> in(fsize);
	thrustHvecPin<float> out(fsize);
	thrustDvec<float> dev_in(fsize);
	thrustDvec<float> dev_out(fsize);

	double tk = 0.0;
	double tc = 0.0;
	cx::timer ker;
	cx::timer cu;
	for(int k=0;k<frames;k++){
		if(k==0)read_block(fin,in.data(),fsize);
		cu.reset();
		dev_in = in;
		ker.reset();
		mashData<<<blocks,threads,0,0 >>>(dev_in.data().get(),dev_out.data().get(),fsize,ktime);
		cudaDeviceSynchronize();
		tk += ker.lap_ms();
		out = dev_out;
		tc += cu.lap_ms();
		//write_block(fout,out.data(),fsize);
	}
	printf("kernel time %.3f cuda time %.3f ms\n",tk,tc);
	return tc;
}

template <typename T> int kcheck(cchar *name1,cchar *name2,int ktot)
{
	uint fsize = 1<<28;
	thrustHvec<T> b1(fsize);
	thrustHvec<T> b2(fsize);
	cx::read_raw<T>(name1,b1.data(),fsize,0);
	cx::read_raw<T>(name2,b2.data(),fsize,0);
	int errs = 0;
	for(uint k=0;k<fsize;k++){
		if(b1[k] == b2[k]) continue;
		printf("check mismatch for k %5d %f %f\n",k,b1[k],b2[k]);
		errs++;
		if(errs>10)break;
	}
	if(errs==0)printf("check for %s %s OK\n",name1,name2);
	else       printf("check for %s %s FAILED\n",name1,name2);
	return 0;
}

// create a set of input files A1.bin ...
// NB our script requires 12 1GB files.
template <typename T> int build_A_files(int files,uint size)
{
	thrustHvec<T> buf(size);
	for(uint k=0;k<size;k++) buf[k] = (T)k;
	char name[256];
	printf("Building %d test input files\n",files);
	for(int k=1;k<=files;k++){
		sprintf(name,"A%d.bin",k);
		cx::write_raw(name,buf.data(),size);
	}
	return 0;
}

cx::timer post;  // global
void myexit()
{
	printf("job time %.3f ms\n",post.lap_ms());
	checkCudaErrors(cudaDeviceReset());
}

int main(int argc,char *argv[])
{
	if(argc < 2) {
		printf("usage asyncDiskIO_ex <file in> <file out> <blocks> <threads> dataSize|2^28 frames|16 ktime|100 gpu|1 doflush|0 check|(0 or 1 <file>)\n");
		printf("NB if flush set >= 2 then the required input files will be generated\n");
		printf("gpu = 0: standard run as per figure 7.3\n");
		printf("gpu = 1: same as gpu = 0\n");
		printf("gpu = 2: call baseline\n");
		printf("gpu = 3: call kernel_baseline\n");
		printf("gpu = 4: call kernel_baseline2\n");
		printf("gpu = 5: call one_thread\n");
		printf("gpu = 6: call kcheck\n");
		printf("gpu = 7: call baselineX\n");
		printf("gpu > 7: same as gpu = 0\n");
		return 0;
	}

	int blocks = (argc > 3) ? atoi(argv[3]) : 256;
	int threads = (argc > 4) ? atoi(argv[4]) : 256;
	uint dsize = (argc > 5) ? 1 << atoi(argv[5]) : 1 << 28;
	int frames = (argc > 6) ? atoi(argv[6]) : 16;
	int ktime = (argc > 7) ? atoi(argv[7]) : 100;
	int gpu = (argc > 8) ? atoi(argv[8]) : 1;
	int flush = (argc > 9) ? atoi(argv[9]) : 0;
	int docheck = (argc > 10) ? atoi(argv[10]) : 0;
	uint fsize = dsize / frames;


	if(flush==1){  // flush OS file buffers for timing tests 10 GB of reads necessary
		uint flsize = 1<< 28;
		thrustHvec<float> flbuf(flsize);
		char name[256];
		printf("flushing IO read buffers");
		for(int k=0;k<flush;k++){
			sprintf(name,"A%d.bin",k+1);          
			cx::read_raw(name,flbuf.data(),flsize,0);
			// print here to defeat smart compilers
			printf(" %.3f",flbuf[flsize/2]);
		}
		printf("\n");
	}

	if(flush >= 2){
		build_A_files<float>(flush,dsize);
		return 0;
	}

	if(docheck || gpu==6){
		printf("call kcheck(%s,%s,%d)\n",argv[1],argv[2],ktime);
		return kcheck<float>(argv[1],argv[2],ktime);
	}

	printf("blocks %3d threads %3d dsize %u subsets %d dsize %u fsize %u ktime %d gpu %d flush %d check %d\n",blocks,threads,dsize,frames,dsize,fsize,ktime,gpu,flush,docheck);
	if(frames > 1 &&  frames%2 != 0){ printf("frames must be 1 or an even number\n"); return 1; }

	if(gpu==2){
		double tb = baseline<float>(argv[1],argv[2],blocks,threads,frames,fsize,ktime);
		printf("baseline time %.3f ms\n",tb);
		return 0;
	}

	else if(gpu==3){
		double tb = kernel_baseline<float>(argv[1],argv[2],blocks,threads,frames,fsize,ktime);
		printf("kernel_baseline time %.3f ms\n",tb);
		return 0;
	}
	else if(gpu==4){
		for(int k=0;k<10;k++){
			double tb = kernel_baseline2<float>(argv[1],argv[2],blocks,threads,frames,fsize,ktime);
		}
		//printf("kernel_baseline time %.3f ms\n",tb);
		return 0;
	}
	else if(gpu==5 || frames==2){
		double tb = onethread<float>(argv[1],argv[2],blocks,threads,frames,fsize,ktime,flush);
		printf("onethread time %.3f ms\n",tb);
		return 0;
	}
	else if(gpu==7 || frames==1){
		double tb = baselineX<float>(argv[1],argv[2],blocks,threads,frames*fsize,ktime,flush);
		std::atexit(myexit);
		return 0;
	}

	thrustHvecPin<float> inbuf1(fsize);
	thrustHvecPin<float> inbuf2(fsize);
	thrustHvecPin<float> outbuf1(fsize);
	thrustHvecPin<float> outbuf2(fsize);
	thrustDvec<float>    dev_in(fsize);
	thrustDvec<float>    dev_out(fsize);
	thrustHvec<double>   iotime(frames+3);

	FILE *fin =  fopen(argv[1],"rb"); if(!fin)  { printf("bad open on %s for read\n",argv[1]); return 1; }
	FILE *fout = fopen(argv[2],"wb"); if(!fout) { printf("bad open on %s for write\n",argv[2]); return 1; }

	cx::timer tim;
	tim.start();
	cx::timer io1;
	cx::timer io2;

	//double twork = 0;
	//double rwork = 0;

	int fstep = 0;
	std::thread r1;    // place holder
	std::thread w1;    // place holder
	std::thread r2;    // place holder
	std::thread w2;    // place holder
	double gputime = 0;
	while(fstep < frames+3) {
		// even fstep here (0,2,4...) 
		if(w2.joinable()) w2.join();
		if(r2.joinable()) r2.join();
		iotime[fstep] = io2.lap_ms();
		io1.reset();
		if(fstep>=2)       w1 = std::thread(write_block<float>,fout,outbuf1.data(),fsize); // async write blocks w1,w3,w5...
		if(fstep < frames) r1 = std::thread(read_block<float>,fin,inbuf1.data(),fsize);    // async read blocks r1,r3,r5...
		if(fstep >0 && fstep <= frames) gputime += swork<float>(inbuf2,outbuf2,dev_in,dev_out,blocks,threads,fsize,ktime,gpu); // c2,c4,c6...
		fstep++;

		// odd fstep here (1,3,5...)
		if(w1.joinable()) w1.join();
		if(r1.joinable()) r1.join();
		iotime[fstep] = io1.lap_ms();
		io2.reset();
		if(fstep>=3)       w2 = std::thread(write_block<float>,fout,outbuf2.data(),fsize); // async write blocks w2,w4,w6...
		if(fstep < frames) r2 = std::thread(read_block<float>,fin,inbuf2.data(),fsize);   // async read blocks r2,r4,r6...
		if(fstep >0 && fstep <= frames) gputime += swork<float>(inbuf1,outbuf1,dev_in,dev_out,blocks,threads,fsize,ktime,gpu); // c1,c3,c5...
		fstep++;
	}

	double t1 = tim.lap_ms();
	double iotot = 0.0; for(int k=0;k<frames+3;k++) iotot += iotime[k];
	printf("Total time %.3f ms gputime %.3f difference %.3f, iotot %.3f\n",t1,gputime,t1-gputime,iotot);
	FILE *flog = fopen("pipeline.log","a");
	fprintf(flog,"job %.3f ms gpu %.3f difference %.3f, iotot %.3f flush %d ktime %d frames %d\n",t1,gputime,t1-gputime,iotot,flush,ktime,frames);
	fclose(flog);


	//printf("step times\n");
	//printf("even"); for(int k=2 ;k<frames+3;k+=2) printf(" %8.3f",iotime[k]);  printf(" ms\n");
	//printf("odd "); for(int k=1;k<frames+3;k+=2) printf(" %8.3f",iotime[k]);  printf(" ms\n");

	fclose(fin);
	fclose(fout);

	//std::atexit([]{checkCudaErrors(cudaDeviceReset()); });
	std::atexit(myexit);
	printf("done\n");
	return 0;
}