// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program memtests includes examples 9.3 - 9.8 
// 
// C:\bin\memtests.exe 0..7 256 256 24
//
// RTX 2070
// test 0 total time 207.968 kernel time 11.975 ms
// test 1 total time 203.001 kernel time 5.661 ms
// test 2 total time 214.865 kernel time 12.058 ms
// test 3 total time 207.367 kernel time 5.649 ms
// test 4 total time 222.829 kernel time 5.646 ms
// test 5 total time 215.511 kernel time 5.752 ms
// test 6 total time 782.523 kernel time 60.889 ms
// test 7 total time 755.528 kernel time 67.476 ms
//
// RTX 3080
// test 0 total time 123.757 kernel time 8.076 ms
// test 1 total time 144.661 kernel time 5.939 ms
// test 2 total time 131.650 kernel time 8.045 ms
// test 3 total time 164.538 kernel time 5.926 ms
// test 4 total time 140.808 kernel time 5.964 ms
// test 5 total time 155.775 kernel time 5.826 ms
// test 6 total time 631.557 kernel time 50.210 ms
// test 7 total time 629.506 kernel time 49.840 ms
// 
// RTX 3080 Linux note tests 6 & 7 much improved
// test 0 total time 180.471 kernel time 7.095 ms
// test 1 total time 232.240 kernel time 5.586 ms
// test 2 total time 363.982 kernel time 7.407 ms
// test 3 total time 398.766 kernel time 5.548 ms
// test 4 total time 364.661 kernel time 5.570 ms
// test 5 total time 190.071 kernel time 5.586 ms
// test 6 total time 195.511 kernel time 21.775 ms
// test 7 total time 190.013 kernel time 6.196 ms
//
// NB for the uint data type used here the reduction step will overflow for
// buffer sizes greater than about 2^24. The timing information remains correct

#include "cooperative_groups.h"
#include "cx.h"
#include "cxtimers.h"
#include "helper_math.h"

namespace cg = cooperative_groups;

// best reduce version 
__global__ void reduce_warp_vl(r_Ptr<uint> sums,cr_Ptr<uint> data,uint n)
{
	auto b = cg::this_thread_block();    // thread block
	auto w = cg::tiled_partition<32>(b); // warp

	int4 v4 ={0,0,0,0};
	for(int tid = b.size()*b.group_index().x+b.thread_rank(); tid < n/4;
		tid += b.size()*gridDim.x) v4 += reinterpret_cast<const int4 *>(data)[tid];

	uint v = v4.x + v4.y + v4.z + v4.w;
	w.sync();

	v += w.shfl_down(v,16);
	v += w.shfl_down(v,8);
	v += w.shfl_down(v,4);
	v += w.shfl_down(v,2);
	v += w.shfl_down(v,1);
	if(w.thread_rank() == 0) atomicAdd(&sums[b.group_index().x],v);
}

double fill_buf(uint *buf,uint dsize)
{
	double sum = 0.0;
	for(uint k=0;k<dsize;k++) {
		buf[k] = k%419;  // just test data
		sum += buf[k];   // host sum to check correctness
	}
	return sum;
}

// (A) classic CUDA version  exaple 9.4
int reduce_classic(int blocks,int threads,uint dsize,double &t)
{
	uint *host_buf = (uint *)malloc(dsize*sizeof(uint));     // full data host
	uint *dev_buf; cudaMalloc(&dev_buf,dsize*sizeof(uint));  // full data device
	uint *dev_sum; cudaMalloc(&dev_sum,blocks*sizeof(uint)); // block sums(device only)
	uint host_tot;                                           // final sum host (scalar)
	uint *dev_tot; cudaMalloc(&dev_tot,1*sizeof(uint));      // final sum device

	double check = fill_buf(host_buf,dsize);
	cx::timer cuda;

	cudaMemcpy(dev_buf,host_buf,dsize*sizeof(uint),cudaMemcpyHostToDevice);
	reduce_warp_vl<<<blocks,threads>>>(dev_sum,dev_buf,dsize);
	reduce_warp_vl<<<     1,blocks>>>(dev_tot,dev_sum,blocks);
	cudaMemcpy(&host_tot,dev_tot,sizeof(uint),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	t = cuda.lap_ms();
	if(check != host_tot) printf("error classic: sum %u check %.0f\n",host_tot,check);

	free(host_buf);
	cudaFree(dev_buf);
	cudaFree(dev_sum);
	cudaFree(dev_tot);

	return 0;
}

// (B) CUDA classic with host pinned memory  example 9.5
int reduce_classic_pinned(int blocks,int threads,uint dsize,double &t)
{
	uint *host_buf; cudaMallocHost(&host_buf,dsize*sizeof(uint)); // full data host
	uint *dev_buf;  cudaMalloc(&dev_buf,dsize*sizeof(uint));      // full data device
	uint *dev_sum;  cudaMalloc(&dev_sum,blocks*sizeof(uint));     // block sums(device only)

	uint host_tot;                                             // final sum host
	uint *dev_tot; cudaMalloc(&dev_tot,1*sizeof(uint));  // final sum device

	double check = fill_buf(host_buf,dsize);
	cx::timer cuda;
	cudaMemcpy(dev_buf,host_buf,dsize*sizeof(uint),cudaMemcpyDefault);
	reduce_warp_vl<<<blocks,threads>>>(dev_sum,dev_buf,dsize);
	reduce_warp_vl<<<     1,blocks>>>(dev_tot,dev_sum,blocks);
	cudaMemcpy(&host_tot,dev_tot,sizeof(uint),cudaMemcpyDefault);
	cudaDeviceSynchronize();
	t = cuda.lap_ms();

	if(check != host_tot) printf("error classic pinned: sum %u check %.0f\n",host_tot,check);

	cudaFreeHost(host_buf);
	cudaFree(dev_buf);
	cudaFree(dev_sum);
	cudaFree(dev_tot);

	return 0;
}

// (C) thrust container standard host memory (not in book)
int reduce_thrust_standard(int blocks,int threads,uint dsize,double &t)
{
	thrustHvec<uint> host_buf(dsize);  // full data host
	thrustDvec<uint> dev_buf(dsize);  // full data device
	thrustDvec<uint> dev_sum(blocks); // block sums (device only)
	thrustHvec<uint> host_tot(1);     // final sum host
	thrustDvec<uint> dev_tot(1);      // final sum device

	double check = fill_buf(host_buf.data(),dsize);
	cx::timer cuda;
	dev_buf = host_buf;
	reduce_warp_vl<<<blocks,threads>>>(dev_sum.data().get(),dev_buf.data().get(),dsize);
	reduce_warp_vl<<<     1,blocks>>>(dev_tot.data().get(),dev_sum.data().get(),blocks);
	host_tot = dev_tot;
	cudaDeviceSynchronize();
	t = cuda.lap_ms();

	if(check != host_tot[0]) printf("error normal reduce standard done sum %u check %.0f\n",host_tot[0],check);

	return 0;
}

// (D) thust container with pinned host memory example 9.6
int reduce_thrust_pinned(int blocks,int threads,uint dsize,double &t)
{
	thrustHvecPin<uint> host_buf(dsize);  // full data host
	thrustDvec<uint>     dev_buf(dsize);  // full data device
	thrustDvec<uint>     dev_sum(blocks); // block sums (device only)
	thrustHvecPin<uint>  host_tot(1);     // final sum host
	thrustDvec<uint>     dev_tot(1);      // final sum device

	double check = fill_buf(host_buf.data(),dsize);
	cx::timer cuda;
	dev_buf = host_buf;
	reduce_warp_vl<<<blocks,threads>>>(dev_sum.data().get(),dev_buf.data().get(),dsize);
	reduce_warp_vl<<<     1,blocks>>>(dev_tot.data().get(),dev_sum.data().get(),blocks);
	host_tot = dev_tot;
	cudaDeviceSynchronize();
	t = cuda.lap_ms();

	if(check != host_tot[0]) printf("error normal reduce done sum %u check %.0f\n",host_tot[0],check);

	return 0;
}

// (E) thrust container with memcpy hybrid (not in book)
int reduce_thrust_hybrid(int blocks,int threads,uint dsize,double &t)
{
	thrustHvecPin<uint> host_buf(dsize); // full data host
	thrustDvec<uint> dev_buf(dsize);  // full data device
	thrustDvec<uint> dev_sum(blocks); // block sums (device only)
	thrustHvecPin<uint> host_tot(1);     // final sum host
	thrustDvec<uint> dev_tot(1);      // final sum device

	double check = fill_buf(host_buf.data(),dsize);
	cx::timer cuda;
	cudaMemcpy(dev_buf.data().get(),host_buf.data(),dsize*sizeof(uint),cudaMemcpyDefault);
	reduce_warp_vl<<<blocks,threads>>>(dev_sum.data().get(),dev_buf.data().get(),dsize);
	reduce_warp_vl<<<     1,blocks>>>(dev_tot.data().get(),dev_sum.data().get(),blocks);
	cudaMemcpy(host_tot.data(),dev_tot.data().get(),sizeof(uint),cudaMemcpyDefault);
	cudaDeviceSynchronize();
	t = cuda.lap_ms();

	if(check != host_tot[0]) printf("error reduce hybrid done sum %u check %.0f\n",host_tot[0],check);

	return 0;
}

// (F) zero-copy/mapped memory version example 9.7
int reduce_zerocopy(int blocks,int threads,uint dsize,double &t)
{
	uint *host_buf; cudaHostAlloc(&host_buf,dsize*sizeof(uint),cudaHostAllocMapped);
	uint *host_sum; cudaHostAlloc(&host_sum,blocks*sizeof(uint),cudaHostAllocMapped);
	uint *host_tot; cudaHostAlloc(&host_tot,1*sizeof(uint),cudaHostAllocMapped);

	uint *dev_buf; cudaHostGetDevicePointer(&dev_buf,host_buf,0);
	uint *dev_sum; cudaHostGetDevicePointer(&dev_sum,host_sum,0);
	uint *dev_tot; cudaHostGetDevicePointer(&dev_tot,host_tot,0);


	double check = fill_buf(host_buf,dsize);
	cx::timer cuda;

	reduce_warp_vl<<<blocks,threads>>>(dev_sum,dev_buf,dsize);
	reduce_warp_vl<<<     1,blocks>>>(dev_tot,dev_sum,blocks);
	cudaDeviceSynchronize();
	t = cuda.lap_ms();

	if(check != host_tot[0]) printf("error  mapped: sum %u check %.0f\n",host_tot[0],check);

	cudaFreeHost(host_buf);
	cudaFreeHost(host_sum);
	cudaFreeHost(host_tot);

	return 0;
}

// (G) Managed Memory Version example 9.8
int reduce_managed(int blocks,int threads,uint dsize,double &t)
{
	uint *buf; cudaMallocManaged(&buf,dsize*sizeof(uint));  // full data
	uint *sum; cudaMallocManaged(&sum,blocks*sizeof(uint));  // block sums
	uint *tot; cudaMallocManaged(&tot,sizeof(uint));  // grand total

	double check = fill_buf(buf,dsize);
	cx::timer cuda;
	reduce_warp_vl<<<blocks,threads>>>(sum,buf,dsize);
	reduce_warp_vl<<<     1,blocks>>>(tot,sum,blocks);
	cudaDeviceSynchronize(); // necessary
	t = cuda.lap_ms();

	if(check != tot[0]) printf("error managed: sum %u check %.0f\n",tot[0],check);

	cudaFree(sum);
	cudaFree(buf);
	cudaFree(tot);

	return 0;
}

// (H) advanced managed version requires Linux Driver for proper test (not in book)
int reduce_advanced_managed(int blocks,int threads,uint dsize,double &t)
{
	uint *buf; cudaMallocManaged(&buf,dsize*sizeof(uint),cudaMemAttachHost);  // full data
	uint *sum; cudaMallocManaged(&sum,blocks*sizeof(uint));  // block sums
	uint *tot; cudaMallocManaged(&tot,sizeof(uint));  // grand total

	cudaStream_t s1; cudaStreamCreate(&s1);

	double check = fill_buf(buf,dsize);
	cudaStreamAttachMemAsync(s1,buf);
	cudaDeviceSynchronize();
	cx::timer cuda;
	reduce_warp_vl<<<blocks,threads,0,s1>>>(sum,buf,dsize);
	reduce_warp_vl<<<     1,blocks>>>(tot,sum,blocks);
	cudaDeviceSynchronize(); // necessary
	t = cuda.lap_ms();

	if(check != tot[0]) printf("error unified advance reduce done sum %u check %.0f\n",tot[0],check);

	cudaStreamDestroy(s1);
	cudaFree(sum);
	cudaFree(buf);
	cudaFree(tot);

	return 0;
}

// this version has better printing than book
int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage: memtests.exe <test|0> <blocks|256> <threads|256> <size as power of 2|24>\n\n");
		printf("test = 0: classic          using malloc for host and cudaMalloc for device\n");
		printf("test = 1: classic_pinned   using cudaMallocHost for host and cudaMallocfor device\n");
		printf("test = 2: thrust_standard  using thrust for host and device\n");
		printf("test = 3: thrust_pinned    as 2 but using pinned host memory\n");
		printf("test = 4: thrust_hybrid    as 3 but using cudaMemcpy instead of thrust copy\n");
		printf("test = 5: zerocopy         using cudaHostAlloc for both host and device\n");
		printf("test = 6: managed          using cudaMallocManaged for both host and device\n");
		printf("test = 7: advanced_managed as 6 but with cudaStreamAttachMemAsync\n");
		return 0;
	}

	std::vector<std::string> tag(8);
	tag[0] = "classic";
	tag[1] = "classic pinned";
	tag[2] = "thrust";
	tag[3] = "thrust pinned";
	tag[4] = "thrust hydrid";
	tag[5] = "zero-copy/managed";
	tag[6] = "managed";
	tag[7] = "advanced managed";

	int unify  =  (argc > 1) ? atoi(argv[1]) : 0;
	int blocks =  (argc > 2) ? atoi(argv[2]) : 256;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;
	uint dsize =  (argc > 4) ? 1 << atoi(argv[4]) : 1 << 24;
	if(dsize > 16777216) printf("Warning dsize = %u, error likly if dsize > 2^24, timing results still correct\n",dsize);

	double t2 = 0.0;
	cx::timer tim;
	if     (unify==0) reduce_classic(blocks,threads,dsize,t2);
	else if(unify==1) reduce_classic_pinned(blocks,threads,dsize,t2);
	else if(unify==2) reduce_thrust_standard(blocks,threads,dsize,t2);
	else if(unify==3) reduce_thrust_pinned(blocks,threads,dsize,t2);
	else if(unify==4) reduce_thrust_hybrid(blocks,threads,dsize,t2);
	else if(unify==5) reduce_zerocopy(blocks,threads,dsize,t2);
	else if(unify==6) reduce_managed(blocks,threads,dsize,t2);
	else if(unify==7) reduce_advanced_managed(blocks,threads,dsize,t2);
	else { printf("unify must be in range 0-7 got %d\n",unify); return 1; }

	double t1 = tim.lap_ms();
	//printf("test %s total time %.3f kernel time %.3f ms\n",tag[unify].c_str(),t1,t2);
	printf("test %d total time %.3f kernel time %.3f ms\n",unify,t1,t2);

	FILE *flog = fopen("unify.txt","a");
	int a4 = (argc > 4) ? atoi(argv[4]) : 24;
	fprintf(flog,"%2d %2d %.3f %.3f\n",unify,a4,t1,t2);
	fclose(flog);

	std::atexit([]{cudaDeviceReset();});

	return 0;
}

// version of main in book (not used here)
int main_book(int argc,char *argv[])
{
	int test  =   (argc > 1) ? atoi(argv[1]) : 0;
	int blocks =  (argc > 2) ? atoi(argv[2]) : 256;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;
	uint dsize =  (argc > 4) ? 1 << atoi(argv[4]) : 1 << 24;
	double t2 = 0.0;  // kernel time

	cx::timer tim;
	if(test==0)      reduce_classic(blocks,threads,dsize,t2);
	else if(test==1) reduce_classic_pinned(blocks,threads,dsize,t2);
	else if(test==2) reduce_thrust_standard(blocks,threads,dsize,t2);
	else if(test==3) reduce_thrust_pinned(blocks,threads,dsize,t2);
	else if(test==4) reduce_thrust_hybrid(blocks,threads,dsize,t2);
	else if(test==5) reduce_zerocopy(blocks,threads,dsize,t2);
	else if(test==6) reduce_managed(blocks,threads,dsize,t2);
	else  return 1;

	double t1 = tim.lap_ms();
	printf("test %d total time %.3f kernel time %.3f ms\n",test,t1,t2);

	std::atexit([]{cudaDeviceReset();});
	return 0;
}

