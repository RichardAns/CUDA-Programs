// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// mpi_reduce example 9.10
//
// NB no CUDA dependence here.
// We recommend compiling with the intel compiler VS addin
// as we were unable to make the Microsoft MPI work on Windows 10
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <vector>

int main(int argc,char * argv[])
{
	// Initialize the MPI environment
	MPI_Init(&argc,&argv);

	// get number of processes and my rank
	int nproc; 	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
	int rank; 	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int root = 0;

	int frame_size = (argc > 1) ? atoi(argv[1]) : 100;
	int size = nproc*frame_size;
	if(rank==root)printf("dataset size %d frame size %d number of processes %d\n",size,frame_size,nproc);

	std::vector<int> sbuf(size);        // full data buffer
	std::vector<int> rbuf(frame_size);  // frame buffer

	int check = 0;
	if(rank==root) { // fill with test data
		for(int k=0;k<size;k++) sbuf[k] = k+1;
		for(int k=0;k<size;k++) check += sbuf[k];
	}

	// partion data into nproc fames
	if(rank==0)MPI_Scatter(sbuf.data(),frame_size,MPI_INT,rbuf.data(),frame_size,MPI_INT,root,MPI_COMM_WORLD);
	else       MPI_Scatter(nullptr,frame_size,MPI_INT,rbuf.data(),frame_size,MPI_INT,root,MPI_COMM_WORLD);

	// ..... start work for this process
	int procsum = 0;
	for(int k=0;k<frame_size;k++) procsum += rbuf[k];
	// ..... end work for this process

	// sum the the procsumvalues for all nodes and print result
	int fullsum = 0;
	MPI_Allreduce(&procsum,&fullsum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	printf("rank %d: procsum %7d fullsum %d check %d\n",rank,procsum,fullsum,check);

	MPI_Finalize();  //  tidy up

	return 0;
}
