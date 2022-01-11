// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// Example 9.12 Use of MIP_Alltoall

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <vector>

void showmat(const char *tag,std::vector<int> &m,int nx,int ny)
{
	printf("\n%s\n",tag);
	for(int y=0;y<ny;y++){
		for(int x=0;x<nx;x++) printf(" %3d",m[nx*y+x]);
		printf("\n");
	}
}

int main(int argc,char * argv[])
{
	// Initialize the MPI environment
	MPI_Init(&argc,&argv);

	// get number of processes and my rank
	int nproc; 	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
	int rank; 	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int root = (argc > 1) ? atoi(argv[1]) : 0;

	int N = nproc;  // matrix size is number of proesses
	int size = N*N;
	std::vector<int> row(N);
	std::vector<int> col(N);
	std::vector<int> mat;  // empty placeholder

	if(rank==root) {
		for(int k=0;k<size;k++) mat.push_back(k+1);
		showmat("matrix",mat,N,N);
	}
	MPI_Barrier(MPI_COMM_WORLD); // just in case

	// copy one row to each process
	MPI_Scatter(mat.data(),N,MPI_INT,row.data(),N,MPI_INT,root,MPI_COMM_WORLD);

	// get per process row data by sharing column data
	MPI_Alltoall(row.data(),1,MPI_INT,col.data(),1,MPI_INT,MPI_COMM_WORLD);

	// gather all coluns to get transpose in root process
	MPI_Gather(col.data(),N,MPI_INT,mat.data(),N,MPI_INT,root,MPI_COMM_WORLD);
	if(rank==root) showmat("transpose",mat,N,N);

	MPI_Finalize();  //  tidy up

	return 0;
}

// alternative overwiting row with columns
//MPI_Alltoall(MPI_IN_PLACE,1,MPI_INT,row.data(),1,MPI_INT,MPI_COMM_WORLD);
//MPI_Gather(row.data(),N,MPI_INT,mat.data(),N,MPI_INT,root,MPI_COMM_WORLD);
//if(rank==root) showmat("transpose using MPI_IN_PLACE",mat,N,N);



