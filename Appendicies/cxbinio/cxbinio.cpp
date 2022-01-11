// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program cxbinio example G.4

// this version of the program includes an option 
// to generate a set of test files.

#include "cx.h"
#include "cxbinio.h"
#include <vector>

// this version of the program includes an option 
// to generate a set of test files.

// this function creates a set of test files
template <typename T> int make_test(const char *head,int size,int num)
{
	thrustHvec<T> buf(size);
	char name[256];
	for(int j = 0;j<num;j++){
		for(int k=0;k<size;k++) buf[k] = (T)(size*j+k);
		sprintf(name,"%s%4.4d.raw",head,j);
		cx::write_raw(name,buf.data(),size);
	}
	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 3){
		printf("Usage binio <input tag> <output file> <input size> <makefiles|0>\n");
		printf("if makefiles is greater than zero a set of test files is generated\n");
		return 0;
	}
	// argv[1] is input file head – e.g. test for test0000.raw to test0099.raw etc.
	// argv[2] is output file
	int size      = (argc > 3) ? atoi(argv[3]) : 1024; // input size in words
	int makefiles = (argc > 4) ? atoi(argv[4]) : 0;    // number of files to generate
	 
	if(makefiles > 0){
		make_test<ushort>(argv[1], size, makefiles);
		return 0;
	}

	std::vector<ushort> buf(size);  // define data type here, implicit elsewhere
	int file = 0;

	// This is a way to use sprintf with dynamic array sizing based on
	// "safe" C++ string constants.
	// Note we have to count the added charaters including the terminating null
	// For those with long memories this is reminicent of Hollerith strings in 
	// Fortan II
	std::vector<char> name(strlen(argv[1])+9); // for head + nnnn.raw + \0
	sprintf(&name[0],"%s%4.4d.raw",argv[1],file);

	while(cx::can_be_opened(&name[0])){
		if(cx::read_raw(&name[0],buf.data(),size,0) == 0)
			cx::append_raw(argv[2],buf.data(),size);   // append if read OK
		file++;
		sprintf(&name[0],"%s%4.4d.raw",argv[1],file);  // next file in sequence
	}

	printf("good files copied to %s\n",argv[2]);
	return 0;
}
