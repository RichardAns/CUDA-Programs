// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program iter.exe  example I.1 (Appendix I)
// 
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc,char *atgv[])
{
	std::vector<int> a(100);  // vector of 100 elements     
	// set elements of a to (0,1,2...,99)

	// Traditional for loop
	// at each step k is an index to an element of the array
	for(int k=0;k<100;k++) a[k] = k;
	printf("a[20]= %d\n",a[20]);

	// C++ iterator over the elements of a
	// at each step iter is a pointer to an element of a
	int k=100;
	for(auto iter = a.begin(); iter != a.end(); iter++) iter[0] = k++;
	printf("a[20] = %d\n",a[20]);

	// C++11 range-based loop, 
	// at each step iter is a reference to an element
	for(auto &iter : a) iter = k++;  // NB implicit assumtion here that steps are in standard order
	printf("a[20] = %d\n",a[20]);    // appears to be true here, but maybe not guaranteed.

	return 0;
}

