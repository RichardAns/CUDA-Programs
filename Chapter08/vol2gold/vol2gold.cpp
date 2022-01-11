// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// vol2gold
// this is asupport program for chapter 8 PET simualtion
// converts fillsim generated volume file to floats as stores
// result in current directory. The reco program assumes output file is
// name gold.raw and optionallly uses it comupute a chisqd with respect
// the the reconstucted file durin iterations.
//
// NB reco.exe now converts vol files from uint to float directly so this progrma
// is no longer necessary. It is still necesary to copy a vol file to gold.raw.
//
#include <stdio.h>
#include <stdlib.h>
#include "cx.h"
#include "cxbinio.h"
#include "scanner.h"


int main(int argc,char *argv[])
{
	if(argc < 2 ) {
		printf("usage gold.exe <input fullsim volume> <output|gold.raw>\n");
		return 0;
	}

	uint csize = cryNum*radNum*zNum;
	thrustHvec<uint>  a(csize);
	thrustHvec<float> b(csize);
	if(cx::read_raw(argv[1],a.data(),csize)) return 1;
	for(uint k=0;k<csize;k++) b[k] = (float)a[k];
	if(argc < 3) cx::write_raw("gold.raw",b.data(),csize);
	else         cx::write_raw(argv[2],b.data(),csize);

	return 0;
}
