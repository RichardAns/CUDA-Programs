// file cxbinio.h
#pragma once
// Copyright Richard Ansorge 2020. 
// Part of cx utility suite devevoped for CUP Publication:
//         Programming in Parallel with CUDA 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma warning( disable : 4996)  // warnings: unsafe functions e.g fread 
//=================================================================================
// Binary IO functions for POD data. supports read, write & append.
// These functions mostly return 0 on success or 1 on error. The optional
// argument, verbose, controls the output of messages confirming the operation. 
// The defaults vary. I find these messages, while useful for debugging, can
// become irritating when processing a lot of files. If you prefer to use
// iostream instead of plain old C feel free to make your own versions.
// I prefer the C versions as iostream is irriatatingly verbose. Also  
// versions of these functions are old friends, I have been using them since 
// 1968 when I worked with an IBM 360 mainframe doing IO to mag tapes.
//==================================================================================
namespace cx {
// read an exisitng file. 
template <typename T> int read_raw(const char *name,T *buf,size_t len,int verbose=1)
{
	FILE *fin = fopen(name,"rb");
	if(!fin) { printf("bad open on %s for read\n",name); return 1; }
	size_t check = fread(buf,sizeof(T),len,fin);
	if(check != len) { printf("bad read on %s got %zd items expected %zd\n",name,check,len); fclose(fin); return 1; }
	if(verbose)printf("file %s read\n",name);
	fclose(fin);
	return 0;
}

// write to new file (overwrites any prexisiting version without warning so use with CARE)
template <typename T> int  write_raw(const char *name,T *buf,size_t len,int verbose=1)
{
	FILE *fout = fopen(name,"wb");
	if(!fout) { printf("bad open on %s for write\n",name); return 1; }
	size_t check = fwrite(buf,sizeof(T),len,fout);
	if(check != len) { printf("bad writeon %s got %d items expected %d\n",name,(int)check,(int)len); fclose(fout); return 1; }
	if(verbose)printf("file %s written\n",name);
	fclose(fout);
	return 0;
}

// append to existing file or create new file
template <typename T> int  append_raw(const char *name,T *buf,size_t len,int verbose=1)
{
	FILE *fout = fopen(name,"ab");
	if(!fout) { printf("bad open on %s for append\n",name); return 1; }
	int check = (int)fwrite(buf,sizeof(T),len,fout);
	if(check != len) { printf("bad append on %s got %d items expected %d\n",name,check,(int)len); fclose(fout); return 1; }
	if(verbose)printf("file %s appended\n",name);
	fclose(fout);
	return 0;
}

// read len WORDS skiping first skip BYTES
template <typename T> int read_raw_skip(const char *name,T *buf,size_t len,size_t skip,int verbose=0)
{
	FILE *fin = fopen(name,"rb");
	if(!fin) { printf("bad open on %s for read\n",name); return 1; }
	if(fseek(fin,(long)skip,SEEK_SET)) { printf("seek error on %s skip =%lld\n",name,skip); return 1; }
	size_t check = fread(buf,sizeof(T),len,fin);
	if(check != len) { printf("bad read on %s got %lld items expected %lld\n",name,check,len); fclose(fin); return 1; }
	if(verbose)printf("file %s read skiping %lld bytes\n",name,skip);
	fclose(fin);
	return 0;
}

// returns length of file as bytes/sizeof(t) i.e. type-T words
template <typename T> size_t length_of(const char *name)
{
	//if(verbose)printf("length_of for %s\n",name);
	FILE *fin = fopen(name,"rb");
	if(!fin) { printf("bad open %s\n",name); return 0; }

	fseek(fin,0,SEEK_END);
	long offset = ftell(fin);
	size_t len = offset/sizeof(T);
	fclose(fin);
	return len;
}

// test if file exists and is available for reading NB name change 6/1/21
int can_be_opened(const char *name)
{
	FILE *fin = fopen(name,"r");
	if(fin==nullptr) return 0;
	else fclose(fin);
	return 1;
}

} // end namespace

// end file cxbinio.h