// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program smsplit support program for Chapter 8 OSEM PET reconstuction
// Sort system matrix into OSEM subsets as discussed in text

#include "cx.h"
#include "cxbinio.h"
#include "cxtimers.h"
#include "scanner.h"
#include <vector>

int main(int argc,char *argv[])
{
	if(argc < 4){
		printf("Usage smsplit <sysmat file> <systab file> <split|2>  tag|split dzcut|63 valcut|0\n");
		return 0;
	}

	int dzcut =   (argc > 5) ? atoi(argv[5]) : zNum-1;
	float valcut = (argc >6) ? atof(argv[6]) : 0.0f;

	std::vector<smTab> tab(radNum);
	if(cx::read_raw(argv[2],tab.data(),tab.size())) return 1;
	uint sm_size = tab[radNum-1].end;
	std::vector<smPart> sm(sm_size);
	if(cx::read_raw(argv[1],sm.data(),sm.size())) return 1;

	uint splits = atoi(argv[3]);

	if(splits >1 && splits%2==1) { printf("only splits =1 or even value supported\n"); return 1; }
	//if(splits < 1 || splits > 128) return 1;  // sanity

	std::vector< std::vector<smPart> > smset(splits);
	std::vector< std::vector<smTab> > tabset(splits,std::vector<smTab>(radNum));  // !! note syntax ctor(size,value)
	for(uint k=0;k<splits;k++) tabset[k][0].start = 0;

	// for c1%splits option  go through full sm here ring by ring and
	// push back to subset[c1%spilts]...
	//uint set = 0; // set to zero only once
	//for(int r=0;r<radNum;r++){
	//	if(r>0)for(uint k=0;k<splits;k++) tabset[k][r].start = tabset[k][r-1].end;
	//	for( uint smpos = tab[r].start; smpos <tab[r].end;smpos++){
	//		smset[set].push_back(sm[smpos]);
	//		set = (set < splits-1) ? set+1 : 0;		
	//	}
	//	for(uint k=0;k<splits;k++) tabset[k][r].end = smset[k].size();
	//}


	std::vector<uint> nset(splits);

	uint good = 0;
	uint bad = 0;
	double valgood = 0.0;
	double valbad = 0.0;
	uint flip[2]{0,0};
	for(int r=0;r<radNum;r++){
		for(uint k=0;k<splits;k++) 	{
			if(r>0) tabset[k][r].start = tabset[k][r-1].end;  // next tab starts after previous
			tabset[k][r].ring =      tab[r].ring;
			tabset[k][r].phi_steps = tab[r].phi_steps;
		}
		for(uint smpos = tab[r].start; smpos <tab[r].end;smpos++){
			smLor lor = key2lor(sm[smpos].key);
			if(lor.zsm1+lor.zsm2 > dzcut) continue;
			float val = sm[smpos].val;
			if(val < valcut){
				valbad += val;
				bad++;
			}
			else{
				int set = 0;
				if(splits==2) {
					set = (lor.c1+lor.c2)%2 == 0 ? 1 : 0;	// odd c1+c2 subset 0, even c1+c2 subset 1
				}
				else if(splits==4){
					set = (lor.c1+lor.c2)%2 == 0 ? 2 : 0;  // parsets eiter ee/oo or oe/eo thus even/odd c1 in both
					set += lor.c1%2;
				}
				else if(splits==8){
					set = (lor.c1+lor.c2)%2 == 0 ? 4 : 0;  // parsets eiter ee/oo or oe/eo thus even/odd c1 in both
					set += lor.c1%4;
				}

				else if(splits>0){     // this does not work well splits > 8 not recomeded 
					set = flip[0]%splits;
					flip[0]++;
				}
				smset[set].push_back(sm[smpos]);
				nset[set]++;
				valgood += val;
				good++;
			}
		}
		for(uint k=0;k<splits;k++) tabset[k][r].end = smset[k].size();
	}
	double vfrac = 100.0*valbad/(valbad+valgood);
	double nfrac = 100.0*(double)bad/(double)(good+bad);
	printf("nset"); for(uint k=0;k<splits;k++) printf(" %u",nset[k]);
	printf(" good %d %.0f bad %d %.0f fractions %.3f%% %.3f%% \n",good,valgood,bad,valbad,nfrac,vfrac);
	// store the mini sysmat files in one stack so need to adjust offsets by size of all previous mini sysmats
	uint offset = smset[0].size();
	for(uint k=1;k<splits;k++){
		for(int r=0;r<radNum;r++){
			tabset[k][r].start += offset;
			tabset[k][r].end += offset;
		}
		offset += smset[k].size();
	}

	char name[256];
	if(argc < 5)sprintf(name,"sysmat_split%d.raw",splits);
	else        sprintf(name,"sysmat_%s%d.raw",argv[4],splits);
	cx::write_raw(name,smset[0].data(),smset[0].size());
	for(uint k=1;k<splits;k++) cx::append_raw(name,smset[k].data(),smset[k].size());

	if(argc < 5)sprintf(name,"systab_split%d.raw",splits);
	else        sprintf(name,"systab_%s%d.raw",argv[4],splits);
	cx::write_raw(name,tabset[0].data(),tabset[0].size());
	for(uint k=1;k<splits;k++) cx::append_raw(name,tabset[k].data(),tabset[k].size());

	// debug
	//for(int r=0;r<radNum;r++){
	//	//uint sum =0;
	//	for(uint k=0;k<splits;k++) {
	//		printf(" %9d",tabset[k][r].start);
	//		//sum += tabset[k][r].start;
	//	}
	//	printf(" %9d  | ",tab[r].start);
	//	//sum = 0;
	//	for(uint k=0;k<splits;k++) {
	//		printf(" %9d",tabset[k][r].end);
	//		//sum += tabset[k][r].end;
	//	}
	//	printf(" %9d\n",tab[r].end);
	//}
	printf("flip %d %d\n",flip[0],flip[1]);
	return 0;
}