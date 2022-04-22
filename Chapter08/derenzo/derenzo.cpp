// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// dernezo
// This is a support program for chapter 8 PET simualtion
// it generates a batch file for creating a denerzo rod phantom
// by running fullsim in phantom mode many times. The rods have
// different volumes and the number of generated events per rod is
// scaled to give equal counts per unit volume.

#include "cx.h"
#include <vector>
#include <math.h>


struct point {
	double x;
	double y;
};

struct dpoint {
	double x;
	double y;
	double r;
};

int hex_mesh(std::vector<dpoint> &p,double r,double theta,double cut)
{
	// Derenzo Phantom has spots spaced by 2 x diameter.
	// arranged in 6 hexagonal segments
	double s = 0.5;
	double c = sqrt(3)/2.0;
	double vscale = c;

	point m ={-s,c}; // left edge of triangle
	double st = sin(theta);
	double ct = cos(theta);

	double dist = 2.0*r;
	double dist2 = 0.0;

	dpoint spot;
	double sx = 0.0;
	double sy = 1.6*dist;  // distance from centre of first spot
	spot.x =  sx*ct + sy*st;
	spot.y = -sx*st + sy*ct;
	spot.r = r;
	p.push_back(spot);

	int nspots = 2;

	while(1){
		double dist_edge = 2.0*dist*(double)(nspots-1);
		sx = m.x*dist_edge;
		sy = m.y*dist_edge + 1.6*dist;
		dist2 = sqrt(sx*sx+sy+sy);
		if(dist2 >= cut) break;
		for(int k=0;k<nspots;k++){
			spot.x =  sx*ct + sy*st;
			spot.y = -sx*st + sy*ct;
			p.push_back(spot);
			sx += 2.0*dist;
		}
		nspots++;
		//printf("%d %f %f\n",nspots,dist2,cut);
	}
	return p.size();
}

int main(int argc,char *argv[])
{
	if (argc < 2) {
		printf("Usage derenzo OS|0 activity|0.1 savevol|0  radius|50.0 z1|64.0 z2|192.0 [r1, r2 ... r6] \n");
		printf("recommended defaults for r1-6 at radius 50.0: 3.06 4.30 4.90 6.12 8.56 12.24\n");
        printf("set OS to zero for Windows or 1 for Linux\n");
		return 0;
	}
	// fullsim contains the command to run fullsim.exe from working directory
	//const char *fullsim = "D:\\all_code\\vs17\\pet\\x64\\Release\\fullsim.exe";
    /// modifed to include Linux or Windows versions. The path the the executable is now
    // suppplied by the user as the first command line argument.

    int ostype       = (argc > 1) ? atoi(argv[1]) : 0;     // Windows if zero (default) or else Linux.
	double activity = (argc > 2) ? atof(argv[2]) : 0.1;   // activity in 10^6 decays per mm^3
	int savevol     = (argc > 3) ? atof(argv[3]) : 0;     // save volume as [r,z,phi] [100,64,400] 
	double radius   = (argc > 4) ? atof(argv[4]) : 50.0;  // outer radius of phantom mm
	double z1       = (argc > 5) ? atof(argv[5]) : 64.0;  // rod z start in [0, 256] mm
	double z2       = (argc > 6) ? atof(argv[6]) : 192.0; // rod z end in   [0, 256] mm

    // execuable name, note $1 for user supplied path
    const char* fullsim = (ostype==0) ? "$1\\fullsim.exe" : "$1/fullsim";

    // this is name of the phantom dataset created by the batch file 
    const char *phantom = "derenzo_full.raw";

    // radii of cylinder groups
	std::vector<double> rodradii ={ 3.06, 4.30, 4.90, 6.12, 8.56, 12.24 };
	for (int k=7; k<17; k++) if (argc>k) rodradii[k-7] = atof(argv[k]);

	printf("genrated %s for activity %.2f savevol %d radius %.2f z1 %.2f z2 %.2f\n", argv[1], activity, savevol, radius, z1, z2);
	printf("rod radii "); for (int k=0; k<6; k++) printf(" %8.3f", rodradii[k]); printf("\n");
   
    const char *scriptfile = (ostype==0) ? "derenzo.bat" : "derenzo.sh";
	FILE *flog = fopen(scriptfile,"w");
    if (!flog) { printf("cannot create %s \n",scriptfile); return 1; }
    if (ostype != 0)fprintf(flog, "#!/bin/bash\n");

    double theta = 0.0;	
	std::vector<dpoint> p;  // p holds all rods
	for(int k=0;k<6;k++) {
		hex_mesh(p,rodradii[k], theta,radius);
		theta += cx::pi<double>/3.0;
	}
	printf("%d points in phantom\n",(int)p.size());
	for(int k=0;k<p.size();k++){
		double volume = cx::pi<double>*p[k].r*p[k].r*(z2-z1);
		double dist = sqrt(p[k].x*p[k].x+p[k].y*p[k].y);
		//printf("%3d %12.6f %12.6f %6.1f dist %12.3f\n",k,p[k].x,p[k].y,p[k].r,dist);
		fprintf(flog,"%s 2 256 1024 %4d %4d %s 0 %6.2f 0 360 %8.3f %8.3f %12.6f %12.6f 0 %d 1\n",
			fullsim,(int)(volume*activity),1234+k*37,phantom,p[k].r,z1,z2,p[k].x,p[k].y,savevol);
	}
	double area = 0.0;
	for (int k=0; k<p.size(); k++) area += cx::pi<double>*p[k].r*p[k].r;
	double vol = area*(z2-z1);
	printf("total area %f mm^2, vol %f mm^3 activity %f MB\n", area, vol, vol*activity);
    fclose(flog);
    printf("script file %s written\n", scriptfile);
	return 0;
}