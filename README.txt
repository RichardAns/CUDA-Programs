This repository contains all the examples for my book 
     “Programming in Parallel with CUDA”. 

Each example is in a separate directory containing a Visual Studio project file for Windows
and a Makefile for Linux. 

Additional directories are:

bin:            Windows executables
build:          A VS19 solution file to build the entire example set on Windows. The Widowws executables are placed in bin.
data:           Data for image processing examples. 
Chapter08/data: Data files for PET simulation and reconsuction.
include:        Contains cx.h and other include files used by the examples.
Linux:          Linux executables (Ubuntu 22.04)

The top level directory CUDA-Programs contains:

set_vcxproj_params.bat: A Windows script to set paths required for build
makeall.sh:             A Linux script to build all the examples.

One of these files should be edited to reflect your installation paths.

Once the Windows bat file has been edited and run, one of the solution files
 
   CUDA-Programs\build\build_vs2019.sln or build_vs2022.sln

can be used to build the examples.
 



I am happy to receive comments and suggestions at rea1@cam.ac.uk

Richard Ansorge April 2022.

