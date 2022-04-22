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

The top level directory contains:

set_vcxproj_params.bat: A Windows script to set paths required for build. 
makeall.sh:             A Linux script to build all the examples.

I am happy to receive comments and suggestions at rea1@cam.ac.uk

Richard Ansorge April 2022.

