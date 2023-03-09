rem
rem This bat file sets the environment variables needed by
rem the vcxproj files used to build the CUDA examples
rem 
rem Please edit to correspond to your CUDA release version and GPU
rem CC level
rem
rem setx CUDA_SM   "compute_75,sm_75"
rem setx CUDA_SM   "compute_86,sm_86"
rem
setx CUDA_SM   "compute_75,sm_75;compute_86,sm_86;"
setx CUDA_VER  "12.0"
setx CX_ROOT   "C:\Users\Richard\CUDA-Programs\include"
rem
rem these for opencv used by a few of the examples for visualization
rem
setx OpenCV_Root "D:\opencv454\build"
setx OpenCV_Lib  "opencv_world454.lib"
rem