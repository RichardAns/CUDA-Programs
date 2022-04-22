#!/bin/bash
#
# This script file will build all examples. Note a
# single argument, elther all or clean, must be used
#
# e.g:   ./makeall.sh all  or ./makeall clean
#
# This for icc (needed by avxsaxpy & ompsaxpy)
#
source /opt/intel/oneapi/setvars.sh
#
# ensure Linux directory is present
#
mkdir -p Linux
#
# run seperate Makefiles
#
cd Appendicies/atomicCAS       ;    make "$1" NAME=atomicCAS
cd ../../Appendicies/avxsaxpy  ;    make "$1" NAME=avxsaxpy 
cd ../../Appendicies/cpusaxpy  ;    make "$1" NAME=cpusaxpy 
cd ../../Appendicies/cxbinio  ;     make "$1" NAME=cxbinio  
cd ../../Appendicies/gpusaxpy  ;    make "$1" NAME=gpusaxpy 
cd ../../Appendicies/iter       ;   make "$1" NAME=iter
cd ../../Appendicies/ompsaxpy  ;    make "$1" NAME=ompsaxpy 
cd ../../Appendicies/saxpy_vs  ;    make "$1" NAME=saxpy_vs 
cd ../../Chapter01/cpusum  ;        make "$1" NAME=cpusum  
cd ../../Chapter01/gpusum  ;        make "$1" NAME=gpusum  
cd ../../Chapter01/ompsum  ;        make "$1" NAME=ompsum  
cd ../../Chapter02/blasmult  ;      make "$1" NAME=blasmult  
cd ../../Chapter02/gpumult0  ;      make "$1" NAME=gpumult0  
cd ../../Chapter02/gpumult1  ;      make "$1" NAME=gpumult1  
cd ../../Chapter02/gpumult2  ;      make "$1" NAME=gpumult2  
cd ../../Chapter02/gpusum_tla  ;    make "$1" NAME=gpusum_tla
cd ../../Chapter02/gputiled  ;      make "$1" NAME=gputiled  
cd ../../Chapter02/gputiled1  ;     make "$1" NAME=gputiled1 
cd ../../Chapter02/grid3D     ;     make "$1" NAME=grid3D    
cd ../../Chapter02/grid3D_linear ;  make "$1" NAME=grid3D_linear 
cd ../../Chapter02/hostmult0  ;     make "$1" NAME=hostmult0
cd ../../Chapter02/hostmult1  ;     make "$1" NAME=hostmult1
cd ../../Chapter02/reduce0  ;       make "$1" NAME=reduce0  
cd ../../Chapter02/reduce1  ;       make "$1" NAME=reduce1  
cd ../../Chapter02/reduce2  ;       make "$1" NAME=reduce2  
cd ../../Chapter02/reduce3  ;       make "$1" NAME=reduce3  
cd ../../Chapter02/reduce4  ;       make "$1" NAME=reduce4  
cd ../../Chapter02/reduce_shared  ; make "$1" NAME=reduce_shared 
cd ../../Chapter02/shared_example ; make "$1" NAME=shared_example
cd ../../Chapter03/cgwarp  ;        make "$1" NAME=cgwarp  
cd ../../Chapter03/coop3D  ;        make "$1" NAME=coop3D  
cd ../../Chapter03/deadlock  ;      make "$1" NAME=deadlock 
cd ../../Chapter03/deadlock_coal  ; make "$1" NAME=deadlock_coal
cd ../../Chapter03/reduce5  ;       make "$1" NAME=reduce5 
cd ../../Chapter03/reduce6  ;       make "$1" NAME=reduce6 
cd ../../Chapter03/reduce7  ;       make "$1" NAME=reduce7 
cd ../../Chapter03/reduce7_vl    ;  make "$1" NAME=reduce7_vl
cd ../../Chapter03/reduce7_vl_coal ; make "$1" NAME=reduce7_vl_coal
cd ../../Chapter03/reduce7_vl_coal_any ; make "$1" NAME=reduce7_vl_coal_any
cd ../../Chapter03/reduce8       ;   make "$1" NAME=reduce8     
cd ../../Chapter03/reduce8_vl    ;   make "$1" NAME=reduce8_vl  
cd ../../Chapter04/batcher9PT    ;   make "$1" NAME=batcher9PT  
cd ../../Chapter04/cascade       ;   make "$1" NAME=cascade     
cd ../../Chapter04/filter9PT     ;   make "$1" NAME=filter9PT   
cd ../../Chapter04/filter9PT_2   ;   make "$1" NAME=filter9PT_2 
cd ../../Chapter04/filter9PT_3   ;   make "$1" NAME=filter9PT_3 
cd ../../Chapter04/median9PT      ;  make "$1" NAME=median9PT   
cd ../../Chapter04/reduce_maxdiff ;  make "$1" NAME=reduce_maxdiff  
cd ../../Chapter04/sobel6PT      ;   make "$1" NAME=sobel6PT     
cd ../../Chapter04/stencil2D     ;   make "$1" NAME=stencil2D    
cd ../../Chapter04/stencil2D_sm  ;   make "$1" NAME=stencil2D_sm 
cd ../../Chapter04/stencil3D     ;   make "$1" NAME=stencil3D    
cd ../../Chapter04/stencil9PT    ;   make "$1" NAME=stencil9PT   
cd ../../Chapter04/stencil9PT_sm  ;  make "$1" NAME=stencil9PT_sm
cd ../../Chapter05/affine3D      ;   make "$1" NAME=affine3D     
cd ../../Chapter05/register      ;   make "$1" NAME=register    
cd ../../Chapter05/rotate1       ;   make "$1" NAME=rotate1     
cd ../../Chapter05/rotate1B      ;   make "$1" NAME=rotate1B    
cd ../../Chapter05/rotate2       ;   make "$1" NAME=rotate2     
cd ../../Chapter05/rotate3       ;   make "$1" NAME=rotate3     
cd ../../Chapter05/rotate4       ;   make "$1" NAME=rotate4     
cd ../../Chapter05/rotate4cv     ;   make "$1" NAME=rotate4cv   
cd ../../Chapter06/ising         ;   make "$1" NAME=ising       
cd ../../Chapter06/piG           ;   make "$1" NAME=piG         
cd ../../Chapter06/piH           ;   make "$1" NAME=piH         
cd ../../Chapter06/piH2          ;   make "$1" NAME=piH2        
cd ../../Chapter06/piH4          ;   make "$1" NAME=piH4        
cd ../../Chapter06/piH5          ;   make "$1" NAME=piH5        
cd ../../Chapter06/piH6          ;   make "$1" NAME=piH6        
cd ../../Chapter06/piOMP         ;   make "$1" NAME=piOMP       
cd ../../Chapter07/asyncDiskIO   ;   make "$1" NAME=asyncDiskIO 
cd ../../Chapter07/asyncDiskIO_Ex ;  make "$1" NAME=asyncDiskIO_Ex
cd ../../Chapter07/event1        ;   make "$1" NAME=event1    
cd ../../Chapter07/event2        ;   make "$1" NAME=event2    
cd ../../Chapter07/graph         ;   make "$1" NAME=graph     
cd ../../Chapter07/pipeline      ;   make "$1" NAME=pipeline        
cd ../../Chapter08/derenzo       ;   make "$1" NAME=derenzo     
cd ../../Chapter08/fullblk       ;   make "$1" NAME=fullblk     
cd ../../Chapter08/fulldoi      ;   make "$1" NAME=fulldoi     
cd ../../Chapter08/fullsim      ;   make "$1" NAME=fullsim     
cd ../../Chapter08/polmake      ;   make "$1" NAME=polmake     
cd ../../Chapter08/poluse       ;   make "$1" NAME=poluse      
cd ../../Chapter08/readspot     ;   make "$1" NAME=readspot    
cd ../../Chapter08/reco         ;   make "$1" NAME=reco        
cd ../../Chapter08/recosem      ;   make "$1" NAME=recosem     
cd ../../Chapter08/RLdeconv     ;   make "$1" NAME=RLdeconv    
cd ../../Chapter08/smsplit      ;   make "$1" NAME=smsplit     
cd ../../Chapter08/vol2gold     ;   make "$1" NAME=vol2gold    
cd ../../Chapter09/memtests     ;   make "$1" NAME=memtests    
cd ../../Chapter09/memtests2    ;   make "$1" NAME=memtests2   
cd ../../Chapter09/mpi_alltoall ;   make "$1" NAME=mpi_alltoall
cd ../../Chapter09/mpi_reduce   ;   make "$1" NAME=mpi_reduce  
cd ../../Chapter09/multiGPU     ;   make "$1" NAME=multiGPU    
cd ../../Chapter09/p2ptest      ;   make "$1" NAME=p2ptest     
cd ../../Chapter10/gpulog       ;   make "$1" NAME=gpulog      
cd ../../Chapter10/gpulog_nvtx  ;   make "$1" NAME=gpulog_nvtx 
cd ../../Chapter11/matmulT      ;   make "$1" NAME=matmulT     
cd ../../Chapter11/matmulTS     ;   make "$1" NAME=matmulTS    
cd ../../Chapter11/reduceT      ;   make "$1" NAME=reduceT     

