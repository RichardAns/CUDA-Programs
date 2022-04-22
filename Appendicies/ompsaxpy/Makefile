################################################################################
# Simple MakeFile for examples for the book Programming in Parallel with CUDA
# Note customised files for each example are not provided
# please use "make build NAME=gpusum" etc. to provide program names as necessary
#
# This Makefile is for Intel icc complier on Linux
# I found this link:
#            https://estuarine.jp/2021/03/install-oneapi/?lang=en
# 
# helpful for installation of Intel toolkit.
##
# please run 
#           source /opt/intel/oneapi/setvars.sh
#
# to set paths etc.
################################################################################

# default program name. Note use command line option NAME=... as neceessary
NAME := defaultname

# change these to match your installation
CUDA_PATH := /usr/local/cuda
# assumes CUDA samles installed in ~/samples
NVCUDASAMPLES_ROOT := ~/samples/cuda-samples/Common
# assumes examples installed in directory ~/CudaBook 
CX_ROOT := ~/CudaBook/CUDA-Programs/include


# Target rules
all: build

build: $(NAME)

$(NAME): $(NAME).o 
	icc -o $(NAME)  $(NAME).o -qopenmp
	cp $(NAME) ../../Linux/
	
$(NAME).o: $(NAME).cpp
	icc -c -O2 -I $(CX_ROOT) -I $(NVCUDASAMPLES_ROOT) -qopenmp $(NAME).cpp
    
run: build
	$(EXEC) ./$(NAME)

testrun: build

clean:
	rm -f    $(NAME) $(NAME).o
	rm -rf ../../Linux/$(NAME)
echo:
	$(info NAME is $(NAME))
#
