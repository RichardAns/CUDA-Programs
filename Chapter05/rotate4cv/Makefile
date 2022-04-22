################################################################################
# Simple MakeFile for examples for the book Programming in Parallel with CUDA
# Note customised files for each example are not provided
# please use "make build NAME=gpusum" etc. to provide program names as necessary
#
# modified for rotate4cv  (-lopencv_world added to link step) 
################################################################################

# default program name. Note use command line option NAME=... as neceessary
NAME := defaultname

# change these to match your installation
CUDA_PATH := /usr/local/cuda
# assumes CUDA samles installed in ~/samples
NVCUDASAMPLES_ROOT := ~/samples/cuda-samples/Common
# assumes examples installed in directory ~/CudaBook 
CX_ROOT := ~/CudaBook/CUDA-Programs/include

# this for opencv
OPENCV_INCLUDE := /usr/local/include
OPENCV_LIB := /usr/local/lib


# Target rules
all: build

build: $(NAME)

$(NAME).o: $(NAME).cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -I $(NVCUDASAMPLES_ROOT)/ -I $(CX_ROOT)/ -I $(OPENCV_INCLUDE)/ -m64  --threads 0 --use_fast_math -gencode arch=compute_86,code=sm_86 -o $(NAME).o -c $(NAME).cu
    
$(NAME): $(NAME).o   
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_86,code=sm_86 -lopencv_world -o $(NAME) $(NAME).o
	cp $(NAME) ../../Linux/

run: build
	$(EXEC) ./$(NAME)

testrun: build

clean:
	rm -f    $(NAME) $(NAME).o
	rm -rf ../../Linux/$(NAME)
echo:
	$(info NAME is $(NAME))
#
