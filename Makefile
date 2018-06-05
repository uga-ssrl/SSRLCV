CUDA_INSTALL_PATH := /usr/local/cuda

CXX := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -std=c++11 -Iinclude -lcublas
# compute_<#> and sm_<#> will need to change depending on the device
# if this is not done you will receive a no kernel image is availabe error
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_61,code=sm_61 -Iinclude -lcublas -lthrust -lcusolve

LIB_CUDA :=  -gencode=arch=compute_61,code=sm_61 -L/usr/local/cuda-9.1/lib64 -lcudart -lcublas

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS1 = reprojection.cu.o
_OBJS2 = octree.cu.o
_OBJS2 += poisson.cu.o
_OBJS2 += reconstruction.cu.o


OBJS1 = ${patsubst %, ${OBJDIR}/%, ${_OBJS1}}
OBJS2 = ${patsubst %, ${OBJDIR}/%, ${_OBJS2}}

TARGET1 = reprojection.exe
TARGET2 = reconstruction.exe
LINKLINE1 = ${LINK} -o ${BINDIR}/${TARGET1} ${OBJS1} ${LIB_CUDA}
LINKLINE2 = ${LINK} -o ${BINDIR}/${TARGET2} ${OBJS2} ${LIB_CUDA}

.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET1} ${BINDIR}/${TARGET2}

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} -c $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

${BINDIR}/${TARGET1}: ${OBJS1} Makefile
	${LINKLINE1}

${BINDIR}/${TARGET2}: ${OBJS2} Makefile
	${LINKLINE2}

clean:
	rm -f bin/*
	rm -f out/*
	rm -f src/*~
	rm -f util/*~
	rm -f obj/*
