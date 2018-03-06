CUDA_INSTALL_PATH := /usr/local/cuda

CXX := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_60,code=sm_60 -Iinclude -lcublas -lthrust
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -g -std=c++11 -Iinclude -lcublas

LIB_CUDA := -gencode=arch=compute_60,code=sm_60 -L/usr/local/cuda-9.1/lib64 -lcudart -lcublas

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS1 = reprojection.cu.o
_OBJS2 = octree.cu.o
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
	rm -f *.ply
	rm -f src/*~
	rm -f util/*~
	rm -f obj/*.o
