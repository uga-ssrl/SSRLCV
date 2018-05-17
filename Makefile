CUDA_INSTALL_PATH := /usr/local/cuda

CXX := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_62,code=sm_62 -Iinclude -lcublas
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -g -std=c++11 -Iinclude -lcublas

LIB_CUDA := -L/usr/local/cuda-9.0/lib64 -lcudart -lcublas

SRCDIR = ./src
OBJDIR = ./util
BINDIR = ./bin

_OBJS = 2viewreprojection.cu.o

OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = 2viewreprojection.x
LINKLINE = ${LINK} -o ${BINDIR}/${TARGET} ${OBJS} ${LIB_CUDA}


.SUFFIXES: .cpp .cu .o

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} -c $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

${BINDIR}/${TARGET}: ${OBJS} Makefile
	${LINKLINE}

clean:
	rm -f bin/*
	rm -f *.ply
	rm -f src/*~
	rm -f util/*~
	rm -f util/*.o
