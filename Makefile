CUDA_INSTALL_PATH := /usr/local/cuda

CXX := /usr/bin/g++-6
LINK := /usr/bin/g++-6
NVCC  := nvcc

# Includes
INCLUDES = -I. -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_60,code=sm_60 -Iinclude -lcublas
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -g -std=c++11 -Iinclude -lcublas

LIB_CUDA := -L/usr/local/cuda-9.1/lib64 -lcudart -lcublas

SRCDIR = ./src
OBJDIR = ./util
BINDIR = ./bin

_OBJS = reprojection.cpp.o

OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = reprojection.x
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
