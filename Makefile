CUDA_INSTALL_PATH := /usr/local/cuda

# CUDA stuff
CXX := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I./include -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -std=c++11
# compute_<#> and sm_<#> will need to change depending on the device
# if this is not done you will receive a no kernel image is availabe error
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_61,code=sm_61

LIB :=  -L/usr/local/cuda/lib64 -lcublas -lcuda -lcudart -lcusparse -lcusolver\
        -lpng -Xcompiler -fopenmp

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin


_OBJS = io_util.cpp.o
_OBJS += tinyply.cpp.o
_OBJS += cuda_util.cu.o
_OBJS += Feature.cu.o
_OBJS += Image.cu.o
_OBJS += Quadtree.cu.o
_OBJS += FeatureFactory.cu.o
_OBJS += MatchFactory.cu.o
_OBJS += PointCloudFactory.cu.o
_OBJS += reprojection.cu.o
_OBJS += Octree.cu.o
_OBJS += MeshFactory.cu.o
_OBJS += SFM.cu.o

OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = SFM

LINKLINE = ${LINK} -gencode=arch=compute_61,code=sm_61 ${OBJS} ${LIB} -o ${BINDIR}/${TARGET}


.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET}

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

#-------------------------------------------------------------
#  Cuda Cuda Reconstruction
#
${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${INCLUDES} ${NVCCFLAGS} -dc $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${INCLUDES} ${CXXFLAGS} -c $< -o $@

${BINDIR}/${TARGET}: ${OBJS} Makefile
	${LINKLINE}

clean:
	rm -f out/*.ply
	rm -f bin/*
	rm -f out/*
	rm -f src/*~
	rm -f util/*~
	rm -f obj/*
	rm -f util/*.o
	rm -f util/io/*~
	rm -f util/examples/*~
	rm -f uril/CI/*~
	rm -f .DS_Store
	rm -f *._*
	rm -f *.~
	rm -f *.kp
	rm -f *.txt
