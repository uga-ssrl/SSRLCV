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
NVCCFLAGS += -std=c++11 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61

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
_OBJS += PointCloudFactory.cu.o 
_DSIFT_OBJS = image_io.cpp.o cuda_util.cu.o Feature.cu.o Image.cu.o\
FeatureFactory.cu.o MatchFactory.cu.o DSIFTFeatureMatching.cu.o
_DPIPE_OBJS = image_io.cpp.o cuda_util.cu.o Feature.cu.o Image.cu.o\
FeatureFactory.cu.o MatchFactory.cu.o DSIFTPipeline.cu.o
_REPRO_OBJS = cuda_util.cu.o reprojection.cu.o 2ViewReprojection.cu.o
_RECON_OBJS = tinyply.cpp.o cuda_util.cu.o Octree.cu.o surface.cu.o Reconstruction.cu.o


DSIFT_OBJS = ${patsubst %, ${OBJDIR}/%, ${_DSIFT_OBJS}}
DPIPE_OBJS = ${patsubst %, ${OBJDIR}/%, ${_DPIPE_OBJS}}
REPRO_OBJS = ${patsubst %, ${OBJDIR}/%, ${_REPRO_OBJS}}
RECON_OBJS = ${patsubst %, ${OBJDIR}/%, ${_RECON_OBJS}}
OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}


TARGET_DSIFT = DSIFT_Matching
TARGET_DPIPE = DPIPE_example
TARGET_REPRO = Reprojection
TARGET_RECON = SurfaceReconstruction
TARGET = SFM

LINKLINE_DSIFT = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${DSIFT_OBJS} ${LIB} -o ${BINDIR}/${TARGET_DSIFT}
LINKLINE_DPIPE = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${DPIPE_OBJS} ${LIB} -o ${BINDIR}/${TARGET_DPIPE}
LINKLINE_REPRO = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${REPRO_OBJS} ${LIB} -o ${BINDIR}/${TARGET_REPRO}
LINKLINE_RECON = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${RECON_OBJS} ${LIB} -o ${BINDIR}/${TARGET_RECON}
LINKLINE_SFM = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${OBJS} ${LIB} -o ${BINDIR}/${TARGET}


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

${BINDIR}/%: ${DSIFT_OBJS} ${DPIPE_OBJS} ${REPRO_OBJS} ${RECON_OBJS} ${OBJS} Makefile
	${LINKLINE_DSIFT}
	${LINKLINE_DPIPE}
	${LINKLINE_REPRO}
	${LINKLINE_RECON}
	${LINKLINE_SFM}

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
