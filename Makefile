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
        -L/opt/openblas/lib -lopenblas -lpng -Xcompiler -fopenmp

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin


_OBJS = image_io.cpp.o
_OBJS += cuda_util.cu.o
_OBJS += Feature.cu.o
_OBJS += Image.cu.o
_OBJS += FeatureFactory.cu.o
_OBJS += MatchFactory.cu.o
_OBJS += reprojection.cu.o
_OBJS += octree.cu.o surface.cu.o
_OBJS += SFM.cu.o
_DSIFT_OBJS = image_io.cpp.o cuda_util.cu.o Feature.cu.o Image.cu.o\
FeatureFactory.cu.o MatchFactory.cu.o DSIFTFeatureMatching.cu.o
_REPRO_OBJS = cuda_util.cu.o reprojection.cu.o 2ViewReprojection.cu.o
_RECON_OBJS = cuda_util.cu.o octree.cu.o surface.cu.o Reconstruction.cu.o


DSIFT_OBJS = ${patsubst %, ${OBJDIR}/%, ${_DSIFT_OBJS}}
REPRO_OBJS = ${patsubst %, ${OBJDIR}/%, ${_REPRO_OBJS}}
RECON_OBJS = ${patsubst %, ${OBJDIR}/%, ${_RECON_OBJS}}
OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}


TARGET_DSIFT = DSIFT_Matching
TARGET_REPRO = Reprojection
TARGET_RECON = SurfaceReconstruction
TARGET = SFM

LINKLINE_DSIFT = ${LINK} -gencode=arch=compute_61,code=sm_61 ${DSIFT_OBJS} ${LIB} -o ${BINDIR}/${TARGET_DSIFT}
LINKLINE_REPRO = ${LINK} -gencode=arch=compute_61,code=sm_61 ${REPRO_OBJS} ${LIB} -o ${BINDIR}/${TARGET_REPRO}
LINKLINE_RECON = ${LINK} -gencode=arch=compute_61,code=sm_61 ${RECON_OBJS} ${LIB} -o ${BINDIR}/${TARGET_RECON}
LINKLINE_SFM = ${LINK} -gencode=arch=compute_61,code=sm_61 ${OBJS} ${LIB} -o ${BINDIR}/${TARGET}


.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET_DSIFT}  ${BINDIR}/${TARGET_REPRO}  ${BINDIR}/${TARGET_RECON}  ${BINDIR}/${TARGET}

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

${BINDIR}/%: ${DSIFT_OBJS} ${REPRO_OBJS} ${RECON_OBJS} ${OBJS} Makefile
	${LINKLINE_DSIFT}
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
