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
NVCCFLAGS += -std=c++11

# Gencode arguments
SMS ?= 35 37 50 52 60 61 70

ifeq ($(SM),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODEFLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SM),$(eval GENCODEFLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SM)))
ifneq ($(HIGHEST_SM),)
GENCODEFLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

NVCCFLAGS += ${GENCODEFLAGS}

LIB :=  -L/usr/local/cuda/lib64 -lcublas -lcuda -lcudart -lcusparse -lcusolver\
        -lpng -Xcompiler -fopenmp

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin
OUTDIR = ./out


BASE_OBJS = io_util.cpp.o
BASE_OBJS += tinyply.cpp.o
BASE_OBJS += cuda_util.cu.o
BASE_OBJS += Feature.cu.o
BASE_OBJS += Image.cu.o
BASE_OBJS += Quadtree.cu.o
BASE_OBJS += FeatureFactory.cu.o
BASE_OBJS += SIFT_FeatureFactory.cu.o
BASE_OBJS += MatchFactory.cu.o
BASE_OBJS += matrix_util.cu.o
BASE_OBJS += PointCloudFactory.cu.o
BASE_OBJS += Octree.cu.o
BASE_OBJS += MeshFactory.cu.o

_SFM_OBJS = ${BASE_OBJS}
_SFM_OBJS += SFM.cu.o
_SD_OBJS = ${BASE_OBJS}
_SD_OBJS += StereoDisparity.cu.o
_T_OBJS = ${BASE_OBJS}
_T_OBJS += Tester.cu.o

SFM_OBJS = ${patsubst %, ${OBJDIR}/%, ${_SFM_OBJS}}
SD_OBJS = ${patsubst %, ${OBJDIR}/%, ${_SD_OBJS}}
T_OBJS = ${patsubst %, ${OBJDIR}/%, ${_T_OBJS}}

TARGET_SFM = SFM
TARGET_SD = StereoDisparity
TARGET_T = Tester

LINKLINE = ${LINK} ${GENCODEFLAGS} ${SFM_OBJS} ${LIB} -o ${BINDIR}/${TARGET_SFM}
LINKLINE_SD = ${LINK} ${GENCODEFLAGS} ${SD_OBJS} ${LIB} -o ${BINDIR}/${TARGET_SD}
LINKLINE_T = ${LINK} ${GENCODEFLAGS} ${T_OBJS} ${LIB} -o ${BINDIR}/${TARGET_T}

.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET_SFM} ${BINDIR}/${TARGET_SD} ${BINDIR}/${TARGET_T}

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

$(OUTDIR):
			-mkdir -p $(OUTDIR)

#-------------------------------------------------------------
#  Cuda Cuda Reconstruction
#
${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${INCLUDES} ${NVCCFLAGS} -dc $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${INCLUDES} ${CXXFLAGS} -c $< -o $@

${BINDIR}/%: ${SFM_OBJS} ${SD_OBJS} ${T_OBJS} Makefile
	${LINKLINE}
	${LINKLINE_SD}
	${LINKLINE_T}

clean:
	rm -f out/*
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
