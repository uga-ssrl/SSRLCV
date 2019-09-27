CUDA_INSTALL_PATH := /usr/local/cuda

# CUDA stuff
CXX := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I./include -I/usr/local/cuda/include
LIB :=  -L/usr/local/cuda/lib64 -lcublas -lcuda -lcudart -lcusparse -lcusolver\
        -lpng -Xcompiler -fopenmp

# Common flags
COMMONFLAGS += ${INCLUDES}
COMMONFLAGS += -g # Output debug symbols 
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -std=c++11
# compute_<#> and sm_<#> will need to change depending on the device
# if this is not done you will receive a no kernel image is availabe error
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11

SMDETECTOR_EXISTS := $(shell ./util/detect-compute-capability 2> /dev/null)
ifndef SMDETECTOR_EXISTS
SMDETECTOR : ./util/detect-compute-capability.cu
	${NVCC} ${INCLUDES} \
	-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_62,code=sm_62 -o ./util/detect-compute-capability ./util/detect-compute-capability.cu
endif 

COMPUTE = $(shell ./util/detect-compute-capability)

GENCODEFLAGS = -gencode arch=compute_$(COMPUTE),code=compute_$(COMPUTE)


#
# Files 
#

SRCDIR 		= ./src
OBJDIR 		= ./obj
BINDIR 		= ./bin
OUTDIR 		= ./out
TESTDIR 	= ./util/CI


BASE_OBJS  = io_util.cpp.o
BASE_OBJS += io_3d.cu.o
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
_SS_OBJS = ${BASE_OBJS}
_SS_OBJS += ScaleSpaceTest.cu.o

SFM_OBJS = ${patsubst %, ${OBJDIR}/%, ${_SFM_OBJS}}
SD_OBJS = ${patsubst %, ${OBJDIR}/%, ${_SD_OBJS}}
T_OBJS = ${patsubst %, ${OBJDIR}/%, ${_T_OBJS}}
SS_OBJS = ${patsubst %, ${OBJDIR}/%, ${_SS_OBJS}}

TEST_OBJS = ${patsubst %, ${OBJDIR}/%, ${BASE_OBJS}}

TARGET_SFM = SFM
TARGET_SD = StereoDisparity
TARGET_T = Tester
TARGET_SS = ScaleSpaceTest

## Test sensing
TestsIn_cpp 	= $(wildcard ${TESTDIR}/src/*.cpp)
TestsIn_cu 		= $(wildcard ${TESTDIR}/src/*.cu)
TESTS_CPP 		= $(patsubst ${TESTDIR}/src/%.cpp, ${TESTDIR}/bin/cpp/%, $(TestsIn_cpp))
TESTS_CU 			= $(patsubst ${TESTDIR}/src/%.cu, ${TESTDIR}/bin/cu/%, $(TestsIn_cu))
TESTS 				= ${TESTS_CU} ${TESTS_CPP}

NVCCFLAGS += ${GENCODEFLAGS}
LINKLINE_SFM = ${LINK} ${GENCODEFLAGS} ${SFM_OBJS} ${LIB} -o ${BINDIR}/${TARGET_SFM}
LINKLINE_SD = ${LINK} ${GENCODEFLAGS} ${SD_OBJS} ${LIB} -o ${BINDIR}/${TARGET_SD}
LINKLINE_T = ${LINK} ${GENCODEFLAGS} ${T_OBJS} ${LIB} -o ${BINDIR}/${TARGET_T}
LINKLINE_SS = ${LINK} ${GENCODEFLAGS} ${SS_OBJS} ${LIB} -o ${BINDIR}/${TARGET_SS}

.SUFFIXES: .cpp .cu .o
.PHONY: all clean test

all: ${BINDIR}/${TARGET_SFM} ${BINDIR}/${TARGET_SD} ${BINDIR}/${TARGET_T} ${BINDIR}/${TARGET_SS} ${TESTS}

test: all ${TEST_OBJS}
	${TESTDIR}/test-all

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

$(OUTDIR):
			-mkdir -p $(OUTDIR)

#-------------------------------------------------------------
#  Cuda Cuda Reconstruction
#

# Compiling
${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${INCLUDES} ${NVCCFLAGS} -dc $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${INCLUDES} ${CXXFLAGS} -c $< -o $@

# Linking targets
${BINDIR}/${TARGET_SFM}: ${SFM_OBJS} Makefile
	${LINKLINE_SFM}

${BINDIR}/${TARGET_SD}: ${SD_OBJS} Makefile
	${LINKLINE_SD}

${BINDIR}/${TARGET_T}: ${T_OBJS} Makefile
	${LINKLINE_T}

${BINDIR}/${TARGET_SS}: ${SS_OBJS} Makefile
	${LINKLINE_SS}


#
# Tests
#

${TESTDIR}/obj/%.cpp.o: ${TESTDIR}/src/%.cpp
	${CXX} ${INCLUDES} -I./util/CI/ ${CXXFLAGS}  -c -o $@ $<

${TESTDIR}/obj/%.cu.o: ${TESTDIR}/src/%.cu
	${NVCC} ${INCLUDES} -I./util/CI/ ${NVCCFLAGS} -c -o $@ $<

${TESTDIR}/bin/cpp/%: ${TESTDIR}/obj/%.cpp.o ${TEST_OBJS}
	${LINK} ${GENCODEFLAGS} ${LIB} ${TEST_OBJS} $< -o $@

${TESTDIR}/bin/cu/%: ${TESTDIR}/obj/%.cu.o ${TEST_OBJS}
	${LINK} ${GENCODEFLAGS} ${LIB} ${TEST_OBJS} $< -o $@



#
# Clean
#

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
	rm -rf ${TESTDIR}/obj/*
	rm -rf ${TESTDIR}/tmp/*
	rm -rf ${TESTDIR}/bin/cu/*
	rm -rf ${TESTDIR}/bin/cpp/*
