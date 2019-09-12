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
OUTDIR = ./out


_OBJS = io_util.cpp.o
_OBJS += io_file.cu.o
_OBJS += tinyply.cpp.o
_OBJS += cuda_util.cu.o
_OBJS += Feature.cu.o
_OBJS += Image.cu.o
_OBJS += Quadtree.cu.o
_OBJS += FeatureFactory.cu.o
_OBJS += SIFT_FeatureFactory.cu.o
_OBJS += MatchFactory.cu.o
_OBJS += matrix_util.cu.o
_OBJS += PointCloudFactory.cu.o
_OBJS += Octree.cu.o
_OBJS += MeshFactory.cu.o
_OBJS += SFM.cu.o
_SD_OBJS = io_util.cpp.o
_SD_OBJS += io_file.cu.o
_SD_OBJS += tinyply.cpp.o
_SD_OBJS += cuda_util.cu.o
_SD_OBJS += Feature.cu.o
_SD_OBJS += Image.cu.o
_SD_OBJS += Quadtree.cu.o
_SD_OBJS += FeatureFactory.cu.o
_SD_OBJS += SIFT_FeatureFactory.cu.o
_SD_OBJS += MatchFactory.cu.o
_SD_OBJS += matrix_util.cu.o
_SD_OBJS += PointCloudFactory.cu.o
_SD_OBJS += Octree.cu.o
_SD_OBJS += MeshFactory.cu.o
_SD_OBJS += StereoDisparity.cu.o
_T_OBJS = io_util.cpp.o
_T_OBJS += io_file.cu.o
_T_OBJS += tinyply.cpp.o
_T_OBJS += cuda_util.cu.o
_T_OBJS += Feature.cu.o
_T_OBJS += Image.cu.o
_T_OBJS += Quadtree.cu.o
_T_OBJS += FeatureFactory.cu.o
_T_OBJS += SIFT_FeatureFactory.cu.o
_T_OBJS += MatchFactory.cu.o
_T_OBJS += matrix_util.cu.o
_T_OBJS += PointCloudFactory.cu.o
_T_OBJS += Octree.cu.o
_T_OBJS += MeshFactory.cu.o
_T_OBJS += Tester.cu.o

OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}
SD_OBJS = ${patsubst %, ${OBJDIR}/%, ${_SD_OBJS}}
T_OBJS = ${patsubst %, ${OBJDIR}/%, ${_T_OBJS}}

TARGET = SFM
TARGET_SD = StereoDisparity
TARGET_T = Tester

LINKLINE = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${OBJS} ${LIB} -o ${BINDIR}/${TARGET}
LINKLINE_SD = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${SD_OBJS} ${LIB} -o ${BINDIR}/${TARGET_SD}
LINKLINE_T = ${LINK} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 ${T_OBJS} ${LIB} -o ${BINDIR}/${TARGET_T}

## Test sensing
TestsIn_cpp 	= $(wildcard tests/src/*.cpp)
TestsIn_cu 		= $(wildcard tests/src/*.cu)
TESTS_CPP 		= $(patsubst tests/src/%.cpp, tests/bin/cpp/%, $(TestsIn_cpp))
TESTS_CU 			= $(patsubst tests/src/%.cu, tests/bin/cu/%, $(TestsIn_cu))
TESTS 				= ${TESTS_CU} ${TESTS_CPP}

.SUFFIXES: .cpp .cu .o
.PHONY: all clean test

all: ${BINDIR}/${TARGET} ${BINDIR}/${TARGET_SD} ${BINDIR}/${TARGET_T} ${TESTS}

test: all
	./test-all

all: ${BINDIR}/${TARGET} ${BINDIR}/${TARGET_SD} ${BINDIR}/${TARGET_T}

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

${BINDIR}/%: ${OBJS} ${SD_OBJS} ${T_OBJS} Makefile
	${LINKLINE}
	${LINKLINE_SD}
	${LINKLINE_T}


#
# Tests
#

TEST_DIRS = mkdir -p tests/tmp; mkdir -p tests/obj; mkdir -p tests/bin/cu; mkdir -p tests/bin/cpp

tests/obj/%.cpp.o: tests/src/%.cpp
	@${TEST_DIRS}
	${CXX} ${INCLUDES} ${CXXFLAGS}  -c -o $@ $<

tests/obj/%.cu.o: tests/src/%.cu
	@${TEST_DIRS}
	${NVCC} ${INCLUDES} ${NVCCFLAGS} -c -o $@ $<

tests/bin/cpp/%: tests/obj/%.cpp.o ${OBJS}
	@${TEST_DIRS}
	${LINK} ${LINKFLAGS} ${LIB} ${OBJS} $< -o $@

tests/bin/cu/%: tests/obj/%.cu.o ${OBJS}
	@${TEST_DIRS}
	${LINK} ${LINKFLAGS} ${LIB} ${OBJS} $< -o $@






#
# Clean
#

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
	rm -rf tests/obj
	rm -rf tests/tmp
	rm -rf tests/bin 
