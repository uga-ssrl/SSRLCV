CUDA_INSTALL_PATH := /usr/local/cuda

# CUDA stuff
CXX := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I./include -I/usr/local/cuda/include
LIB :=  -L/usr/local/cuda/lib64 -lcublas -lcuda -lcudart -lcusparse -lcusolver\
        -lpng -Xcompiler -fopenmp -ltiff -ljpeg

# Common flags
COMMONFLAGS += $(INCLUDES)
COMMONFLAGS += -g # Output debug symbols 
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -Wall -std=c++11
# compute_<#> and sm_<#> will need to change depending on the device
# if this is not done you will receive a no kernel image is availabe error
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += -std=c++11

# Gencode arguments
SM ?= 35 37 50 52 60 61 70

ifeq ($(SM),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
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

#
# Files
#

SRCDIR 		= ./src
OBJDIR 		= ./obj
BINDIR 		= ./bin
OUTDIR 		= ./out
TESTDIR 	= ./util/CI

_BASE_OBJS  = io_util.cpp.o
_BASE_OBJS += io_fmt_ply.cu.o
_BASE_OBJS += io_fmt_anatomy.cu.o
_BASE_OBJS += tinyply.cpp.o
_BASE_OBJS += cuda_util.cu.o
_BASE_OBJS += Feature.cu.o
_BASE_OBJS += Image.cu.o
_BASE_OBJS += Quadtree.cu.o
_BASE_OBJS += FeatureFactory.cu.o
_BASE_OBJS += SIFT_FeatureFactory.cu.o
_BASE_OBJS += MatchFactory.cu.o
_BASE_OBJS += matrix_util.cu.o
_BASE_OBJS += PointCloudFactory.cu.o
_BASE_OBJS += Octree.cu.o
_BASE_OBJS += MeshFactory.cu.o
_SFM_OBJS = SFM.cu.o
_SD_OBJS = StereoDisparity.cu.o
_TD_OBJS = TrueDisparity.cu.o
_T_OBJS = Tester.cu.o

BASE_OBJS = $(patsubst %, $(OBJDIR)/%, $(_BASE_OBJS))
SFM_OBJS = $(patsubst %, $(OBJDIR)/%, $(_SFM_OBJS))
SD_OBJS = $(patsubst %, $(OBJDIR)/%, $(_SD_OBJS))
TD_OBJS = $(patsubst %, $(OBJDIR)/%, $(_TD_OBJS))
T_OBJS = $(patsubst %, $(OBJDIR)/%, $(_T_OBJS))

TEST_OBJS = $(patsubst %, $(OBJDIR)/%, $(_BASE_OBJS))

TARGET_SFM = SFM
TARGET_SD = StereoDisparity
TARGET_TD = TrueDisparity
TARGET_T = Tester

## Test sensing
TestsIn_cpp 	= $(wildcard $(TESTDIR)/src/*.cpp)
TestsIn_cu 		= $(wildcard $(TESTDIR)/src/*.cu)
TESTS_CPP 		= $(patsubst $(TESTDIR)/src/%.cpp, $(TESTDIR)/bin/cpp/%, $(TestsIn_cpp))
TESTS_CU 		= $(patsubst $(TESTDIR)/src/%.cu, $(TESTDIR)/bin/cu/%, $(TestsIn_cu))
TESTS 			= $(TESTS_CU) $(TESTS_CPP)

NVCCFLAGS += $(GENCODEFLAGS)
LINKLINE_SFM = $(LINK) $(GENCODEFLAGS) $(BASE_OBJS) $(SFM_OBJS) $(LIB) -o $(BINDIR)/$(TARGET_SFM)
LINKLINE_SD = $(LINK) $(GENCODEFLAGS) $(BASE_OBJS) $(SD_OBJS) $(LIB) -o $(BINDIR)/$(TARGET_SD)
LINKLINE_TD = $(LINK) $(GENCODEFLAGS) $(BASE_OBJS) $(TD_OBJS) $(LIB) -o $(BINDIR)/$(TARGET_TD)
LINKLINE_T = $(LINK) $(GENCODEFLAGS) $(BASE_OBJS) $(T_OBJS) $(LIB) -o $(BINDIR)/$(TARGET_T)

.SUFFIXES: .cpp .cu .o
.PHONY: all clean test

all: base $(BINDIR)/$(TARGET_SFM) $(BINDIR)/$(TARGET_SD) $(BINDIR)/$(TARGET_TD) $(BINDIR)/$(TARGET_T) $(TESTS)

base: $(BASE_OBJS)

sfm: base $(BINDIR)/$(TARGET_SFM)

stereo: base $(BINDIR)/$(TARGET_SD)

disp: base $(BINDIR)/$(TARGET_TD)

misc: base $(BINDIR)/$(TARGET_T)

test: all $(TEST_OBJS)
	cd $(TESTDIR); ./test-all

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

$(OUTDIR):
			-mkdir -p $(OUTDIR)






# Compiling

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -dc $< -o $@

$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

# Linking targets
$(BINDIR)/$(TARGET_SFM): $(BASE_OBJS) $(SFM_OBJS) Makefile
	$(LINKLINE_SFM)

$(BINDIR)/$(TARGET_SD): $(BASE_OBJS) $(SD_OBJS) Makefile
	$(LINKLINE_SD)

$(BINDIR)/$(TARGET_TD): $(BASE_OBJS) $(TD_OBJS) Makefile
	$(LINKLINE_TD)

$(BINDIR)/$(TARGET_T): $(BASE_OBJS) $(T_OBJS) Makefile
	$(LINKLINE_T)

#
# Tests
#

$(TESTDIR)/obj/%.cpp.o: $(TESTDIR)/src/%.cpp $(TESTDIR)/unit-testing.h
	$(CXX) $(INCLUDES) -I./util/CI/ $(CXXFLAGS)  -c -o $@ $<

$(TESTDIR)/obj/%.cu.o: $(TESTDIR)/src/%.cu $(TESTDIR)/unit-testing.h
	$(NVCC) $(INCLUDES) -I./util/CI/ $(NVCCFLAGS) -c -o $@ $<

$(TESTDIR)/bin/cpp/%: $(TESTDIR)/obj/%.cpp.o $(TEST_OBJS)
	$(LINK) $(GENCODEFLAGS) $(LIB) $(TEST_OBJS) $< -o $@

$(TESTDIR)/bin/cu/%: $(TESTDIR)/obj/%.cu.o $(TEST_OBJS)
	$(LINK) $(GENCODEFLAGS) $(LIB) $(TEST_OBJS) $< -o $@

#
# Docs
#
docs:
	doxygen doc/doxygen/Doxyfile

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
	rm -f *.kpa
	rm -f *.txt
	rm -rf $(TESTDIR)/obj/*
	rm -rf $(TESTDIR)/tmp/*
	rm -rf $(TESTDIR)/bin/cu/*
	rm -rf $(TESTDIR)/bin/cpp/*
