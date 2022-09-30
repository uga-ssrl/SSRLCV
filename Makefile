
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

TESTLINKFLAGS += -lgtest -lgtest_main

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
TESTDIR     = ./test
TESTOBJDIR  = ./test_obj
GTEST_DIR   = 

_BASE_OBJS = Logger.cpp.o
_BASE_OBJS += io_util.cpp.o
_BASE_OBJS += io_fmt_ply.cu.o
_BASE_OBJS += io_fmt_anatomy.cu.o
_BASE_OBJS += tinyply.cpp.o
_BASE_OBJS += cuda_vec_util.cu.o
_BASE_OBJS += cuda_util.cu.o
_BASE_OBJS += Feature.cu.o
_BASE_OBJS += Image.cu.o
_BASE_OBJS += Quadtree.cu.o
_BASE_OBJS += FeatureFactory.cu.o
_BASE_OBJS += SIFT_FeatureFactory.cu.o
_BASE_OBJS += MatchFactory.cu.o
_BASE_OBJS += KDTree.cu.o
_BASE_OBJS += matrix_util.cu.o
_BASE_OBJS += Octree.cu.o
_BASE_OBJS += PointCloudFactory.cu.o
_BASE_OBJS += MeshFactory.cu.o
_BASE_OBJS += Pipeline.cu.o
_SFM_OBJS = SFM.cu.o
_TEST_OBJS += Pipeline.cu.o
_T_OBJS += Tester.cu.o

BASE_OBJS = $(patsubst %, $(OBJDIR)/%, $(_BASE_OBJS))
SFM_OBJS = $(patsubst %, $(OBJDIR)/%, $(_SFM_OBJS))
T_OBJS = $(patsubst %, $(OBJDIR)/%, $(_T_OBJS))
TEST_OBJS = $(patsubst %, $(TESTOBJDIR)/%, $(_TEST_OBJS))

TARGET_SFM = SFM
TARGET_T = Tester
TARGET_TEST = test

NVCCFLAGS += $(GENCODEFLAGS)
LINKLINE_SFM = $(LINK) $(GENCODEFLAGS) $(BASE_OBJS) $(SFM_OBJS) $(LIB) -o $(BINDIR)/$(TARGET_SFM)
LINKLINE_T = $(LINK) $(GENCODEFLAGS) $(BASE_OBJS) $(T_OBJS) $(LIB) -o $(BINDIR)/$(TARGET_T)
LINKLINE_TEST = $(LINK) $(GENCODEFLAGS) $(BASE_OBJS) $(TEST_OBJS) $(LIB) $(TESTLINKFLAGS) -o $(BINDIR)/$(TARGET_TEST)

.SUFFIXES: .cpp .cu .o
.PHONY: all clean test

all: base $(BINDIR)/$(TARGET_SFM) $(BINDIR)/$(TARGET_T)

base: $(BASE_OBJS)

sfm: base $(BINDIR)/$(TARGET_SFM)

misc: base $(BINDIR)/$(TARGET_T)

test: base $(BINDIR)/$(TARGET_TEST)

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

$(OUTDIR):
			-mkdir -p $(OUTDIR)

$(TESTOBJDIR):
	    -mkdir -p $(TESTOBJDIR)

# Compiling

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -dc $< -o $@

$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

$(TESTOBJDIR)/%.cu.o: $(TESTDIR)/%.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -dc $< -o $@

# Linking targets
$(BINDIR)/$(TARGET_SFM): $(BASE_OBJS) $(SFM_OBJS) Makefile
	$(LINKLINE_SFM)

$(BINDIR)/$(TARGET_T): $(BASE_OBJS) $(T_OBJS) Makefile
	$(LINKLINE_T)

$(BINDIR)/$(TARGET_TEST): $(BASE_OBJS) $(TEST_OBJS) Makefile
	$(LINKLINE_TEST)

#
# Docs
#
docs:
	doxygen doc/doxygen/Doxyfile

#
# Clean
#

clean:
	rm -f bin/*
	rm -f out/*.log
	rm -f out/tests/*/*.log
	rm -f src/*~
	rm -f util/*~
	rm -f obj/*
	rm -f testobj/*
	rm -f util/*.o
	rm -f util/io/*~
	rm -f util/examples/*~
	rm -f uril/CI/*~
	rm -f .DS_Store
	rm -f *._*
	rm -f *.~
	rm -f *.kpa
	rm -f *.txt
	rm -f data/*.uty
	rm -f src/examples/*.out
	rm -f src/examples/*.uty
	rm -rf doc/doxygen/documentation/*
