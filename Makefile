CUDA_INSTALL_PATH := /usr/local/cuda

# CUDA stuff
CXX := gcc
LINK := nvcc
NVCC  := nvcc

# compilers configuration for sift-cpu
S_CC = gcc
S_OFLAGS = -g -O3
S_LIBS = -L/usr/local/lib -lpng -lm
S_CFLAGS = -Wall -Wno-write-strings  -pedantic -std=c99 -D_POSIX_C_SOURCE=200809L

# Source files with executables.
S_SRC_ALGO = sift_cli
S_SRC_MATCH = match_cli

SRCa = lib_sift.c \
	   lib_sift_anatomy.c \
	   lib_scalespace.c \
	   lib_description.c \
       lib_discrete.c \
	   lib_keypoint.c \
	   lib_util.c

S_SRCb = lib_io_scalespace.c

S_SRCc = lib_matching.c

S_SRCDIR = src
S_OBJDIR = src
S_BINDIR = bin

# Includes
INCLUDES = -I. -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_62,code=sm_62 -Iinclude -lcublas
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -G -g -std=c++11 -Iinclude -lcublas

LIB_CUDA := -L/usr/local/cuda-9.0/lib64 -lcudart -lcublas

SRCDIR = ./src
OBJDIR = ./util
BINDIR = ./bin

_OBJS = 2viewreprojection.cu.o

OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

# for sift-cpu
S_OBJa = $(addprefix $(OBJDIR)/,$(SRCa:.c=.o))
S_OBJb = $(addprefix $(OBJDIR)/,$(SRCb:.c=.o))
S_OBJc = $(addprefix $(OBJDIR)/,$(SRCc:.c=.o))

S_OBJ = $(OBJa) $(OBJb) $(OBJc)

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
	rm -f util/io/*~
	rm -f util/examples/*~
	rm -f uril/CI/*~
	rm -f .DS_Store
	rm -f *._*
