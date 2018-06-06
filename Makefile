CUDA_INSTALL_PATH := /usr/local/cuda

# CUDA stuff
CXX := gcc
LINK := nvcc
NVCC  := nvcc

# compilers configuration for sift-cpu
CC = gcc
OFLAGS = -g -O3
LIBS = -L/usr/local/lib -lpng -lm
CFLAGS = -Wall -Wno-write-strings  -pedantic -std=c99 -D_POSIX_C_SOURCE=200809L

# Source files with executables.
SRC_ALGO = sift_cli
SRC_MATCH = match_cli

S_SRC=sift-cpu

SRCa = lib_sift.c \
	   lib_sift_anatomy.c \
	   lib_scalespace.c \
	   lib_description.c \
       lib_discrete.c \
	   lib_keypoint.c \
	   lib_util.c

SRCb = lib_io_scalespace.c

SRCc = lib_matching.c

# Includes
INCLUDES = -I. -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -std=c++11 -Iinclude -lcublas
# compute_<#> and sm_<#> will need to change depending on the device
# if this is not done you will receive a no kernel image is availabe error
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_61,code=sm_61 -Iinclude -lcublas -lthrust

LIB_CUDA :=  -gencode=arch=compute_61,code=sm_61 -L/usr/local/cuda-9.1/lib64 -lcudart -lcublas

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS1 = 2viewreprojection.cu.o
_OBJS2 = octree.cu.o
_OBJS2 += poisson.cu.o
_OBJS2 += reconstruction.cu.o


OBJS1 = ${patsubst %, ${OBJDIR}/%, ${_OBJS1}}
OBJS2 = ${patsubst %, ${OBJDIR}/%, ${_OBJS2}}
# for sift-cpu
S_OBJa = $(addprefix $(OBJDIR)/,$(SRCa:.c=.o))
S_OBJb = $(addprefix $(OBJDIR)/,$(SRCb:.c=.o))
S_OBJc = $(addprefix $(OBJDIR)/,$(SRCc:.c=.o))
S_OBJ = $(OBJa) $(OBJb) $(OBJc)

TARGET1 = 2viewreprojection.exe
LINKLINE1 = ${LINK} -o ${BINDIR}/${TARGET1} ${OBJS1} ${LIB_CUDA}

TARGET2 = reconstruction.exe
LINKLINE2 = ${LINK} -o ${BINDIR}/${TARGET2} ${OBJS2} ${LIB_CUDA}

sift= $(BIN)
match= $(BINMATCH)
demo= $(BINDEMO)


.SUFFIXES: .cpp .cu .o

all: $(OBJDIR) $(BINDIR) $(sift) $(match) $(demo) ${BINDIR}/${TARGET1} ${BINDIR}/${TARGET2}

#---------------------------------------------------------------
#  SIFT CLI
#

$(BIN) : $(BINDIR)/% : $(SRCDIR)/$(S_SRC)/%.c $(OBJDIR)/lib_sift_anatomy.o  $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_scalespace.o $(OBJDIR)/lib_description.o   $(OBJDIR)/lib_discrete.o  $(OBJDIR)/lib_io_scalespace.o  $(OBJDIR)/lib_util.o   $(OBJDIR)/io_png.o $(OBJDIR)/lib_sift.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

#---------------------------------------------------------------
#  LIB_SIFT
#
$(OBJDIR)/lib_sift.o : $(SRCDIR)/$(S_SRC)/lib_sift.c $(OBJDIR)/lib_sift_anatomy.o $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_scalespace.o : $(SRCDIR)/$(S_SRC)/lib_scalespace.c $(OBJDIR)/lib_discrete.o  $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_discrete.o : $(SRCDIR)/$(S_SRC)/lib_discrete.c $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_description.o : $(SRCDIR)/$(S_SRC)/lib_description.c $(OBJDIR)/lib_discrete.o $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_keypoint.o : $(SRCDIR)/$(S_SRC)/lib_keypoint.c $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_sift_anatomy.o : $(SRCDIR)/$(S_SRC)/lib_sift_anatomy.c $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_discrete.o $(OBJDIR)/lib_scalespace.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_util.o : $(SRCDIR)/$(S_SRC)/lib_util.c
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

#--------------------------------------------------------------
#   IN (image) and OUT (scalespace)
#
$(OBJDIR)/io_png.o : $(SRCDIR)/$(S_SRC)/io_png.c
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_io_scalespace.o : $(SRCDIR)/$(S_SRC)/lib_io_scalespace.c $(OBJDIR)/io_png.o  $(OBJDIR)/lib_scalespace.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<


#-------------------------------------------------------------
#   Matching algorithm
#
$(OBJDIR)/lib_matching.o : $(SRCDIR)/$(S_SRC)/lib_matching.c $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(BINMATCH) : $(SRCDIR)/$(S_SRC)/match_cli.c $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_matching.o   $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) -lm

#-------------------------------------------------------------
#  Tools used in the demo
#
$(BINDEMO) : $(BINDIR)/% :	 $(SRCDIR)/demo_extract_patch.c  $(OBJDIR)/lib_discrete.o $(OBJDIR)/io_png.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#-------------------------------------------------------------
#  Cuda Reprojection and Cuda Reconstruction
#
${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} -c $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

${BINDIR}/${TARGET1}: ${OBJS1} Makefile
	${LINKLINE1}

${BINDIR}/${TARGET2}: ${OBJS2} Makefile
	${LINKLINE2}

clean:
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
