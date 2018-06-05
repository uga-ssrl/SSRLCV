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

S_SRC=sift-cpu

SRCa = $(S_SRC)/lib_sift.c \
	   $(S_SRC)/lib_sift_anatomy.c \
	   $(S_SRC)/lib_scalespace.c \
	   $(S_SRC)/lib_description.c \
       $(S_SRC)/lib_discrete.c \
	   $(S_SRC)/lib_keypoint.c \
	   $(S_SRC)/lib_util.c

SRCb = $(S_SRC)/lib_io_scalespace.c

SRCc = $(S_SRC)/lib_matching.c

# S_SRCDIR = src
# S_OBJDIR = src
# S_BINDIR = bin

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

#sift cpu stuff
BIN = $(addprefix $(BINDIR)/,$(SRC_ALGO))
BINMATCH = $(addprefix $(BINDIR)/,$(SRC_MATCH))
BINDEMO = $(addprefix $(BINDIR)/,$(SRC_DEMO))

sift= $(BIN)
match= $(BINMATCH)
demo= $(BINDEMO)
default: $(OBJDIR) $(BINDIR) $(sift) $(match) $(demo)

#---------------------------------------------------------------
#  SIFT CLI
#

$(BIN) : $(BINDIR)/% : $(SRCDIR)/%.c $(OBJDIR)/lib_sift_anatomy.o  $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_scalespace.o $(OBJDIR)/lib_description.o   $(OBJDIR)/lib_discrete.o  $(OBJDIR)/lib_io_scalespace.o  $(OBJDIR)/lib_util.o   $(OBJDIR)/io_png.o $(OBJDIR)/lib_sift.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

#---------------------------------------------------------------
#  LIB_SIFT
#
$(OBJDIR)/lib_sift.o : $(SRCDIR)/lib_sift.c $(OBJDIR)/lib_sift_anatomy.o $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_scalespace.o : $(SRCDIR)/lib_scalespace.c $(OBJDIR)/lib_discrete.o  $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_discrete.o : $(SRCDIR)/lib_discrete.c $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_description.o : $(SRCDIR)/lib_description.c $(OBJDIR)/lib_discrete.o $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_keypoint.o : $(SRCDIR)/lib_keypoint.c $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_sift_anatomy.o : $(SRCDIR)/lib_sift_anatomy.c $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_discrete.o $(OBJDIR)/lib_scalespace.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_util.o : $(SRCDIR)/lib_util.c
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

#--------------------------------------------------------------
#   IN (image) and OUT (scalespace)
#
$(OBJDIR)/io_png.o : $(SRCDIR)/io_png.c
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(OBJDIR)/lib_io_scalespace.o : $(SRCDIR)/lib_io_scalespace.c $(OBJDIR)/io_png.o  $(OBJDIR)/lib_scalespace.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<


#-------------------------------------------------------------
#   Matching algorithm
#
$(OBJDIR)/lib_matching.o : $(SRCDIR)/lib_matching.c $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) $(OFLAGS) -c -o $@ $<

$(BINMATCH) : $(SRCDIR)/match_cli.c $(OBJDIR)/lib_keypoint.o $(OBJDIR)/lib_matching.o   $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) -lm

#-------------------------------------------------------------
#  Tools used in the demo
#
$(BINDEMO) : $(BINDIR)/% :	 $(SRCDIR)/demo_extract_patch.c  $(OBJDIR)/lib_discrete.o $(OBJDIR)/io_png.o $(OBJDIR)/lib_util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)



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
