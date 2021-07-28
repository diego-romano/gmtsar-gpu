#	Standard Makefile Macro Setup for GMTSAR
#   changes for gmtsar-gpu by Diego Romano 2021-07-27
#
# Edit this file only, NOT the makefile itself.
#-------------------------------------------------------------------------------
# The purpose of this section is to contain common make macros
# that should be processed by every execution of that utility.
#-------------------------------------------------------------------------------


# Compilers, if not set in environment
CC		= gcc 
NVCC	= nvcc
CPP		= gcc -E

CPPFLAGS	= $(INCLUDES) $(DEFINES) 
CUDA_INC	= -I/usr/local/cuda/samples/common/inc/

#-------------------------------------------------------------------------------
#	Math library specification 
#-------------------------------------------------------------------------------
#
LIBS		= -lm -lcufft
#
#-------------------------------------------------------------------------------
#	Miscellaneous Standard Utilities
#-------------------------------------------------------------------------------
#
AR		= ar
RANLIB		= ranlib

#
#-------------------------------------------------------------------------------
#	Required directives for GMTSAR library
#-------------------------------------------------------------------------------
GMTSAR		= -L. -lgmtsar_d
#
#-------------------------------------------------------------------------------
#	Compiler switches and linker flags
#-------------------------------------------------------------------------------
#
CFLAGS		=  -O2 -Wall -m64 -fPIC -fno-strict-aliasing -std=c99
LDFLAGS		=  -m64 -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBEXT		= a

