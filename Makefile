#	makefile for gmtsar-gpu directory (only xcorr supported)

include config.mk
#-------------------------------------------------------------------------------
#	software targets
#-------------------------------------------------------------------------------
all:		lib xcorr-gpu

spotless::	clean

clean:	
		rm -f *.a *.o *% core tags
		

INCLUDES	= $(GMT_INC) -I./

LIB_C		= print_results.c get_locations.c parse_xcorr_input.c utils.c read_xcorr_data.c sio_struct.c plxyz.c

LIB_O		= $(LIB_C:.c=.o)
LIB		= libgmtsar_d.$(LIBEXT)

#-------------------------------------------------------------------------------
#	library
#-------------------------------------------------------------------------------

$(LIB):		$(LIB_O)
		$(AR) cvur $@ $?
		$(RANLIB) $@

lib:		$(LIB)

#-------------------------------------------------------------------------------
#	program rules
#-------------------------------------------------------------------------------

xcorr.o: xcorr.cu
		$(NVCC) -c $^ $(CUDA_INC) 

fft_interpolate_routines.o: fft_interpolate_routines.cu
		$(NVCC) -c $^ $(CUDA_INC) 
		
conv_ampl_demean.o: conv_ampl_demean.cu
		$(NVCC) -maxrregcount 36 -c $^ $(CUDA_INC) 
		
do_freq_xcorr.o: do_freq_xcorr.cu
		$(NVCC) -c $^ $(CUDA_INC)
		
do_time_int_xcorr.o: do_time_int_xcorr.cu
		$(NVCC) -c $^ $(CUDA_INC) 

highres_corr.o: highres_corr.cu
		$(NVCC) -c $^ $(CUDA_INC) 

xcorr-gpu: xcorr.o fft_interpolate_routines.o conv_ampl_demean.o do_freq_xcorr.o do_time_int_xcorr.o highres_corr.o
		$(NVCC)  $^ $(GMTSAR) $(LIBS) -o $@
