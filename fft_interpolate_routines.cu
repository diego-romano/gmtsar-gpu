/*  created by Diego Romano 2021-07-27									   *
 *  based on some fft-based interpolation routines - rjm July 2009				*
 *  sequential version: fft_interpolate_routines.c 109 2015-01-19 23:01:24Z sandwell $
 *  This software contains source code provided by NVIDIA Corporation.         */
extern "C"{
#include "gmtsar_ext.h"
}
#include <stdio.h>
#include <cufft.h>
#include <iostream>
#include "gpu_err.h"

using namespace std;

/*------------------------------------------------------------------------*/

/* transpose matrix on device							*
 * adapted from transpose source code in CUDA SDK		*/
__global__ void d_trans_mat(cufftComplex *odata,cufftComplex *idata, int pitch, int ynum){

	__shared__ cufftComplex tile[32][32];

	int xIndex = blockIdx.x*32 + threadIdx.x;
	int yIndex = blockIdx.y*32 + threadIdx.y;
	int index_in = xIndex + (yIndex)*pitch;

	if (xIndex<pitch && yIndex<ynum){
		xIndex = blockIdx.y * 32 + threadIdx.x;
		yIndex = blockIdx.x * 32 + threadIdx.y;
		int index_out = xIndex + (yIndex)*ynum;

		for (int i=0; i<32; i+=16) {
			tile[threadIdx.y+i][threadIdx.x] =
					idata[index_in+i*pitch];
		}
		__syncthreads();
		for (int i=0; i<32; i+=16) {
			odata[index_out+i*ynum] =
					tile[threadIdx.x][threadIdx.y+i];
		}
	}

}

/* scaling for inverse 1D fft */
__global__ void d_scale_strip(cufftComplex *d_data,int M, int NY, int nl, float nm){
	int i=threadIdx.x + blockIdx.x*blockDim.x;

	if (i<M*NY*nl){
		float tmp1=d_data[i].x;
		float tmp2=d_data[i].y;

		d_data[i].x=tmp1*nm;
		d_data[i].y=tmp2*nm;
	}
}

/* scaling for inverse 2D fft */
__global__ void d_scaleFft2D(cufftComplex *d_data, int N, int M, int ranfft){
	int ix=threadIdx.x + blockIdx.x*blockDim.x;
	int iy=threadIdx.y + blockIdx.y*blockDim.y;
	int index= ix + iy * N;

	if (iy<M){
		if (ix<N){
			d_data[index].x /= ranfft;
			d_data[index].y /= ranfft;
		}
	}
}


/*------------------------------------------------------------------------*/

/* 1D fft interpolation by zero-padding in frequency domain  */
void d_fft_interpolate_1d_strip(cufftComplex *d_in, int N, int NY, int nl, cufftComplex *d_out, int ifactor)
{
	int M;
	int n[1];
	int nembed[1];
	cufftHandle myplanC2C;

	M = ifactor * N;

	/* forward fft */
	n[0]=N;
	nembed[0]=N;

	gfftErrCk( cufftPlanMany(&myplanC2C, 1, n, nembed, 1, N, nembed, 1, N, CUFFT_C2C, NY*nl) );
	gfftErrCk( cufftExecC2C(myplanC2C, d_in, d_in, CUFFT_FORWARD) );
	gfftErrCk( cufftDestroy(myplanC2C) );

	/* re-arrange values in fft */
	int N2 = N / 2;

	gErrCk( cudaMemcpy2D(d_out,M*sizeof(cufftComplex),d_in,N*sizeof(cufftComplex),N2*sizeof(cufftComplex),nl*NY,cudaMemcpyDeviceToDevice) );
	gErrCk( cudaMemcpy2D(d_out + (2*ifactor-1)*N2,M*sizeof(cufftComplex),d_in + N2,N*sizeof(cufftComplex),N2*sizeof(cufftComplex),nl*NY,cudaMemcpyDeviceToDevice) );

	/* backward fft */
	n[0]=M;
	nembed[0]=M;

	gErrCk( cufftPlanMany(&myplanC2C, 1, n, nembed, 1, M, nembed, 1, M, CUFFT_C2C, NY*nl));
	gfftErrCk( cufftExecC2C(myplanC2C, d_out, d_out, CUFFT_INVERSE) );
	gfftErrCk( cufftDestroy(myplanC2C) );

	/* scale amplitude */
	float nm = (float)ifactor*2.0/(2ULL*M);
	dim3 blck=ceil(M*NY*nl/1024.);

	d_scale_strip<<<blck,1024>>>(d_out,M, NY, nl, nm);

}

/*--------------------------------------------------------------------------------------*/

/* 2D fft interpolation by zero-padding in frequency domain  */
void d_fft_interpolate_2d_strip(cufftComplex *d_in, int N1, int M1, int nl, cufftComplex *d_out, int N,  int M, int ifactor, struct d_xcorr d_xc)
{
	cufftComplex *d_tmp1, *d_tmp2, *d_tmp3;
#if 0
	/* sanity checks */
	if (N != (N1 * ifactor)) error_flag = 1;
	if (M != (M1 * ifactor)) error_flag = 1;
#endif

	d_tmp1=d_xc.tmp1;
	d_tmp2=d_xc.tmp2;
	d_tmp3=d_xc.tmp3;

	gErrCk( cudaMemset((void*) d_tmp1, 0, nl*N1*M*sizeof(cufftComplex)) );
	gErrCk( cudaMemset((void*) d_tmp3, 0, nl*N*M*sizeof(cufftComplex)) );

	d_fft_interpolate_1d_strip(&d_in[0], M1, N1, nl, &d_tmp1[0], ifactor);

	dim3 grid(ceil(M/32.),ceil(N1*nl/32.));
	dim3 block(32,16);

	/* now do columns - need to transpose */
	d_trans_mat<<<grid,block>>>(&d_tmp2[0], &d_tmp1[0], M, nl*N1);

	d_fft_interpolate_1d_strip(&d_tmp2[0], N1, M, nl, &d_tmp3[0], ifactor);

	grid.x=ceil(N*nl/32.) , grid.y=ceil(M/32.);
	d_trans_mat<<<grid,block>>>(&d_out[0], &d_tmp3[0], nl*N,M);

}
