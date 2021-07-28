/***************************************************************************/
/*  Calculates frequency correlation for gmtsar-gpu				   *
 *  created by Diego Romano 2021-07-27									   *								   *
 *  based on do_freq_xcorr.c 109 2015-01-19 23:01:24Z sandwell $	*/
/*-------------------------------------------------------*/
extern "C"{
#include <math.h>
#include "gmtsar_ext.h"
#include "siocomplex.h"
#include "sarleader_ALOS.h"
#include "sarleader_fdr.h"
#include "xcorr_d.h"
}
#include <cufft.h>
#include <iostream>
#include <stdio.h>
#include "gpu_err.h"

using namespace std;

void h_reduce_strip(double *d_in, int nx, int ny, int nl, double *gmean, double *gmax, int *ip, int *jp, struct d_xcorr d_xc);
void d_calc_time_corr_strip(struct xcorr xc, struct d_xcorr d_xc, double *gamma, int nl);
__global__ void d_scaleFft2D(cufftComplex *d_data, int N, int M, int ranfft);
__global__ void d_scaled_strip(double *d_data, int size, int nl, double *value);

__device__ float d_Cabs(cufftComplex z)
{
	return (float)hypot((double)z.x, (double)z.y);
}

/* device function for complex multiplication */
__device__ cufftComplex d_Cmul(cufftComplex x, cufftComplex y)
	{
	double zx,zy;
	    cufftComplex z;
	    zx = x.x*y.x;
	    zy = x.y*y.y;
	    z.x = zx - zy;
	    zx = x.y*y.x;
	    zy = x.x*y.y;
	    z.y = zx + zy;
	    return z;
	}

/* kernel for multiplication by conjugate on GPU */
__global__ void d_mulaconjb(cufftComplex *d_c1,cufftComplex *d_c2,cufftComplex *d_c3, int N, int M){
	int ix=threadIdx.x + blockIdx.x*blockDim.x;
	int iy=threadIdx.y + blockIdx.y*blockDim.y;
	int index= ix + iy * N;
	int isign;
	cufftComplex ctmp1,ctmp2;

	if (iy<M){
		if (ix<N){
			if ((ix+iy)%2==0)
				isign=1;
			else isign=-1;
			ctmp1 = cuConjf(d_c2[index]);
			ctmp2 = d_Cmul(d_c1[index],ctmp1);
			d_c3[index].x = ctmp2.x*isign;
			d_c3[index].y = ctmp2.y*isign;
		}
	}
}

/* kernel for copying data into correlation matrix */
__global__ void d_fillcorrmat_strip(double *d_corr, cufftComplex *d_c3, int nx, int ny, int nx_c, int ny_c, int npx, int npy, int nl){
	int ix=threadIdx.x + blockIdx.x*blockDim.x;
	int iy=threadIdx.y + blockIdx.y*blockDim.y;
	int index= ix + iy * nx;
	int ii, k, coi, c3i;

	if (iy<ny){
		if (ix<nx){
			for(k=0; k<nl; k++){
				coi= k * nx * ny + index;
				ii = (iy + ny_c / 2)*npx + ix + (nx_c / 2);
				c3i= k * npx * npy + ii;
				d_corr[coi] = d_Cabs(d_c3[c3i]);

			}
		}
	}
}


/*-------------------------------------------------------------------------------*/

/* multiplies strips of matrices in frequency domain on GPU */
void d_fft_multiply_strip(int N, int M, int nl, cufftComplex *d_c1, cufftComplex *d_c2, cufftComplex *d_c3)
{
	cufftHandle myplanC2C;
	int n[2];

	/* do forward fft 					*/
	n[0]=M;
	n[1]=N;

	gfftErrCk(cufftPlanMany(&myplanC2C, 2, n, NULL, 1, N*M, NULL, 1, N*M, CUFFT_C2C, nl));
	gfftErrCk(cufftExecC2C(myplanC2C, d_c1, d_c1, CUFFT_FORWARD));
	gfftErrCk(cufftExecC2C(myplanC2C, d_c2, d_c2, CUFFT_FORWARD));

	/* multiply a with conj(b)				*/
	/* the isign should shift the results appropriately	*/
	/* normalize by absolute values				*/
	dim3 grid(ceil(N/32.),ceil((M*nl)/32.));
	dim3 block(32,32);

	d_mulaconjb<<<grid,block>>>(d_c1,d_c2,d_c3, N, M*nl);

	/* inverse fft  for cross-correlation (c matrix) 	*/

	gfftErrCk(cufftExecC2C(myplanC2C, d_c3, d_c3, CUFFT_INVERSE));
	d_scaleFft2D<<<grid,block>>>(d_c3, N, M*nl, N*M);

	cufftDestroy(myplanC2C);
}


/*-------------------------------------------------------------------------------*/

/* calculates correlation between strips of matrices in frequency domain */
void d_do_freq_corr_strip (struct xcorr xc, int iloc, struct d_xcorr d_xc, int istep)
{
	int 	j, *ip, *jp;
	float	ipeak, jpeak;
	double	*cmax, *cave, *max_corr, msc;
	int mstride,cstride;

	cave = (double *) malloc(istep*xc.nxl*sizeof(double));
	cmax = (double *) malloc(istep*xc.nxl*sizeof(double));
	max_corr = (double *) malloc(istep*xc.nxl*sizeof(double));
	ip = (int *) malloc(istep*xc.nxl*sizeof(int));
	jp = (int *) malloc(istep*xc.nxl*sizeof(int));


	/* multiply c1 and c2 using fft */
	d_fft_multiply_strip(xc.npx, xc.npy, istep*xc.nxl, d_xc.c1, d_xc.c2, d_xc.c3);

	/* transfer results into correlation matrix		*/
	dim3 grid(ceil(xc.nxc/32.),ceil(xc.nyc/32.));
	dim3 block(32,32);
	d_fillcorrmat_strip<<<grid,block>>>(d_xc.corr, d_xc.c3, xc.nxc, xc.nyc, xc.nx_corr, xc.ny_corr, xc.npx, xc.npy, istep*xc.nxl);


	/* calculate normalized correlation at best point */
	h_reduce_strip(d_xc.corr, xc.nxc, xc.nyc, istep*xc.nxl, cave, cmax, ip, jp, d_xc);

	/* cycle to process each peak in the strip on host memory and prepare matrices for time correlation */
	for (j=0; j<istep*xc.nxl; j++){

		mstride= xc.npx * xc.npy * j ;
		cstride= xc.nx_corr * xc.ny_corr * j ;

		max_corr[j] = -999.0;
		jpeak = ipeak = -999.0f;

		jpeak = (jp[j]- xc.nxc / 2.0f);
		ipeak = (ip[j]- xc.nyc / 2.0f);

		if ((ipeak == -999.0) || (jpeak == -999.0)){
			fprintf(stderr,"error! jpeak %f ipeak %f cmax %lf xc %f \n", jpeak, ipeak, cmax[j], xc.corr[100]);
			exit(1);
		}

		/* estimate maximum correlation using frequency - poor */
		max_corr[j] = (cmax[j]/cave[j]);

		/* copy data to temporary array for time correlation */
		int astride=mstride+(xc.ysearch+(int) ipeak)*xc.npx + (int) jpeak + xc.xsearch;
		int bstride=mstride+(xc.ysearch)*xc.npx + xc.xsearch;

		gErrCk(cudaMemcpy2DAsync(&d_xc.a[cstride], xc.nx_corr*sizeof(int), &d_xc.i1[astride], xc.npx*sizeof(int), xc.nx_corr*sizeof(int), xc.ny_corr, cudaMemcpyDeviceToDevice));
		gErrCk(cudaMemcpy2DAsync(&d_xc.b[cstride], xc.nx_corr*sizeof(int), &d_xc.i2[bstride], xc.npx*sizeof(int), xc.nx_corr*sizeof(int), xc.ny_corr, cudaMemcpyDeviceToDevice));
	}

	/* use normalized time correlation rather than frequency estimate for value */
	if (ipeak != -999.0) d_calc_time_corr_strip(xc, d_xc, max_corr, istep*xc.nxl);

	/* cycle to memorize offset in host memory */
	for (j=0; j<istep*xc.nxl; j++){

		jpeak = (jp[j]- xc.nxc / 2.0f);
		ipeak = (ip[j]- xc.nyc / 2.0f);

		xc.loc[iloc].xoff = -1 * jpeak;
		xc.loc[iloc].yoff = -1 * ipeak;
		xc.loc[iloc].corr = (float)max_corr[j];

		/* calculate normalization factor per patch and transfer to device */
		msc = (double)cmax[j]/max_corr[j];
		gErrCk(cudaMemcpy(d_xc.msc+j, &msc, sizeof(double), cudaMemcpyHostToDevice));

		iloc++;
	}

	/* normalize correlation data on device */
	int grd=ceil(xc.nxc*xc.nyc/1024.);
	d_scaled_strip<<<grd,1024,istep*xc.nxl*sizeof(double)>>>(d_xc.corr, xc.nxc*xc.nyc, istep*xc.nxl, &d_xc.msc[0]);
}
