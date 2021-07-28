/***************************************************************************/
/* xcorr-gpu does a 2-D cross correlation on complex SLC images            */
/* using a wavenumber multiplication and exploiting CUDA GPU capabilities. */
/*                                                                         */
/***************************************************************************/

/***************************************************************************
 * Creator:  Diego Romano 												   *
 * 			 (ICAR-CNR)													   *
 * 		   based on original sequential code by: 						   *
 * 		   	 Rob J. Mellors                                                *
 *           (San Diego State University)                                  *
 * Date   :  July 27, 2021                                                 *
 ***************************************************************************/

/***************************************************************************
 * Modification history:                                                   *
 *                                                                         *
 * DATE				        	                           				   *
 *							   							                   *
 ***************************************************************************/

/*-------------------------------------------------------*/
extern "C"{
#include "gmtsar_ext.h"
}
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include "gpu_err.h"

using namespace std;

const char	*USAGE = "xcorr [GMT5SAR] - Compute 2-D cross-correlation of two images\n\n"
"\nUsage: xcorr master.PRM slave.PRM [-time] [-real] [-freq] [-nx n] [-ny n] [-xsearch xs] [-ysearch ys]\n"
"master.PRM     	PRM file for reference image\n"
"slave.PRM     	 	PRM file of secondary image\n"
"-time      		use time cross-correlation\n"
"-freq      		use frequency cross-correlation (default)\n"
"-real      		read float numbers instead of complex numbers\n"
"-noshift  		ignore ashift and rshift in prm file (set to 0)\n"
"-nx  nx    		number of locations in x (range) direction (int)\n"
"-ny  ny    		number of locations in y (azimuth) direction (int)\n"
"-nointerp     		do not interpolate correlation function\n"
"-range_interp ri  	interpolate range by ri (power of two) [default: 2]\n"
"-norange     		do not range interpolate \n"
"-xsearch xs		search window size in x (range) direction (int power of 2 [32 64 128 256])\n"
"-ysearch ys		search window size in y (azimuth) direction (int power of 2 [32 64 128 256])\n"
"-interp  factor    	interpolate correlation function by factor (int) [default, 16]\n"
"-v			verbose\n"
"output: \n freq_xcorr.dat (default) \n time_xcorr.dat (if -time option))\n"
"\nuse fitoffset.csh to convert output to PRM format\n"
"\nExample:\n"
"xcorr IMG-HH-ALPSRP075880660-H1.0__A.PRM IMG-HH-ALPSRP129560660-H1.0__A.PRM -nx 20 -ny 50 \n";

void d_fft_interpolate_1d_strip(cufftComplex *d_in, int N, int NY, int nl, cufftComplex *d_out, int ifactor);
void d_do_freq_corr_strip (struct xcorr xc, int iloc, struct d_xcorr d_xc, int istep);
void h_reduceFD_strip(float *d_in, int nx, int ny, int nl, double *gmean, struct d_xcorr d_xc);
void d_do_highres_corr(struct xcorr xc, int iloc, struct d_xcorr d_xc, int ll, int istep);
__global__ void d_conv_ampl_strip(cufftComplex *d_c, float *d_sc, int nx, int ny, int nl);
__global__ void d_mean_mask_strip(cufftComplex *d_c, short *d_mask, int *d_i, bool flag, double *mean, int nx, int ny, int nl);


/* Allocates device memory and pinned host memory arrays */
void d_allocate_arrays(struct d_xcorr *d_xc, struct xcorr xc, int istep)
{
	int thrds=512;
	int blcks=xc.npx*xc.npy/thrds;

	gErrCk( cudaMalloc((float**) &d_xc->d_of, istep*xc.nxl*blcks*sizeof(float)*3) );

	gErrCk( cudaMallocHost((float**) &d_xc->h_of, istep*xc.nxl*blcks*sizeof(float)*3) );

	blcks=xc.nxc*xc.nyc/(thrds*4);
	gErrCk( cudaMalloc((double**) &d_xc->d_od, istep*xc.nxl*blcks*sizeof(double)*3) );
	gErrCk( cudaMallocHost((double**) &d_xc->h_od, istep*xc.nxl*blcks*sizeof(double)*3) );

	int nx_c = xc.n2x * (xc.interp_factor);
	int ny_c = xc.n2y * (xc.interp_factor);
	blcks=nx_c*ny_c/thrds;

	gErrCk( cudaMalloc((float**) &d_xc->d_oc, istep*xc.nxl*blcks*sizeof(float)*3) );
	gErrCk( cudaMallocHost((float**) &d_xc->h_oc, istep*xc.nxl*blcks*sizeof(float)*3) );

	blcks=xc.nx_corr*xc.ny_corr/thrds;

	gErrCk( cudaMalloc((float**) &d_xc->d_ol, istep*xc.nxl*blcks*sizeof(long long)*3) );
	gErrCk( cudaMallocHost((float**) &d_xc->h_ol, istep*xc.nxl*blcks*sizeof(long long)*3) );
	gErrCk( cudaMallocHost((void**) &d_xc->gamma_num, istep*xc.nxl*sizeof(long long)) );
	gErrCk( cudaMallocHost((void**) &d_xc->gamma_denom1, istep*xc.nxl*sizeof(long long)) );
	gErrCk( cudaMallocHost((void**) &d_xc->gamma_denom2, istep*xc.nxl*sizeof(long long)) );
	gErrCk( cudaMalloc((void**) &d_xc->i1, istep*xc.nxl*xc.npx*xc.npy*sizeof(int)) );
	gErrCk( cudaMalloc((void**) &d_xc->i2, istep*xc.nxl*xc.npx*xc.npy*sizeof(int)) );
	gErrCk( cudaMalloc((void**) &d_xc->c1, istep*xc.nxl*xc.npx*xc.npy*sizeof(cufftComplex)) );
	gErrCk( cudaMalloc((void**) &d_xc->c2, istep*xc.nxl*xc.npx*xc.npy*sizeof(cufftComplex)) );
	gErrCk( cudaMalloc((void**) &d_xc->c3, istep*xc.nxl*xc.npx*xc.npy*sizeof(cufftComplex)) );
	gErrCk( cudaMalloc((void**) &d_xc->ritmp, istep*xc.nxl*xc.ri*xc.npx*xc.npy*sizeof(cufftComplex)) );
	gErrCk( cudaMalloc((void**) &d_xc->mask, xc.npx*xc.npy*sizeof(short)) );
	gErrCk( cudaMalloc((void**) &d_xc->sc, istep*xc.nxl*xc.npx*xc.npy*sizeof(float)) );
	gErrCk( cudaMalloc((void**) &d_xc->corr, istep*xc.nxl*xc.nxc*xc.nyc*sizeof(double)) );
	gErrCk( cudaMalloc((void**) &d_xc->a, istep*xc.nxl*xc.nx_corr*xc.ny_corr*sizeof(int)) );
	gErrCk( cudaMalloc((void**) &d_xc->b, istep*xc.nxl*xc.nx_corr*xc.ny_corr*sizeof(int)) );
	gErrCk( cudaMalloc((void**) &d_xc->c, istep*xc.nxl*xc.nx_corr*xc.ny_corr*sizeof(long long)) );
	gErrCk( cudaMalloc((void**) &d_xc->loc, xc.nyl*(xc.nxl+1)*sizeof(struct locs)) );
	gErrCk( cudaMalloc((void**) &d_xc->msc, istep*xc.nxl*sizeof(double)) );
	gErrCk( cudaMalloc((void**) &d_xc->mean, istep*xc.nxl*sizeof(double)) );
	gErrCk( cudaMallocHost((double**) &d_xc->h_mean, istep*xc.nxl*sizeof(double)) );

	d_xc->interp_flag = xc.interp_flag;
	if (xc.interp_flag == 1){
		int ifc = xc.interp_factor;
		gErrCk( cudaMalloc((void**) &d_xc->md, istep*xc.nxl*xc.n2x*xc.n2y*sizeof(cufftComplex)) );
		gErrCk( cudaMalloc((void**) &d_xc->cd_exp, istep*xc.nxl*nx_c*ny_c*sizeof(cufftComplex)) );
		gErrCk( cudaMemset((void*) d_xc->md, 0, istep*xc.nxl*xc.n2x*xc.n2y*sizeof(cufftComplex)) );
		gErrCk( cudaMalloc((void**) &d_xc->tmp1, istep*xc.nxl*xc.n2x*xc.n2y*ifc*sizeof(cufftComplex)) );
		gErrCk( cudaMalloc((void**) &d_xc->tmp2, istep*xc.nxl*xc.n2x*xc.n2y*ifc*sizeof(cufftComplex)) );
		gErrCk( cudaMalloc((void**) &d_xc->tmp3, istep*xc.nxl*xc.n2x*xc.n2y*ifc*ifc*sizeof(cufftComplex)) );
	}
}

/* Frees device memory and pinned host memory */
void d_free_arrays(struct d_xcorr *d_xc){
	gErrCk( cudaFree(d_xc->mask) );
	gErrCk( cudaFree(d_xc->c1) );
	gErrCk( cudaFree(d_xc->c2) );
	gErrCk( cudaFree(d_xc->c3) );
	gErrCk( cudaFree(d_xc->i1) );
	gErrCk( cudaFree(d_xc->i2) );
	gErrCk( cudaFree(d_xc->ritmp) );
	gErrCk( cudaFree(d_xc->sc) );
	gErrCk( cudaFree(d_xc->d_of) );
	gErrCk( cudaFreeHost(d_xc->h_of) );
	gErrCk( cudaFree(d_xc->d_od) );
	gErrCk( cudaFreeHost(d_xc->h_od) );
	gErrCk( cudaFree(d_xc->d_oc) );
	gErrCk( cudaFreeHost(d_xc->h_oc) );
	gErrCk( cudaFree(d_xc->d_ol) );
	gErrCk( cudaFreeHost(d_xc->h_ol) );
	gErrCk( cudaFreeHost(d_xc->gamma_num) );
	gErrCk( cudaFreeHost(d_xc->gamma_denom1) );
	gErrCk( cudaFreeHost(d_xc->gamma_denom2) );
	gErrCk( cudaFree(d_xc->corr) );
	gErrCk( cudaFree(d_xc->a) );
	gErrCk( cudaFree(d_xc->b) );
	gErrCk( cudaFree(d_xc->c) );
	gErrCk( cudaFree(d_xc->loc) );
	gErrCk( cudaFree(d_xc->msc) );
	gErrCk( cudaFree(d_xc->mean) );
	gErrCk( cudaFreeHost(d_xc->h_mean) );

	if (d_xc->interp_flag == 1){
		gErrCk( cudaFree(d_xc->md) );
		gErrCk( cudaFree(d_xc->cd_exp) );
		gErrCk( cudaFree(d_xc->tmp1) );
		gErrCk( cudaFree(d_xc->tmp2) );
		gErrCk( cudaFree(d_xc->tmp3) );
	}
}

/* Estimates the number of patch rows allocatable in device memory and allocates them */
int estimate_istep(struct d_xcorr *d_xc, struct xcorr xc)
{
	int istep;
	int i;
	size_t free,total,worksize, wsize;
	cudaMemGetInfo 	(&free,&total);

	int thrds=512;
	int blcks=xc.npx*xc.npy/thrds;
	int fact1 = blcks*sizeof(float)*3;

	blcks=xc.nxc*xc.nyc/(thrds*4);
	fact1 += blcks*sizeof(double)*3;

	int nx_c = xc.n2x * (xc.interp_factor);
	int ny_c = xc.n2y * (xc.interp_factor);
	blcks=nx_c*ny_c/thrds;
	fact1 += blcks*sizeof(float)*3;

	blcks=xc.nx_corr*xc.ny_corr/512;
	fact1 += blcks*sizeof(long long)*3;
	fact1 *= xc.nxl;

	int fact2 = xc.nxl*(xc.npx*xc.npy*(sizeof(int)*2 + (sizeof(cufftComplex))*(3+xc.ri) +
			sizeof(float)) + xc.nxc*xc.nyc*sizeof(double) + xc.nx_corr*xc.ny_corr*(2*sizeof(int)+sizeof(long long)));

	int fact3 = 0;
	if (xc.interp_flag == 1){
		fact3 = xc.nxl*(xc.n2x*xc.n2y*sizeof(cufftComplex) + nx_c*ny_c*sizeof(cufftComplex));
	}


	int fact4 = xc.npx*xc.npy*sizeof(short) + xc.nyl*(xc.nxl+1)*sizeof(struct locs);

	int n[1], nembed[1], M;
	size_t fact5=0;
	M = xc.npx * xc.ri;

	istep = xc.nyl;
	i=istep;
	do{
			if(xc.nyl%i==0)
				istep=i;
			i--;
			worksize = (istep*(size_t)(fact1+fact2+fact3)+fact4);
	} while ((i>0 && worksize>=free)||(istep*xc.nxl)>1024);

	d_allocate_arrays(d_xc, xc, istep);

	cudaMemGetInfo 	(&free,&total);
	n[0]=xc.npx, nembed[0]=xc.npx;
	cufftEstimateMany(1, n, nembed, 1, xc.npx, nembed, 1, xc.npx, CUFFT_C2C, istep*xc.npy*xc.nxl, &wsize);
	fact5 += wsize;
	n[0]=M, nembed[0]=M;
	cufftEstimateMany(1, n, nembed, 1, M, nembed, 1, M, CUFFT_C2C, istep*xc.npy*xc.nxl, &wsize);
	fact5 += wsize;

	if (free<=fact5){
		d_free_arrays(d_xc);
		do {
			i--;
		} while(!(xc.nyl%i==0));
		istep=i;
		d_allocate_arrays(d_xc, xc, istep);
	}

	cout<<"istep = "<<istep<<"  available = "<<free<<endl;

	return istep;
}


/*-------------------------------------------------------------------------------*/


/* Does range interpolation on device by concurrently processing a strip of patches */
int d_do_range_interpolate_strip(cufftComplex *d_c, int nx, int ny, int nl, int ri, cufftComplex *d_work)
{

	gErrCk( cudaMemset2D((void*) d_work, ri*nx*sizeof(cufftComplex), 0, ri*nx*sizeof(cufftComplex),ny*nl) );

	/* interpolate c and put into work */
	d_fft_interpolate_1d_strip(d_c, nx, ny, nl, d_work, ri);

	gErrCk( cudaMemcpy2D(d_c, nx*sizeof(cufftComplex), d_work+nx/2, nx*ri*sizeof(cufftComplex), nx*sizeof(cufftComplex), ny*nl, cudaMemcpyDeviceToDevice) );

	gErrCk( cudaDeviceSynchronize() );

	return(EXIT_SUCCESS);
}

/* Assigns values to device arrays for subsequent proper correlation. Includes  *
 * calls to range interpolation, amplitude conversion and demeaning 			*/
/*-------------------------------------------------------------------------------*/
/* complex arrays used in fft correlation		*/
/* c1 is master									*/
/* c2 is aligned								*/
/* c3 used in fft complex correlation			*/
/* c1, c2, and c3 are npy by npx by nxl by istep*/
/*-------------------------------------------------------------------------------*/
void d_assign_values(struct xcorr xc, int iloc, struct d_xcorr d_xc, int istep)
{

	/* set grid and thread block dimensions for CUDA kernels */
	dim3 grid(ceil(istep*xc.nxl*xc.npy*xc.npx/1024.));
	dim3 block(1024);

	/* range interpolate */
	if (xc.ri > 1) {
		d_do_range_interpolate_strip(d_xc.c1, xc.npx, xc.npy, istep*xc.nxl, xc.ri, d_xc.ritmp);
		d_do_range_interpolate_strip(d_xc.c2, xc.npx, xc.npy, istep*xc.nxl, xc.ri, d_xc.ritmp);
	}

	/* convert to amplitude and demean c1 */
	d_conv_ampl_strip<<<grid,block>>>(d_xc.c1, d_xc.sc, xc.npx, xc.npy, istep*xc.nxl);

	h_reduceFD_strip(d_xc.sc, xc.npx, xc.npy, istep*xc.nxl, d_xc.h_mean, d_xc);
	gErrCk( cudaMemcpy(d_xc.mean, d_xc.h_mean, istep*xc.nxl*sizeof(double), cudaMemcpyHostToDevice) );

	d_mean_mask_strip<<<grid,block,istep*xc.nxl*sizeof(float)>>>(d_xc.c1, d_xc.mask, d_xc.i1, false, d_xc.mean, xc.npx, xc.npy, istep*xc.nxl);

	/* ---------------------------------------------- */

	/* convert to amplitude and demean c2 */
	d_conv_ampl_strip<<<grid,block>>>(d_xc.c2, d_xc.sc, xc.npx, xc.npy, istep*xc.nxl);

	h_reduceFD_strip(d_xc.sc, xc.npx, xc.npy, istep*xc.nxl, d_xc.h_mean, d_xc);
	gErrCk( cudaMemcpy(d_xc.mean, d_xc.h_mean, istep*xc.nxl*sizeof(double), cudaMemcpyHostToDevice) );

	d_mean_mask_strip<<<grid,block,istep*xc.nxl*sizeof(float)>>>(d_xc.c2, d_xc.mask, d_xc.i2, true, d_xc.mean, xc.npx, xc.npy, istep*xc.nxl);

}


/*-------------------------------------------------------------------------------*/


/* Routine to prepare data for correlation, call to correlation and hi-res offset calculation */
void do_correlation(struct xcorr xc)
{
	int	i, j, k, iloc, istep;
	struct d_xcorr d_xc;
	int mstride, cstride;

	/* opportunity for concurrency on multiple rows of patches. 					  			*
	 * Use on devices with big memory to try a further speed-up.				   	  			*
	 * Uncomment the following line for automatic estimation and comment the static assignment  */
	//istep = estimate_istep(&d_xc, xc);
	istep = 1;

	/* allocate arrays  */
	allocate_arrays(&xc);
	/* Comment the following line if using automatic estimation of istep */
	d_allocate_arrays(&d_xc, xc, istep);

	/* make mask */
	make_mask(xc);
	gErrCk( cudaMemcpy(d_xc.mask, xc.mask, xc.npx*xc.npy*sizeof(short), cudaMemcpyHostToDevice) );

	iloc = 0;
	/* cycle on the rows of patches */
	for (i=0; i<xc.nyl; i+=istep){

		/* c1, c2 are npy by npx by nxl by istep		*/
		/* d1, d2 are npy by nx (length of line in SLC)	*/
		for (k=0; k<istep; k++){
			/* read in data for each row */
			read_xcorr_data(xc, iloc+k*xc.nxl);

			for (j=0; j<xc.nxl; j++){
				/* master and aligned x offsets */
				int mx = xc.loc[iloc+j+k*xc.nxl].x - xc.npx/2;
				int sx = xc.loc[iloc+j+k*xc.nxl].x + xc.x_offset - xc.npx/2;
				mstride= xc.npx * xc.npy * ( j + k*xc.nxl);

				gErrCk( cudaMemcpy2DAsync(&d_xc.c1[mstride], xc.npx*sizeof(cufftComplex), &xc.d1[mx], xc.m_nx*sizeof(cufftComplex), xc.npx*sizeof(cufftComplex), xc.npy, cudaMemcpyHostToDevice) );
				gErrCk( cudaMemcpy2DAsync(&d_xc.c2[mstride], xc.npx*sizeof(cufftComplex), &xc.d2[sx], xc.s_nx*sizeof(cufftComplex), xc.npx*sizeof(cufftComplex), xc.npy, cudaMemcpyHostToDevice) );
			}
			gErrCk( cudaDeviceSynchronize() );
		}

		gErrCk( cudaMemset((void*) &d_xc.md[0], 0, istep*xc.nxl*xc.n2x*xc.n2y*sizeof(cufftComplex)) );

		/* assign values to proper device arrays for correlation */
		d_assign_values(xc, iloc, d_xc, istep);

		/* correlate patch with data over offsets in freq domain *
		 * TODO: time correlation (not yet supported on GPU)     */
		if (xc.corr_flag == 2) d_do_freq_corr_strip(xc, iloc, d_xc, istep);

		/* oversample correlation surface  to obtain sub-pixel resolution */
		if (xc.interp_flag == 1) d_do_highres_corr(xc, iloc, d_xc, j, istep);

		/* cycle to print a strip of results, and calculate time *
		 * correlation on CPU if needed							 */
		for (j=0; j<istep*xc.nxl; j++){

			if (debug) fprintf(stderr," initial: iloc %d (%d,%d)\n",iloc, xc.loc[iloc].x,xc.loc[iloc].y);

			if (debug) print_complex(xc.c1, xc.npy, xc.npx, 1);
			if (debug) print_complex(xc.c2, xc.npy, xc.npx, 1);

			/* correlate patch with data over offsets in time domain */
			if (xc.corr_flag < 2) {
				cstride= xc.nxc * xc.nyc * j ;
				gErrCk( cudaMemcpy((xc.corr), d_xc.corr+cstride, xc.nxc*xc.nyc*sizeof(double), cudaMemcpyDeviceToHost) );
				do_time_corr(xc, iloc);
			}

			/* write out results */
			print_results(xc, iloc);

			iloc++;
		} /* end of x iloc loop */
	} /* end of y iloc loop */

	d_free_arrays(&d_xc);
}


/*-------------------------------------------------------------------------------*/
/* want to avoid circular correlation so mask out most of b			*/
/* could adjust shape for different geometries					*/
/*-------------------------------------------------------------------------------*/
void make_mask(struct xcorr xc)
{
int i,j,imask;
imask = 0;

	for (i=0; i<xc.npy; i++){
		for (j=0; j<xc.npx; j++){
			 xc.mask[i*xc.npx + j] = 1;
			if ((i < xc.ysearch) || (i >= (xc.npy - xc.ysearch))) {
				xc.mask[i*xc.npx +j] = imask;
				}
			if ((j < xc.xsearch) || (j >= (xc.npx - xc.xsearch))) {
				xc.mask[i*xc.npx + j] = imask;
				}
			}
		}
}


/*-------------------------------------------------------------------------------*/

/* Allocate host arrays in pinned memory for faster copy to device */
void allocate_arrays(struct xcorr *xc)
{
	int	nx, ny, nx_exp, ny_exp;

	gErrCk( cudaMallocHost((void**) &xc->d1, xc->m_nx*xc->npy*sizeof(struct FCOMPLEX)) );
	gErrCk( cudaMallocHost((void**) &xc->d2, xc->s_nx*xc->npy*sizeof(struct FCOMPLEX)) );
	gErrCk( cudaMallocHost((void**) &xc->i1, xc->npx*xc->npy*sizeof(int)) );
	gErrCk( cudaMallocHost((void**) &xc->i2, xc->npx*xc->npy*sizeof(int)) );
	gErrCk( cudaMallocHost((void**) &xc->c1, xc->npx*xc->npy*sizeof(struct FCOMPLEX)) );
	gErrCk( cudaMallocHost((void**) &xc->c2, xc->npx*xc->npy*sizeof(struct FCOMPLEX)) );
	gErrCk( cudaMallocHost((void**) &xc->c3, xc->npx*xc->npy*sizeof(struct FCOMPLEX)) );

	xc->mask = (short *) malloc(xc->npx*xc->npy*sizeof(short));

	/* this is size of correlation patch */
	gErrCk( cudaMallocHost((void**) &xc->corr, xc->npx*xc->npy*sizeof(double)) );

	if (xc->interp_flag == 1){
		nx = 2*xc->n2x;
		ny = 2*xc->n2y;
		nx_exp = nx * (xc->interp_factor);
		ny_exp = ny * (xc->interp_factor);
		gErrCk( cudaMallocHost((void**) &xc->md, nx * ny * sizeof(struct FCOMPLEX)) );
		gErrCk( cudaMallocHost((void**) &xc->cd_exp, nx_exp * ny_exp * sizeof(struct FCOMPLEX)) );
	}
}

/*-------------------------------------------------------*/


int main(int argc,char **argv)
{
	int	input_flag, nfiles;
	struct	xcorr xc;
	clock_t start, end;
	double	cpu_time;

	verbose = 0;
	debug = 0;
	input_flag = 0;
	nfiles = 2;
	xc.interp_flag = 0;
	xc.corr_flag = 2;

	/* change device ID if needed */
	gErrCk( cudaSetDevice(0) );
	gErrCk( cudaDeviceReset() );

	if (argc < 3) die (USAGE, "");

	set_defaults(&xc);
	
	parse_command_line(argc, argv, &xc, &nfiles, &input_flag, USAGE);

	/* read prm files */
	if (input_flag == 0) handle_prm(argv, &xc, nfiles);

	if (debug) print_params(&xc);

	/* output file */
	if (xc.corr_flag == 0) strcpy(xc.filename,"time_xcorr.dat");
	if (xc.corr_flag == 1) strcpy(xc.filename,"time_xcorr_Gatelli.dat");
	if (xc.corr_flag == 2) strcpy(xc.filename,"freq_xcorr.dat");

	xc.file = fopen(xc.filename,"w");
	if (xc.file == NULL) die("Can't open output file",xc.filename); 

	/* x locations, y locations */
	get_locations(&xc);

	/* calculate correlation at all points */
	start = clock();

	do_correlation (xc);

	end = clock();
	cpu_time  = ((double) (end-start)) / CLOCKS_PER_SEC;	
	fprintf(stdout, " elapsed time: %lf \n", cpu_time);

	fclose(xc.data1);
	fclose(xc.data2);

	return(EXIT_SUCCESS);
}
