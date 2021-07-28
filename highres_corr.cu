/***************************************************************************/
/*  Calculates High resolution correlation for gmtsar-gpu				   *
 *  created by Diego Romano 2021-07-27									   *
 *  based on highres_corr.c 39 2013-04-07 00:49:34Z pwessel $			   */
/*-------------------------------------------------------*/
extern "C"{
#include "gmtsar_ext.h"
#include "xcorr_d.h"
#include "lib_functions.h"
}
#include <iostream>
#include "gpu_err.h"

using namespace std;


void d_fft_interpolate_2d_strip(cufftComplex *d_in, int N1, int M1, int nl, cufftComplex *d_out, int N,  int M, int ifactor, struct d_xcorr d_xc);
void h_reduceCr_strip(cufftComplex *d_in, int nx, int ny, int nl, float *gmax, int *ip, int *jp, struct d_xcorr d_xc);
__global__ void d_cppow_strip(cufftComplex *md, double *corr, int nx, int ny, int nl, int nxc, int nyc, struct locs *loc, int iloc);


/*-------------------------------------------------------------------------------*/
void d_do_highres_corr(struct xcorr xc, int iloc, struct d_xcorr d_xc, int ll, int istep)
{
	int *ip, *jp;
	int	nx, ny, nx2, ny2, ifc;
	float	max_corr, ipeak, jpeak, sub_xoff, sub_yoff, *gmax;

	ifc = xc.interp_factor;

	/* size of complex version of correlation 	*/
	/* must be power of two 			*/
	nx = xc.n2x;
	ny = xc.n2y;

	/* size of interpolated matrix			*/
	nx2 = ifc*nx;
	ny2 = ifc*ny;

	gmax = (float *) malloc(istep*xc.nxl*sizeof(float));
	ip = (int *) malloc(istep*xc.nxl*sizeof(int));
	jp = (int *) malloc(istep*xc.nxl*sizeof(int));

	gErrCk(cudaMemcpy(&d_xc.loc[iloc], &xc.loc[iloc], istep*xc.nxl*sizeof(struct locs), cudaMemcpyHostToDevice));

	/* remove calculated offset from 1 pixel resolution 	*/
	/* factor of 2 on xoff for range interpolation		*/
	/* copy values from correlation to complex 	*/
	/* use values centered around highest value	*/
	dim3 grid(ceil(nx/8.),ceil(ny*istep*xc.nxl/32.));
	dim3 block(8,32);
	d_cppow_strip<<<grid,block>>>(d_xc.md, d_xc.corr, nx, ny, istep*xc.nxl, xc.nxc, xc.nyc, d_xc.loc, iloc);

	d_fft_interpolate_2d_strip(&d_xc.md[0], ny, nx, istep*xc.nxl, &d_xc.cd_exp[0], ny2, nx2, ifc, d_xc);

	/* find maximum in interpolated matrix		*/
	h_reduceCr_strip(&d_xc.cd_exp[0], nx2, ny2, istep*xc.nxl, gmax, ip, jp, d_xc);

	for (int jj=0; jj<istep*xc.nxl; jj++){

		jpeak = (jp[jj] - (ny2 / 2.0f));
		ipeak = (ip[jj] - (ny2 / 2.0f));
		max_corr = gmax[jj];


		/* fft interpolation */
		sub_xoff = jpeak / (float) (ifc);
		sub_yoff = ipeak / (float) (ifc);

		if (debug) {
			fprintf(stderr," highres [ri %d ifc %d](%4.1f %4.1f) (nx %d ny %d) jpeak %f ipeak %f : %f %f : %4.2f\n",
					xc.ri, ifc, xc.loc[iloc].xoff, xc.loc[iloc].yoff, nx2, ny2, jpeak, ipeak, sub_xoff, sub_yoff,  max_corr);
		}

		xc.loc[iloc].xfrac = sub_xoff;
		xc.loc[iloc].yfrac = sub_yoff;
		iloc++;
	}
}
