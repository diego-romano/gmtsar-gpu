/***************************************************************************/
/*  Several kernels and routines for gmtsar-gpu							   *
 *  created by Diego Romano 2021-07-27									   *
 *  																	   *
 *  This software contains source code provided by NVIDIA Corporation.     *
/*-------------------------------------------------------------------------*/

#include <cooperative_groups.h>
#include "cufft.h"
extern "C" {
#include "soi_ext.h"
#include "xcorr_d.h"
}

#include <iostream>
#include "gpu_err.h"

using namespace std;

namespace cg = cooperative_groups;

extern "C"
bool isPow2(unsigned int x)
{
	return ((x&(x-1))==0);
}

struct d_max_id{
	double m;
	int i;
};

template<class T>
struct SharedMemory
{
	__device__ inline operator       T *()
    		{
		extern __shared__ int __smem[];
		return (T *)__smem;
    		}

	__device__ inline operator const T *() const
    		{
		extern __shared__ int __smem[];
		return (T *)__smem;
    		}
};

/* kernel to scale values in a matrix strip by values in a single matrix */
__global__ void d_scaled_strip(double *d_data, int size, int nl, double *value){

	int i=threadIdx.x + blockIdx.x*blockDim.x;
	int ix=threadIdx.x;
	int k, inx;
	double *sval= SharedMemory<double>();

	if (ix<nl)
		sval[ix]= value[ix];

	__syncthreads();

	if(i<size){
		for (k=0; k<nl; k++){
			inx= k * size + i;
			d_data[inx]/=sval[k];
		}
	}
}

/* GPU kernel to */
/* remove calculated offset from 1 pixel resolution 	*/
/* factor of 2 on xoff for range interpolation		*/
/* copy values from correlation to complex 	*/
/* use values centered around highest value	*/
__global__ void d_cppow_strip(cufftComplex *md, double *corr, int nx, int ny, int nl, int nxc, int nyc, struct locs *loc, int iloc){
	int ix=threadIdx.x + blockIdx.x*blockDim.x;
	int iyo=threadIdx.y + blockIdx.y*blockDim.y;
	int iy=iyo%ny;
	int iyb=floorf(iyo/ny);
	int i = ix + (iyb * ny + iy ) * nx;
	iloc+=iyb;
	int jc = (nxc / 2) - nx / 2 - (int) loc[iloc].xoff;
	int ic = (nyc / 2) - ny / 2 - (int) loc[iloc].yoff;
	int k = (ic+iy)*nxc + (jc+ix);
	int j = iyb * nxc * nyc + k;

	if (iy<ny&&iyb<nl){
		if (ix<nx){
			if (!((k < 0)||(k > nxc*nyc))){
				md[i].x = powf(corr[j], 0.25f);
				md[i].y = 0.0f;
			}
		}
	}
}

/* kernel to convert a complex matrix strip to amplitude values */
__global__ void d_conv_ampl_strip(cufftComplex *d_c, float *d_sc, int nx, int ny, int nl){
	int i=threadIdx.x + blockIdx.x*blockDim.x;
	float tmp;

	if (i<nx*ny*nl){
		tmp = hypotf(d_c[i].x,d_c[i].y);
		d_sc[i] = tmp;
		d_c[i].x = tmp;
		d_c[i].y = 0.0f;
	}
}

/* kernel to demean and mask a matrix strip */
__global__ void d_mean_mask_strip(cufftComplex *d_c, short *d_mask, int *d_i, bool flag, double *mean, int nx, int ny, int nl){
	int j=threadIdx.x + blockIdx.x*blockDim.x;
	int i= threadIdx.x;
	int k;
	float supp;
	float *smean= SharedMemory<float>();

	if (i<nl)
		smean[i]= (float)mean[i];

	__syncthreads();

	if (j<nx*ny){
		if (flag)
			supp = (float) d_mask[j];
		else
			supp = 1.0f;

		for (k=0; k<nl; k++) {
			i = k * nx * ny + j;
			d_c[i].x -= smean[k];

			/* apply mask */
			d_c[i].y = 0.0f;
			d_c[i].x = d_c[i].x * supp;
			d_i[i] = (int) (d_c[i].x);
		}
	}
}

/* sums complex array elements (real part) on device per strip	*
 * adapted from reduce source code in CUDA SDK					*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6C(cufftComplex *g_idata, T *g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
	maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
	const unsigned int mask = (0xffffffff) >> maskLength;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i].x;

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n){
			mySum += g_idata[i+blockSize].x;
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64){
			mySum += sdata[tid + 32];
		}
		// Reduce final warp using shuffle
		for (int offset = tile32.size()/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(mask,mySum, offset);
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) {
		g_odata[blockIdx.x*3] = mySum;
	}
}


/* sums complex array elements (real part) and find maximum on device per strip	*
 * adapted from reduce source code in CUDA SDK									*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6bC(cufftComplex *g_idata, T *g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	struct d_max_id *sdata2= SharedMemory<struct d_max_id>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
	maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
	const unsigned int mask = (0xffffffff) >> maskLength;

	struct d_max_id myMax;
	myMax.m = 0;
	myMax.i = i;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		//myMax = g_idata[i] > myMax ? g_idata[i] : myMax;
		if (g_idata[i].x > myMax.m){
			myMax.m = g_idata[i].x;
			myMax.i = i;
		}

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n){
			//myMax = g_idata[i+blockSize] > myMax ? g_idata[i+blockSize] : myMax;
			if (g_idata[i+blockSize].x > myMax.m){
				myMax.m = g_idata[i+blockSize].x;
				myMax.i = i+blockSize;
			}
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata2[tid] = myMax;
	cg::sync(cta);

	// do reduction in shared mem
	if ((blockSize == 1024) && (tid < 512))
	{
		//sdata2[tid] = myMax = sdata2[tid + 256] > myMax ? sdata2[tid + 256] : myMax;
		if (sdata2[tid + 512].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 512];
		}
	}

	cg::sync(cta);

	if ((blockSize >= 512) && (tid < 256))
	{
		//sdata2[tid] = myMax = sdata2[tid + 256] > myMax ? sdata2[tid + 256] : myMax;
		if (sdata2[tid + 256].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 256];
		}
	}

	cg::sync(cta);

	if ((blockSize >= 256) &&(tid < 128))
	{
		//sdata2[tid] = myMax = sdata2[tid + 128] > myMax ? sdata2[tid + 128] : myMax;
		if (sdata2[tid + 128].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 128];
		}
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid <  64))
	{
		//sdata2[tid] = myMax = sdata2[tid + 64] > myMax ? sdata2[tid + 64] : myMax;
		if (sdata2[tid + 64].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 64];
		}
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64){
			//myMax = sdata2[tid + 32] > myMax ? sdata2[tid + 32] : myMax;
			if (sdata2[tid + 32].m > myMax.m){
				sdata2[tid] = myMax = sdata2[tid + 32];
			}
		}
		// Reduce final warp using shuffle
		for (int offset = tile32.size()/2; offset > 0; offset /= 2)
		{
			double leamax = myMax.m;
			int leaind = myMax.i;
			leamax = __shfl_down_sync(mask,leamax, offset);
			leaind = __shfl_down_sync(mask,leaind, offset);
			//myMax = regmax > myMax ? regmax : myMax;
			if (leamax > myMax.m){
				myMax.m = leamax;
				myMax.i = leaind;
			}
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) {
		g_odata[blockIdx.x*3+1] = myMax.m;
		g_odata[blockIdx.x*3+2] = myMax.i;
	}
}

/* sums array elements on device per strip			*
 * adapted from reduce source code in CUDA SDK		*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
	maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
	const unsigned int mask = (0xffffffff) >> maskLength;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n){
			mySum += g_idata[i+blockSize];
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid +  64];
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64){
			mySum += sdata[tid + 32];
		}
		// Reduce final warp using shuffle
		for (int offset = tile32.size()/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down_sync(mask,mySum, offset);
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) {
		g_odata[blockIdx.x*3] = mySum;
	}
}


/* sums array elements and find maximum on device per strip	*
 * adapted from reduce source code in CUDA SDK				*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6b(T *g_idata, T *g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	struct d_max_id *sdata2= SharedMemory<struct d_max_id>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
	maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
	const unsigned int mask = (0xffffffff) >> maskLength;

	struct d_max_id myMax;
	myMax.m = 0;
	myMax.i = i;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		//myMax = g_idata[i] > myMax ? g_idata[i] : myMax;
		if (g_idata[i] > myMax.m){
			myMax.m = g_idata[i];
			myMax.i = i;
		}

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n){
			//myMax = g_idata[i+blockSize] > myMax ? g_idata[i+blockSize] : myMax;
			if (g_idata[i+blockSize] > myMax.m){
				myMax.m = g_idata[i+blockSize];
				myMax.i = i+blockSize;
			}
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata2[tid] = myMax;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize == 1024) && (tid < 512))
	{
		//sdata2[tid] = myMax = sdata2[tid + 256] > myMax ? sdata2[tid + 256] : myMax;
		if (sdata2[tid + 512].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 512];
		}
	}

	cg::sync(cta);

	if ((blockSize >= 512) && (tid < 256))
	{
		//sdata2[tid] = myMax = sdata2[tid + 256] > myMax ? sdata2[tid + 256] : myMax;
		if (sdata2[tid + 256].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 256];
		}
	}

	cg::sync(cta);

	if ((blockSize >= 256) &&(tid < 128))
	{
		//sdata2[tid] = myMax = sdata2[tid + 128] > myMax ? sdata2[tid + 128] : myMax;
		if (sdata2[tid + 128].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 128];
		}
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid <  64))
	{
		//sdata2[tid] = myMax = sdata2[tid + 64] > myMax ? sdata2[tid + 64] : myMax;
		if (sdata2[tid + 64].m > myMax.m){
			sdata2[tid] = myMax = sdata2[tid + 64];
		}
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64){
			//myMax = sdata2[tid + 32] > myMax ? sdata2[tid + 32] : myMax;
			if (sdata2[tid + 32].m > myMax.m){
				sdata2[tid] = myMax = sdata2[tid + 32];
			}
		}
		// Reduce final warp using shuffle
		for (int offset = tile32.size()/2; offset > 0; offset /= 2)
		{
			double leamax = myMax.m;
			int leaind = myMax.i;
			leamax = __shfl_down_sync(mask,leamax, offset);
			leaind = __shfl_down_sync(mask,leaind, offset);
			//myMax = regmax > myMax ? regmax : myMax;
			if (leamax > myMax.m){
				myMax.m = leamax;
				myMax.i = leaind;
			}
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) {
		g_odata[blockIdx.x*3+1] = myMax.m;
		g_odata[blockIdx.x*3+2] = myMax.i;
	}
}

/* caller of GPU kernels for complex array reduction *
*  adapted from reduce source code in CUDA SDK		 */
template <class T>
void
reduceC(int size, int threads, int blocks, cufftComplex *d_idata, T *d_odata, int flam)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) * 4 : threads * sizeof(T) * 4;

	// choose which of the optimized versions of reduction to launch
	if (isPow2(size))
	{
		reduce6C<T, 1024, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
		if (flam) reduce6bC<T, 1024, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
	}
	else
	{
		reduce6C<T, 1024, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
		if (flam) reduce6bC<T, 1024, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
	}

}

/* caller of GPU kernels for array reduction 		 *
*  adapted from reduce source code in CUDA SDK		 */
template <class T>
void
reduce(int size, int threads, int blocks, T *d_idata, T *d_odata, int flam)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) * 4 : threads * sizeof(T) * 4;

	// choose which of the optimized versions of reduction to launch
	if (isPow2(size))
	{
		switch (threads)
		{
		case 1024:
			reduce6<T, 1024, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			if (flam) reduce6b<T, 1024, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			break;

		case 512:
			reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			if (flam) reduce6b<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			break;
		}
	}
	else
	{
		switch (threads)
		{
		case 1024:
			reduce6<T, 1024, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			if (flam) reduce6b<T, 1024, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			break;

		case 512:
			reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			if (flam) reduce6b<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
			break;
		}
	}

}

template void
reduce<float>(int size, int threads, int blocks, float *d_idata, float *d_odata, int flam);

template void
reduce<double>(int size, int threads, int blocks, double *d_idata, double *d_odata, int flam);

template void
reduce< long long>(int size, int threads, int blocks,  long long *d_idata,  long long *d_odata, int flam);


/* reduces a double array strip to an array of sums and an array of maximum values interleaved *
 * calculates mean value and returns peak value location per patch in the strip                */
void h_reduce_strip(double *d_in, int nx, int ny, int nl, double *gmean, double *gmax, int *ip, int *jp, struct d_xcorr d_xc){
	double *d_odata, *h_odata;
	int thrds=1024;
	int blcks=nl*nx*ny/(thrds*2);
	int j, mstride1;

	d_odata=d_xc.d_od;
	h_odata=d_xc.h_od;

	reduce<double>(nl*nx*ny, thrds, blcks, d_in, d_odata, 1);

	gErrCk(cudaMemcpy(h_odata, d_odata, blcks*sizeof(double)*3, cudaMemcpyDeviceToHost));

	for (j=0; j<nl; j++){
		mstride1 = (blcks * 3 * j)/nl;
		gmax[j]=-1;
		gmean[j]=0.0;
		int peak=0;
		for (int ind=0; ind<(blcks/nl); ind++)
		{
			if(h_odata[mstride1+ind*3+1] > (gmax[j])){
				gmax[j] = h_odata[mstride1+ind*3+1];
				peak = h_odata[mstride1+ind*3+2];
			}
			gmean[j] += h_odata[mstride1+ind*3];
		}

		gmean[j]/=nx*ny;
		peak -=nx*ny*j;

		ip[j]=floor(peak/nx);
		jp[j]=peak%nx;
	}
}


/* reduces a long long array strip to an array of sums */
void h_reduceLI_strip( long long *d_in, int nx, int ny, int nl, long long *gmean, struct d_xcorr d_xc){
	//long long *d_odata, *h_odata;// in[65536],sum=0;
	int thrds=512;
	int blcks=nx*ny*nl/(thrds*2);
	int mstride1;

	reduce< long long>(nl*nx*ny, thrds, blcks, d_in, d_xc.d_ol, 0);

	gErrCk(cudaMemcpy(d_xc.h_ol, d_xc.d_ol, blcks*sizeof( long long)*3, cudaMemcpyDeviceToHost));

	for (int j=0; j<nl; j++){

		mstride1 = (blcks * 3 * j)/nl;
		gmean[j]=0;
		long long tmp;
		for (int ind=0; ind<(blcks/nl); ind++)
		{
			tmp = d_xc.h_ol[mstride1 + ind*3];
			gmean[j] += tmp;
		}
	}
}

/* reduces a long long array strip to an array of mean values */
void h_reduceFD_strip(float *d_in, int nx, int ny, int nl, double *gmean, struct d_xcorr d_xc){
	int thrds=512;
	int blcks=nl*nx*ny/(thrds*2);
	int j, mstride1;

	reduce<float>(nl*nx*ny, thrds, blcks, d_in, d_xc.d_of, 0);

	gErrCk(cudaMemcpy(d_xc.h_of, d_xc.d_of, blcks*sizeof(float)*3, cudaMemcpyDeviceToHost));


	for (j=0; j<nl; j++){

		mstride1 = (blcks * 3 * j)/nl;
		gmean[j]=0;

		float tmp;
		for (int ind=0; ind<(blcks/nl); ind++)
		{
			tmp=d_xc.h_of[mstride1 + ind*3];
			gmean[j] += tmp;
		}
		gmean[j]/=nx*ny;
	}
}

/* reduces a complex array strip to a float array of sums and maximum values interleaved *
 * calculates mean value and returns peak value location per patch in the strip     */
void h_reduceCr_strip(cufftComplex *d_in, int nx, int ny, int nl, float *gmax, int *ip, int *jp, struct d_xcorr d_xc){
	int thrds=1024;
	int blcks=nl*nx*ny/(thrds*2);
	int j, mstride1;

	reduceC<float>(nl*nx*ny, thrds, blcks, d_in, d_xc.d_oc, 1);

	gErrCk(cudaMemcpy(d_xc.h_oc, d_xc.d_oc, blcks*sizeof(float)*3, cudaMemcpyDeviceToHost));

	for (j=0; j<nl; j++){
		mstride1 = (blcks * 3 * j)/nl;
		gmax[j]=-1;
		int peak=0;
		for (int ind=0; ind<(blcks/nl); ind++)
		{
			if(d_xc.h_oc[mstride1+ind*3+1] > (gmax[j])){
				gmax[j] = d_xc.h_oc[mstride1+ind*3+1];
				peak = d_xc.h_oc[mstride1+ind*3+2];
			}
		}

		peak -=nx*ny*j;

		ip[j]=floor(peak/nx);
		jp[j]=peak%nx;
	}
}
