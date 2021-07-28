/***************************************************************************/
/*  Manages CUDA and CUFFT errors for gmtsar-gpu				           *
 *  created by Diego Romano 2021-07-27									   *
 **************************************************************************/
#ifndef GPU_ERR
#define GPU_ERR
#include <iostream>
#include <cuda.h>
#include "helper_cuda.h"

using namespace std;

#define gErrCk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		cerr<<"GPU error: "<<cudaGetErrorString(code)<<" in "<<file<<" at line "<<line<<endl;
		if (abort) exit(code);
	}
}

#define gfftErrCk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cufftResult code, const char *file, int line, bool abort=true)
{

	if (code != CUFFT_SUCCESS)
	{
		cerr<<"CUFFT error: "<<_cudaGetErrorEnum(code)<<" in "<<file<<" at line "<<line<<endl;
		if (abort) exit(code);
	}
}

#endif /* GPU_ERR */
