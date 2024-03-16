#pragma once
#include "platform.h"

#if IS_IN_IDE_PARSER
#ifdef __CUDACC__
static_assert(false, "Is in real compiler, but IDE parser is incorrectly detected.");
#endif
#define __CUDACC__ 1
#define __CUDA_ARCH__ 610
#define __CUDACC_VER_MAJOR__ 10
#define __CUDACC_VER_MINOR__ 1
//#define LAUNCH_CONFIG(dimGrid, dimCta, smemBytes, stream)

//#include <cuda_runtime_api.h>
//extern "C" __device__ void __assert_fail (const char *__assertion, const char *__file,
//               unsigned int __line, const char *__function) throw();
#else
#endif

