#ifndef CUDA_STUB_HPP
#define CUDA_STUB_HPP

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#ifndef __CUDACC__

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __restrict__
#define __restrict__
#endif

typedef void* cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = std::malloc(size);
    if (*ptr) return cudaSuccess;
    return -1;
}

inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream = 0) {
    (void)stream;
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    (void)kind;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
    (void)kind;
    (void)stream;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline void atomicAdd(float* address, float val) {
    *address += val;
}

#endif // __CUDACC__

#endif // CUDA_STUB_HPP
