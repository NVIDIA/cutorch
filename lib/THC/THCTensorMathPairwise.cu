#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

template <typename T>
struct TensorAddConstantOp {
  typedef THCNumerics<T> N_;
  TensorAddConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::s_(N_::add(*in, val));
  }
  __device__ __forceinline__ void operator()(T* v) {
        this->operator()(v, v);
  }
  const typename N_::storage_type val;
};

template <typename T>
struct TensorSubConstantOp {
  typedef THCNumerics<T> N_;
  TensorSubConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::s_(N_::sub(*in, val));
  }
  __device__ __forceinline__ void operator()(T* v) {
        this->operator()(v, v);
  }
  const typename N_::storage_type val;
};


template <typename T>
struct TensorMulConstantOp {
  TensorMulConstantOp(T v) : val(v) {}
  typedef THCNumerics<T> N_;
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::s_(N_::mul(*in, val));
  }
  __device__ __forceinline__ void operator()(T* v) {
        this->operator()(v, v);
  }
  const typename N_::storage_type val;
};

template <typename T>
struct TensorDivConstantOp {
  typedef THCNumerics<T> N_;
  TensorDivConstantOp(const T& v) : val(N_::div(N_::Constants::one(), v)) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = N_::s_(N_::div(*in, val));
  }
  __device__ __forceinline__ void operator()(T* v) {
        this->operator()(v, v);
  }
  const typename N_::storage_type val;
};

template <int Upper>
struct TensorTriOp {
  TensorTriOp(float *start_, long stride0_, long stride1_, long k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__ int mask(float *in) {
    ptrdiff_t n = in - start;
    long row, col;
    if (stride0 > stride1)
    {
      row = (long) (n / stride0);
      col = (long) ((n % stride0) / stride1);
    }
    else
    {
      row = (long) ((n % stride1) / stride0);
      col = (long) (n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = mask(in) ? *in : 0;
  }

  __device__ __forceinline__ void operator()(float* v) {
    if (!mask(v))
      *v = 0;
  }

  const float *start;
  const long stride0, stride1, k;
};

void THCudaTensor_tril(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<0> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_triu(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(cudaGetLastError());
}

#include "generic/THCTensorMathPairwise.cu"
#include "THCGenerateAllTypes.h"

// Copy the kth diagonal of a matrix B to a vector A.
__global__ void THCudaTensor_copyFromDiagonal(float* a, float* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideA) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
__global__ void THCudaTensor_copyToDiagonal(float* a, float* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideB) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

void THCudaTensor_diag(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k){
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  int nDimension = THCudaTensor_nDimension(state, src_);
  THArgCheck((nDimension == 2) || (nDimension == 1), 1, "expected a matrix or a vector");
  if (nDimension == 2) {
    long stride0 = THCudaTensor_stride(state, src_, 0);
    long stride1 = THCudaTensor_stride(state, src_, 1);
    long size0 = THCudaTensor_size(state, src_, 0);
    long size1 = THCudaTensor_size(state, src_, 1);
    long size = (k > 0) ? min((long long)size0, (long long)size1 - k) : min((long long)size0 + k, (long long)size1);
    THCudaTensor_resize1d(state, self_, size);
    long strideSelf = THCudaTensor_stride(state, self_, 0);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (long)threads.x)));
    long start = (k >= 0 ? k * stride1 : -k * stride0);
    THCudaTensor_copyFromDiagonal<<<grid, threads, 0, THCState_getCurrentStream(state)>>>
    (THCudaTensor_data(state, self_), THCudaTensor_data(state, src_), start, size, stride0 + stride1, strideSelf);
  } else {
    ptrdiff_t totalElements = THCudaTensor_nElement(state, src_);
    ptrdiff_t size = (k > 0) ? totalElements + k : totalElements - k;
    long strideSrc = THCudaTensor_stride(state, src_, 0);
    THCudaTensor_resize2d(state, self_, size, size);
    THCudaTensor_zero(state, self_);
    long stride0 = THCudaTensor_stride(state, self_, 0);
    long stride1 = THCudaTensor_stride(state, self_, 1);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (ptrdiff_t)threads.x)));
    ptrdiff_t start = (k >= 0 ? k * stride1 : -k * stride0);
    THCudaTensor_copyToDiagonal<<<grid, threads, 0, THCState_getCurrentStream(state)>>>
    (THCudaTensor_data(state, self_), THCudaTensor_data(state, src_), start, totalElements, stride0 + stride1, strideSrc);
  }
  THCudaCheck(cudaGetLastError());
}

float THCudaTensor_trace(THCState *state, THCudaTensor *src_) {
  THAssert(THCudaTensor_checkGPU(state, 1, src_));
  THArgCheck((src_->nDimension == 2), 1, "expected a matrix");
  THCudaTensor *diag = THCudaTensor_new(state);
  THCudaTensor_diag(state, diag, src_, 0);
  float trace = THCudaTensor_sumall(state, diag);
  THCudaTensor_free(state, diag);
  return trace;
}
