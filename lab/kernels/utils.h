#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err__) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

#define CHECK_IS_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");
#define CHECK_DTYPE(x, expected_dtype)                                      \
  TORCH_CHECK(x.dtype() == expected_dtype,                                  \
              #x " must have type " #expected_dtype);
#define CHECK_MATRIX(x) \
  TORCH_CHECK(x.dim() == 2, #x " must be a matrix (dimension = 2)");
#define CHECK_VECTOR(x) \
  TORCH_CHECK(x.dim() == 1, #x " must be a vector (dimension = 1)");

#define CHECK_INPUT(x, dtype) \
  do {                        \
    CHECK_IS_CUDA(x);         \
    CHECK_CONTIGUOUS(x);      \
    CHECK_DTYPE(x, dtype);    \
  } while (0);

// Uninitialized output storage. Callers must overwrite every element (custom
// kernels) or use beta=0 (cuBLAS). Avoids a fill kernel that breaks single-
// kernel nsight annotations.
inline torch::Tensor create_matrix(const std::array<int64_t, 2>& shape) {
  return torch::empty(shape,
                      torch::device(torch::kCUDA).dtype(torch::kFloat32));
}
