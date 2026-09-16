#include <torch/extension.h>

#include "../../utils.h"

__global__ void vec_add(const float *a, const float *b, float *c, int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
__global__ void vec_add_float4(const float *a, const float *b, float *c,
                               int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_chunks = N / 4;

  if (idx < total_chunks) {
    float4 a_val = reinterpret_cast<const float4 *>(a)[idx];
    float4 b_val = reinterpret_cast<const float4 *>(b)[idx];
    float4 c_val;

    c_val.x = a_val.x + b_val.x;
    c_val.y = a_val.y + b_val.y;
    c_val.z = a_val.z + b_val.z;
    c_val.w = a_val.w + b_val.w;

    reinterpret_cast<float4 *>(c)[idx] = c_val;
  }

  // launch a thread that clean up the remainder
  if (idx == 0) {
    for (int i = total_chunks * 4; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

torch::Tensor vecadd_cuda(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a, torch::kFloat32)
  CHECK_INPUT(b, torch::kFloat32)
  CHECK_SAME_DEVICE(a, b)

  TORCH_CHECK(a.dim() == 1, "a must be a vector");
  TORCH_CHECK(b.dim() == 1, "b must be a vector");
  TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same elements");

  auto y = torch::empty_like(a);

  int N = static_cast<int>(a.numel());

  int threads = 256;
  dim3 blockDim(threads, 1, 1);
  dim3 gridDim((N + threads - 1) / threads, 1, 1);

  vec_add<<<gridDim, blockDim>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                 y.data_ptr<float>(), N);
  CUDA_CHECK(cudaGetLastError());

  return y;
}

torch::Tensor vecadd_cuda_float4(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a, torch::kFloat32);
  CHECK_INPUT(b, torch::kFloat32);
  CHECK_SAME_DEVICE(a, b)

  CHECK_VECTOR(a)
  CHECK_VECTOR(b)

  TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same elements");

  auto y = torch::empty_like(a);

  int N = static_cast<int>(a.numel());

  int threads = 256;
  int chunks = N / 4;
  dim3 blockDim(threads, 1, 1);
  dim3 gridDim(std::max((chunks + threads - 1) / threads, 1), 1, 1);

  vec_add_float4<<<gridDim, blockDim>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N);
  CUDA_CHECK(cudaGetLastError());

  return y;
}