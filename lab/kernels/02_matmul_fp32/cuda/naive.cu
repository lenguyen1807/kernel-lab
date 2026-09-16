#include <torch/extension.h>

#include "../../utils.h"

constexpr int WARP_SIZE = 32;

// A: M x K
// B: K x N
// C: M x N
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < M && col < N) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A, torch::kFloat32)
  CHECK_INPUT(B, torch::kFloat32)
  CHECK_SAME_DEVICE(A, B)

  CHECK_MATRIX(A)
  CHECK_MATRIX(B)
  TORCH_CHECK(A.size(1) == B.size(0),
              "A.shape[1] must equal B.shape[0]");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = create_matrix({M, N}, A);

  /*
  - We can use this API to find block size for maximum occupancy
  - This does not mean it is the most optimal in terms of FLOPs
  reference:
  https://github.com/gau-nernst/learn-cuda/blob/main/02a_matmul_simt/matmul.cu
   */

  static const int block_size_total = [] {
    int block_size, min_grid_size; // min_grid_size unused
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       matmul_naive_kernel, 0, 0);
    return block_size;
  }();

  dim3 blockDim(WARP_SIZE, block_size_total / WARP_SIZE, 1);
  dim3 gridDim((N + WARP_SIZE - 1) / WARP_SIZE,
               (M + blockDim.y - 1) / blockDim.y, 1);

  matmul_naive_kernel<<<gridDim, blockDim>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  CUDA_CHECK(cudaGetLastError());
  return C;
}
