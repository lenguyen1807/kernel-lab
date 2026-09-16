#include <cstddef>
#include <torch/extension.h>

#include "../../utils.h"

template <size_t BLOCK_M, size_t BLOCK_N, size_t BLOCK_K, size_t TM>
__global__ void matmul_1D_coarsening_kernel(const float *A, const float *B,
                                            float *C, int M, int N, int K) {
  // allocate shared memory
  __shared__ float A_shmem[BLOCK_M * BLOCK_K];
  __shared__ float B_shmem[BLOCK_K * BLOCK_N];

  int tid = threadIdx.x;

  // some convenient variables
  int bCol = blockIdx.x;
  int bRow = blockIdx.y;

  // Row and column of output C
  int tCol = tid % BLOCK_N;
  int tRow = tid / BLOCK_N;

  // advance pointer
  A += bRow * BLOCK_M * K;
  B += bCol * BLOCK_N;
  C += bRow * BLOCK_M * N + bCol * BLOCK_N;

  // Row and column of tile A and B
  const int tileColA = tid % BLOCK_K;
  const int tileRowA = tid / BLOCK_K;
  const int tileColB = tid % BLOCK_N;
  const int tileRowB = tid / BLOCK_N;

  // each thread will compute TM elements
  float sum[TM] = {0.f};

  for (int ph = 0; ph < K; ph += BLOCK_K) {
    // populate the SMEM caches (same as before)
    if ((tileRowA + bRow * BLOCK_M < M) && (ph + tileColA < K)) {
      A_shmem[tileRowA * BLOCK_K + tileColA] = A[tileRowA * K + tileColA];
    } else {
      A_shmem[tileRowA * BLOCK_K + tileColA] = 0.f;
    }
    if ((ph + tileRowB < K) && (tileColB + bCol * BLOCK_N < N)) {
      B_shmem[tileRowB * BLOCK_N + tileColB] = B[tileRowB * N + tileColB];
    } else {
      B_shmem[tileRowB * BLOCK_N + tileColB] = 0.f;
    }
    __syncthreads();

    // calculate per thread result
    for (int k = 0; k < BLOCK_K; ++k) {
      float Btmp = B_shmem[k * BLOCK_N + tCol];
      for (int i = 0; i < TM; ++i) {
        sum[i] += A_shmem[(tRow * TM + i) * BLOCK_K + k] * Btmp;
      }
    }
    __syncthreads();

    // advance pointer
    A += BLOCK_K;
    B += BLOCK_K * N;
  }

  // populate result
  for (int i = 0; i < TM; ++i) {
    int innerRow = tRow * TM + i;
    int innerCol = tCol;
    if ((innerRow + bRow * BLOCK_M < M) && (innerCol + bCol * BLOCK_N < N)) {
      C[innerRow * N + innerCol] = sum[i];
    }
  }
}

torch::Tensor matmul_1D_coarsening(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A, torch::kFloat32)
  CHECK_INPUT(B, torch::kFloat32)
  CHECK_SAME_DEVICE(A, B)

  CHECK_MATRIX(A)
  CHECK_MATRIX(B)
  TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = create_matrix({M, N}, A);

  const size_t TM = 8;
  const size_t BLOCK_M = 64;
  const size_t BLOCK_N = 64;
  const size_t BLOCK_K = 8;

  // total 512 threads
  dim3 blockDim((BLOCK_M * BLOCK_N) / TM);
  dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  matmul_1D_coarsening_kernel<BLOCK_M, BLOCK_N, BLOCK_K, TM>
      <<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                              C.data_ptr<float>(), M, N, K);

  CUDA_CHECK(cudaGetLastError());

  return C;
}
