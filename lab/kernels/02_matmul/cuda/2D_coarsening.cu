#include <cstddef>
#include <torch/extension.h>

#include "../../utils.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void matmul_2d_coarsening_kernel(const float *A, const float *B,
                                            float *C, int M, int N, int K) {
  // allocate shared memory
  __shared__ float A_shmem[BLOCK_M * BLOCK_K];
  __shared__ float B_shmem[BLOCK_K * BLOCK_N];

  int tid = threadIdx.x;

  // A thread is responsible for calculating TM*TN elements in the blocktile
  // Here is how we calculate total threads for each block tile
  const int nThreadsTile = (BLOCK_M * BLOCK_N) / (TM * TN);

  // some convenient variables
  int bCol = blockIdx.x;
  int bRow = blockIdx.y;

  // advance pointer
  A += bRow * BLOCK_M * K;
  B += bCol * BLOCK_N;
  C += bRow * BLOCK_M * N + bCol * BLOCK_N;

  // Row and column of the TM x TN output patch computed by this thread.
  int tCol = tid % (BLOCK_N / TN);
  int tRow = tid / (BLOCK_N / TN);

  // Row and column of tile A and B
  const int tileColA = tid % BLOCK_K;
  const int tileRowA = tid / BLOCK_K;
  const int tileColB = tid % BLOCK_N;
  const int tileRowB = tid / BLOCK_N;

  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideA = nThreadsTile / BLOCK_K;
  const int strideB = nThreadsTile / BLOCK_N;

  // each thread will compute TM x TN elements
  float sum[TM * TN] = {0.f};
  // register caches for As and Bs
  float regA[TM] = {0.0};
  float regB[TN] = {0.0};

  // outer loop
  for (int ph = 0; ph < K; ph += BLOCK_K) {
    // populate the SMEM caches (same as before)
    for (int offset = 0; offset < BLOCK_M; offset += strideA) {
      int rowA = tileRowA + offset;
      int colA = tileColA;
      if ((rowA + bRow * BLOCK_M < M) && (ph + colA < K)) {
        A_shmem[rowA * BLOCK_K + colA] = A[rowA * K + colA];
      } else {
        A_shmem[rowA * BLOCK_K + colA] = 0.f;
      }
    }
    for (int offset = 0; offset < BLOCK_K; offset += strideB) {
      int rowB = tileRowB + offset;
      int colB = tileColB;
      if ((rowB + ph < K) && (colB + bCol * BLOCK_N < N)) {
        B_shmem[rowB * BLOCK_N + colB] = B[rowB * N + colB];
      } else {
        B_shmem[rowB * BLOCK_N + colB] = 0.f;
      }
    }
    __syncthreads();

    // calculate result
    for (int k = 0; k < BLOCK_K; ++k) {
      for (int i = 0; i < TM; ++i) {
        // load row of A tile to register
        regA[i] = A_shmem[(tRow * TM + i) * BLOCK_K + k];
      }
      for (int i = 0; i < TN; ++i) {
        // load column of B tile to register
        regB[i] = B_shmem[k * BLOCK_N + (tCol * TN + i)];
      }
      // now calculate result from register
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          sum[i * TN + j] += regA[i] * regB[j];
        }
      }
    }
    __syncthreads();

    // advance pointer
    A += BLOCK_K;
    B += BLOCK_K * N;
  }

  // populate result
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int innerRow = tRow * TM + i;
      int innerCol = tCol * TN + j;
      if ((innerRow + bRow * BLOCK_M < M) &&
          (innerCol + bCol * BLOCK_N < N)) {
        C[innerRow * N + innerCol] = sum[i * TN + j];
      }
    }
  }
}

torch::Tensor matmul_2d_coarsening(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A, torch::kFloat32)
  CHECK_INPUT(B, torch::kFloat32)

  CHECK_MATRIX(A)
  CHECK_MATRIX(B)
  TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = create_matrix({M, N});

  const size_t TN = 8;
  const size_t TM = 8;
  const size_t BLOCK_M = 128;
  const size_t BLOCK_N = 128;
  const size_t BLOCK_K = 8;

  // total 256 threads
  dim3 blockDim((BLOCK_M / TM) * (BLOCK_N / TN));
  dim3 gridDim(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M));

  matmul_2d_coarsening_kernel<BLOCK_M, BLOCK_N, BLOCK_K, TM, TN>
      <<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                              C.data_ptr<float>(), M, N, K);

  CUDA_CHECK(cudaGetLastError());

  return C;
}
