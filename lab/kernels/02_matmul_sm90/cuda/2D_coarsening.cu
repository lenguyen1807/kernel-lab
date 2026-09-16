#include <cstddef>
#include <torch/extension.h>

#include "../../utils.h"
#include "cuda_bf16.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// we upgrade this SIMT kernel with BF16 storage and FP32 arithmetic
// note that B is transpose (column-major)
// the output C is still row-major
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void matmul_2d_coarsening_kernel(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    __nv_bfloat16 *C,
    int M, int N, int K) {
  // allocate shared memory
  __shared__ __nv_bfloat16 A_shmem[BLOCK_M * BLOCK_K];
  __shared__ __nv_bfloat16 B_shmem[BLOCK_K * BLOCK_N];

  int tid = threadIdx.x;

  // A thread is responsible for calculating TM*TN elements in the blocktile
  // Here is how we calculate total threads for each block tile
  const int nThreadsTile = (BLOCK_M * BLOCK_N) / (TM * TN);

  // some convenient variables
  int bCol = blockIdx.x;
  int bRow = blockIdx.y;

  // advance pointer
  A += bRow * BLOCK_M * K;
  B += bCol * BLOCK_N * K;
  C += bRow * BLOCK_M * N + bCol * BLOCK_N;

  // Row and column of the TM x TN output patch computed by this thread.
  int tCol = tid % (BLOCK_N / TN);
  int tRow = tid / (BLOCK_N / TN);

  // Row and column of tile A and B
  const int tileCol = tid % BLOCK_K;
  const int tileRow = tid / BLOCK_K;

  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int stride = nThreadsTile / BLOCK_K;

  // each thread will compute TM x TN elements
  float sum[TM * TN] = {0.f};

  // register caches for As and Bs
  __nv_bfloat16 regA[TM];
  __nv_bfloat16 regB[TN];

  // outer loop
  for (int ph = 0; ph < K; ph += BLOCK_K) {
    // populate the SMEM caches (same as before)
    for (int offset = 0; offset < BLOCK_M; offset += stride) {
      int row = tileRow + offset;
      int col = tileCol;
      if ((row + bRow * BLOCK_M < M) && (ph + col < K)) {
        A_shmem[row * BLOCK_K + col] = A[row * K + col];
      } else {
        A_shmem[row * BLOCK_K + col] = __float2bfloat16(0.f);
      }
    }
    for (int offset = 0; offset < BLOCK_N; offset += stride) {
      int row = tileRow + offset;
      int col = tileCol;
      if ((row + bCol * BLOCK_N < N) && (ph + col < K)) {
        B_shmem[row * BLOCK_K + col] = B[row * K + col];
      } else {
        B_shmem[row * BLOCK_K + col] = __float2bfloat16(0.f);
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
        regB[i] = B_shmem[(tCol * TN + i) * BLOCK_K + k];
      }
      // now calculate result from register
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          sum[i * TN + j] += __bfloat162float(regA[i] * regB[j]);
        }
      }
    }
    __syncthreads();

    // advance pointer
    A += BLOCK_K;
    B += BLOCK_K;
  }

  // populate result
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int innerRow = tRow * TM + i;
      int innerCol = tCol * TN + j;
      if ((innerRow + bRow * BLOCK_M < M) &&
          (innerCol + bCol * BLOCK_N < N)) {
        C[innerRow * N + innerCol] = __float2bfloat16(sum[i * TN + j]);
      }
    }
  }
}

torch::Tensor matmul_2d_coarsening(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A, torch::kBFloat16)
  CHECK_CONTIGUOUS(A)
  CHECK_INPUT(B, torch::kBFloat16)
  CHECK_SAME_DEVICE(A, B)

  CHECK_MATRIX(A)
  CHECK_MATRIX(B)
  TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = create_matrix({M, N}, A);

  const size_t TN = 8;
  const size_t TM = 8;
  const size_t BLOCK_M = 128;
  const size_t BLOCK_N = 128;
  const size_t BLOCK_K = 8;

  // total 256 threads
  dim3 blockDim((BLOCK_M / TM) * (BLOCK_N / TN));
  dim3 gridDim(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M));

  matmul_2d_coarsening_kernel<BLOCK_M, BLOCK_N, BLOCK_K, TM, TN>
      <<<gridDim, blockDim>>>(reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<torch::BFloat16>()),
                              reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<torch::BFloat16>()),
                              reinterpret_cast<__nv_bfloat16*>(C.data_ptr<torch::BFloat16>()),
                              M, N, K);

  CUDA_CHECK(cudaGetLastError());

  return C;
}
