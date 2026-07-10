#include <cstddef>
#include <torch/extension.h>

#include "../../utils.h"

template <size_t TILE_WIDTH>
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
  // allocate shared memory
  __shared__ float A_shmem[TILE_WIDTH * TILE_WIDTH];
  __shared__ float B_shmem[TILE_WIDTH * TILE_WIDTH];

  // some convenient variables
  int bCol = blockIdx.x;
  int bRow = blockIdx.y;
  int tCol = threadIdx.x;
  int tRow = threadIdx.y;

  /*
   * Instead of traversing using a global index
   * we advance pointer to current row (for A)
   * or current column (for B)
   * and then compute tile from there
   */
  A += bRow * TILE_WIDTH * K;
  B += bCol * TILE_WIDTH;
  C += bRow * TILE_WIDTH * N + bCol * TILE_WIDTH;

  float sum = 0.f;

  // the outer loop advances A along the columns and B along
  for (int ph = 0; ph < K; ph += TILE_WIDTH) {
    // load tile into shared memory
    if ((tRow + bRow * TILE_WIDTH < M) && (ph + tCol < K)) {
      A_shmem[tRow * TILE_WIDTH + tCol] = A[tRow * K + tCol];
    } else {
      A_shmem[tRow * TILE_WIDTH + tCol] = 0.f;
    }

    if ((ph + tRow < K) && (tCol + bCol * TILE_WIDTH < N)) {
      B_shmem[tRow * TILE_WIDTH + tCol] = B[tRow * N + tCol];
    } else {
      B_shmem[tRow * TILE_WIDTH + tCol] = 0.f;
    }
    __syncthreads();

    // compute dot product of tile
    for (int k = 0; k < TILE_WIDTH; ++k) {
      sum += A_shmem[tRow * TILE_WIDTH + k] * B_shmem[k * TILE_WIDTH + tCol];
    }
    __syncthreads();

    // advance pointers onto next chunk
    A += TILE_WIDTH;
    B += TILE_WIDTH * N;
  }

  if ((tRow + bRow * TILE_WIDTH < M) && (tCol + bCol * TILE_WIDTH < N)) {
    C[tRow * N + tCol] = sum;
  }
}

template <size_t TILE_WIDTH>
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A, torch::kFloat32)
  CHECK_INPUT(B, torch::kFloat32)

  CHECK_MATRIX(A)
  CHECK_MATRIX(B)
  TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = create_matrix({M, N});

  // We usually set block size equals tile_width
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  matmul_tiled_kernel<TILE_WIDTH><<<gridDim, blockDim>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  CUDA_CHECK(cudaGetLastError());

  return C;
}

// note that we cannot use larger block size (or tile width)
// because 32x32=1024 is the maximum number of threads (in 5090)
torch::Tensor matmul_tiled_32(torch::Tensor A, torch::Tensor B) {
  return matmul_tiled<32>(A, B);
}
