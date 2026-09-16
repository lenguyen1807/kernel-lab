#include <cstddef>
#include <torch/extension.h>

#include "../../utils.h"

template <size_t TILE_WIDTH>
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
  // allocate shared memory
  __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

  // some convenient variables
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // row and col of ouptut (in shared memory or tile not the whole matrix)
  int row = ty + by * TILE_WIDTH;
  int col = tx + bx * TILE_WIDTH;

  float sum = 0.f;

  // ph is tile index
  // we are looping in tile
  int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int ph = 0; ph < num_tiles; ++ph) {
    /* --- Phase 1: Loading --- */

    // first we load from A
    int A_tile_row = row;
    int A_tile_col = ph * TILE_WIDTH + tx;

    // then we load from B
    int B_tile_row = ph * TILE_WIDTH + ty;
    int B_tile_col = col;

    // load element from HBM to shared memory
    if (A_tile_row < M && A_tile_col < K) {
      A_shared[ty][tx] = A[A_tile_row * K + A_tile_col];
    } else {
      A_shared[ty][tx] = 0.f;
    }

    if (B_tile_row < K && B_tile_col < N) {
      B_shared[ty][tx] = B[B_tile_row * N + B_tile_col];
    } else {
      B_shared[ty][tx] = 0.f;
    }

    // wait for all threads to finish loading
    __syncthreads();

    /* --- Phase 2: Computation --- */
    for (int k = 0; k < TILE_WIDTH; ++k) {
      sum += A_shared[ty][k] * B_shared[k][tx];
    }

    // wait for all threads to finish computing
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

template <size_t TILE_WIDTH>
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
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

  // We usually set block size equals tile_width
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  matmul_tiled_kernel<TILE_WIDTH><<<gridDim, blockDim>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  CUDA_CHECK(cudaGetLastError());

  return C;
}

torch::Tensor matmul_tiled_16(torch::Tensor A, torch::Tensor B) {
  return matmul_tiled<16>(A, B);
}

// note that we cannot use larger block size (or tile width)
// because 32x32=1024 is the maximum number of threads (in 5090)
torch::Tensor matmul_tiled_32(torch::Tensor A, torch::Tensor B) {
  return matmul_tiled<32>(A, B);
}
