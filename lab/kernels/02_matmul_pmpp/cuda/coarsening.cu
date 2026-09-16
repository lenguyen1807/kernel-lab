#include "../../utils.h"
#include <torch/extension.h>

template <size_t TILE_WIDTH, size_t COARSE_FACTOR>
__global__ void matmul_coarsening(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
  // allocate shared memory
  __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

  // some convenient variables
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // row of output tile
  int row = ty + by * TILE_WIDTH;
  // col of output tile
  // each thread now will compute COARSE_FACTOR output elements
  // each seperate by TILE_WIDTH
  int col_start = tx + bx * TILE_WIDTH * COARSE_FACTOR;

  // COARSE_FACTOR output tiles
  float output[COARSE_FACTOR] = {0.f};

  // now compute
  int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int ph = 0; ph < num_tiles; ++ph) {
    // we load matrix A *once*
    int A_tile_row = row;
    int A_tile_col = ph * TILE_WIDTH + tx;

    if (A_tile_row < M && A_tile_col < K) {
      A_shared[ty][tx] = A[A_tile_row * K + A_tile_col];
    } else {
      A_shared[ty][tx] = 0.f;
    }
    __syncthreads();

    // now we compute columns
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      // we then load matrix B
      int B_tile_row = ph * TILE_WIDTH + ty;
      int B_tile_col = col_start + c * TILE_WIDTH;

      if (B_tile_row < K && B_tile_col < N) {
        B_shared[ty][tx] = B[B_tile_row * N + B_tile_col];
      } else {
        B_shared[ty][tx] = 0.f;
      }
      __syncthreads();

      // finally, compute
      for (int k = 0; k < TILE_WIDTH; ++k) {
        output[c] += A_shared[ty][k] * B_shared[k][tx];
      }
      __syncthreads();
    }
  }

  // now we load computed tile to the output
  if (row >= M) {
      return;
  }
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int col = col_start + c * TILE_WIDTH;
    if (col < N) {
        C[row * N + col] = output[c];
    }
  }
}

template <size_t TILE_WIDTH, size_t COARSE_FACTOR>
torch::Tensor matmul_coarsening(torch::Tensor A, torch::Tensor B) {
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

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

  /*
   * Because each block now computes COARSE_FACTOR column output tile
   * instead of one so the number of blocks in y dimension should be:
   * ceil(N / TILE_WIDTH * COARSE_FACTOR)
   */
  dim3 gridDim((N + (TILE_WIDTH * COARSE_FACTOR) - 1) / (TILE_WIDTH * COARSE_FACTOR),
               (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  matmul_coarsening<TILE_WIDTH, COARSE_FACTOR><<<gridDim, blockDim>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  CUDA_CHECK(cudaGetLastError());

  return C;
}

torch::Tensor matmul_coarsening_16x4(torch::Tensor A, torch::Tensor B) {
    return matmul_coarsening<16, 4>(A, B);
}

torch::Tensor matmul_coarsening_32x4(torch::Tensor A, torch::Tensor B) {
    return matmul_coarsening<32, 4>(A, B);
}
