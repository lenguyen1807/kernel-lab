#include <torch/extension.h>

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled_16(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled_32(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_coarsening_16x4(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_coarsening_32x4(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_naive", &matmul_naive, "Matmul naive CUDA");
  m.def("matmul_tiled_16", &matmul_tiled_16,
        "Matmul tiling with TILE_WIDTH=16 CUDA");
  m.def("matmul_tiled_32", &matmul_tiled_32,
        "Matmul tiling with TILE_WIDTH=32 CUDA");
  m.def("matmul_coarsening_16x4", &matmul_coarsening_16x4,
        "Matmul coarsening with TILE_WIDTH=16 and COARSE_FACTOR=4 CUDA");
  m.def("matmul_coarsening_32x4", &matmul_coarsening_32x4,
        "Matmul coarsening with TILE_WIDTH=32 and COARSE_FACTOR=4 CUDA");
}
