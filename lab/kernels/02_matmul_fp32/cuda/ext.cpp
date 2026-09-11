#include <torch/extension.h>

torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled_32(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_1D_coarsening(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_2d_coarsening(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_cublas", &matmul_cublas, "Matmul via explicit cublasSgemm");
  m.def("matmul_naive", &matmul_naive, "Matmul naive CUDA");
  m.def("matmul_tiled", &matmul_tiled_32,
        "Matmul tiling with TILE_WIDTH=32 CUDA");
  m.def("matmul_1D_coarsening", &matmul_1D_coarsening,
        "Matmul 1D block tiling (thread coarsening)");
  m.def("matmul_2D_coarsening", &matmul_2d_coarsening,
        "Matmul 2D block tiling (thread coarsening)");
}
