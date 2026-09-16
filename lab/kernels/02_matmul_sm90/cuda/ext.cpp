#include <torch/extension.h>

torch::Tensor matmul_2d_coarsening(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_2D_coarsening", &matmul_2d_coarsening,
        "Matmul 2D block tiling (thread coarsening)");
}
