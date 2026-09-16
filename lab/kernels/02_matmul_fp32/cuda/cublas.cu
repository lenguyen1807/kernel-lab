#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <torch/extension.h>

#include "../../utils.h"

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status__ = (call);                                          \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                   \
      throw std::runtime_error(std::string("cuBLAS error: ") +                 \
                               std::to_string(status__) + " at " + __FILE__ +  \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

namespace {

cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }
  return handle;
}

} // namespace

// Explicit FP32 cuBLAS baseline via cublasSgemm.
// PyTorch stores matrices row-major; cuBLAS expects column-major, so we compute
// C = A @ B as C^T = B^T @ A^T with no explicit transposes.
torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B) {
  CHECK_INPUT(A, torch::kFloat32)
  CHECK_INPUT(B, torch::kFloat32)
  CHECK_SAME_DEVICE(A, B)
  CHECK_MATRIX(A)
  CHECK_MATRIX(B)
  TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");

  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int N = static_cast<int>(B.size(1));

  auto C = create_matrix({M, N}, A);

  const float alpha = 1.f;
  const float beta = 0.f;
  cublasHandle_t handle = get_cublas_handle();
  CUBLAS_CHECK(
      cublasSetStream(handle, c10::cuda::getCurrentCUDAStream().stream()));

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B.data_ptr<float>(), N, A.data_ptr<float>(), K,
                           &beta, C.data_ptr<float>(), N));
  return C;
}
