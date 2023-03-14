#include "uCUDAGemmCutlass.cuh"

__global__ void iree_kernel_microkernel(float* lhs, float* rhs, float* res,
                                        int64_t lhs_dim2, int64_t rhs_dim2) {
  gemm_ukernel<float, float, float, 128, 256, 32, 64, 64, 16, 8, 8, 3, true,
               true>(lhs, 0, lhs_dim2, rhs, 0, rhs_dim2, res, 0, 0, 0, 0);
}
