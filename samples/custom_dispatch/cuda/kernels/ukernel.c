#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __device__ void simple_mul_workgroup(float *lhs, size_t lhs_offset,
                                                float *rhs, size_t rhs_offset,
                                                float *result,
                                                size_t result_offset,
                                                size_t size) {
  int threadId = threadIdx.x;
  if (threadId < size) {
    result[result_offset + threadId] =
        lhs[lhs_offset + threadId] * rhs[rhs_offset + threadId];
  }
}
