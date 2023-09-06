#include "iree/builtins/ukernel/softmax_internal.h"
#include <stddef.h>
#include <limits.h>
#include <float.h>

static double fabs1(double x) {
    if(x >= 0){
        return x;
    } else {
        return x*(-1);
    }
}

static double powerex(double x) {
    double a = 1.0, e = 0;
    int invert = x<0;
    x = fabs1(x);
    for (int n = 1; e != e + a ; ++n) {
        e += a;
        a = a * x / n;
    }
    return invert ? 1/e : e;
}

// The softmax inner function for last dimension
static void iree_uk_softmax_tile_float_1d(
    const float* IREE_UK_RESTRICT src_buffer,
    float* IREE_UK_RESTRICT dst_buffer,
    iree_uk_int32_t N) {
  float beta = 1.0;
  size_t length = N;
  const float *src = src_buffer;
  float *dst = dst_buffer;
  int c;

  // Find max element value which we'll use to ensure numerical stability
  // taking advantage of the following equality:
  // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
  float max = -FLT_MAX;
  for (c = 0; c < length; ++c) {
    max = max > src[c] ? max : src[c];
  }

  // Compute sum.
  float sum = 0.f;
  for (c = 0; c < length; ++c) {
    const float exp_c = powerex((double)((src[c] - max) * beta));
    dst[c] = exp_c;
    sum += exp_c;
  }

  const float reciprocal_sum = 1.0f / sum;

  // Compute result.
  for (c = 0; c < length; ++c) {
    dst[c] = dst[c] * reciprocal_sum;
  }

}

static void iree_uk_softmax_tile_generic(
    const void* IREE_UK_RESTRICT src_buffer,
    void* IREE_UK_RESTRICT dst_buffer,
    iree_uk_int32_t M,
    iree_uk_int32_t N) {
  int i;
  const float *src = src_buffer;
  float *dst = dst_buffer;
  iree_uk_int32_t offset;

  for (i = 0, offset = 0; i < M; i++, offset += N) {
    iree_uk_softmax_tile_float_1d(src + offset, dst + offset, N);
  }
}

static iree_uk_softmax_tile_func_t iree_uk_softmax_select_tile_func_generic(
    const iree_uk_softmax_params_t* params) {
  return iree_uk_softmax_tile_generic;
}

iree_uk_softmax_tile_func_t iree_uk_softmax_select_tile_func(
    const iree_uk_softmax_params_t* params) {
  iree_uk_softmax_tile_func_t arch_tile_func =
      iree_uk_softmax_select_tile_func_arch(params);
  if (arch_tile_func) return arch_tile_func;
  return iree_uk_softmax_select_tile_func_generic(params);
}
