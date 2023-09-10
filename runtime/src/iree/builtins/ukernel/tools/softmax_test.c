// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>

#include "iree/base/api.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/tools/test.h"
#include "iree/builtins/ukernel/tools/util.h"
#include "iree/builtins/ukernel/softmax_internal.h"

#define IREE_UK_SOFTMAX_RELATIVE_ERROR 0x1p-14

static void iree_softmax_reference(const iree_uk_softmax_params_t* params) {
  const float *src_buffer = params->src_buffer;
  float *dst_buffer = params->dst_buffer;
  int M = params->M, N = params->N;
  int i, j;
  double sum;

  for (i = 0; i < M; i++) {
    sum = 0;
    for (j = 0; j < N; j++) {
      double e = exp(src_buffer[i * N + j]);
      sum += e;
      dst_buffer[i * N + j] = e;
    }
    for (j = 0; j < N; j++) {
      dst_buffer[i * N + j] = dst_buffer[i * N + j] / sum;
    }
  }
}

static void iree_uk_test_softmax_for_params(iree_uk_test_t* test,
                                                const void* src_params) {
  iree_uk_softmax_params_t params;
  memcpy(&params, src_params, sizeof(params));
  iree_uk_softmax_params_t reference_params;
  memcpy(&reference_params, src_params, sizeof(params));

  iree_uk_int32_t M = params.M;
  iree_uk_int32_t N = params.N;
  iree_uk_int32_t buffer_size = M * N;
  float* src_buffer = malloc(buffer_size * sizeof(float));
  float* dst_buffer = malloc(buffer_size * sizeof(float));
  float* reference_dst_buffer = malloc(buffer_size * sizeof(float));
  int i, j;

  // Set all elements in last dimension to 1, the output of each element will be 1/N
  iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      src_buffer[i * M + j] = iree_uk_random_engine_get_minus16_plus15(engine);
    }
  }

  params.src_buffer = src_buffer;
  params.dst_buffer = dst_buffer;
  reference_params.src_buffer = src_buffer;
  reference_params.dst_buffer = reference_dst_buffer;

  iree_uk_softmax(&params);
  iree_softmax_reference(&reference_params);

  if (!iree_uk_buffers_equal_f32(dst_buffer, reference_dst_buffer,
        buffer_size, IREE_UK_SOFTMAX_RELATIVE_ERROR)) {
    IREE_UK_TEST_FAIL(test);
  }

  free(src_buffer);
  free(dst_buffer);
  free(reference_dst_buffer);
}

static void iree_uk_test_softmax(iree_uk_uint32_t flags, int M,
                                int N, const char* cpu_features) {
  iree_uk_softmax_params_t params = {
      .flags = flags, .M = M, .N = N};

  char types_str[32];
  iree_uk_softmax_type_t softmax_type = iree_uk_softmax_type(flags);
  iree_uk_type_str(types_str, sizeof types_str, softmax_type);
  char test_label_str[256];
  snprintf(test_label_str, sizeof test_label_str, "types:%s tile:%dx%d",
      types_str, M, N);
  iree_uk_test(test_label_str, iree_uk_test_softmax_for_params, &params, cpu_features);
}


int main(int argc, char** argv) {
  // Generic tests, not matching any particular CPU feature. This is the place
  // to test weird tile shapes to ensure e.g. that we haven't unwittingly baked
  // in a power-of-two assumption
  iree_uk_test_softmax(IREE_UK_FLAG_SOFTMAX_TYPE_F32, 3, 5, "");

#if defined(IREE_ARCH_RISCV_64)
  iree_uk_test_softmax(IREE_UK_FLAG_SOFTMAX_TYPE_F32, 3, 5, "rvv");
#endif  // defined(IREE_ARCH_RISCV_64)
  return iree_uk_test_exit_status();
}
