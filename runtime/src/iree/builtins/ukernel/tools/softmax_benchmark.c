// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/softmax_internal.h"
#include "iree/builtins/ukernel/tools/benchmark.h"
#include "iree/builtins/ukernel/tools/util.h"

static iree_status_t iree_uk_benchmark_softmax(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_uk_benchmark_user_data_t* user_data = benchmark_def->user_data;
  const iree_uk_softmax_params_t* src_params = iree_uk_benchmark_params(user_data);
  iree_uk_softmax_params_t params;
  memcpy(&params, src_params, sizeof params);
  params.cpu_data = iree_uk_benchmark_cpu_data(user_data);

  iree_uk_int32_t M = params.M;
  iree_uk_int32_t N = params.N;
  iree_uk_softmax_type_t softmax_type = iree_uk_softmax_type(params.flags);
  iree_uk_index_t buffer_size = iree_uk_2d_buffer_length(softmax_type, M, N);
  void *src_buffer = malloc(buffer_size);
  void *dst_buffer = malloc(buffer_size);
  iree_uk_random_engine_t* engine = iree_uk_benchmark_random_engine(user_data);
  // It's just about plausible that on some platform, for some number type,
  // performance might be different on zero buffers vs random buffers. But it
  // shouldn't matter that we recreate the random engine every time, getting
  // the same random values again.
  iree_uk_write_random_buffer(src_buffer, buffer_size, softmax_type, engine);

  params.src_buffer = src_buffer;
  params.dst_buffer = dst_buffer;
  int64_t total_iterations = 0;
  int64_t batch_count = 1;
  while (iree_benchmark_keep_running(benchmark_state, batch_count)) {
    for (int i = 0; i < batch_count; ++i) {
      iree_uk_softmax(&params);
    }
    total_iterations += batch_count;
    batch_count *= 2;
  }

  iree_benchmark_set_bytes_processed(benchmark_state,
                                    total_iterations * buffer_size);

  free(src_buffer);
  free(dst_buffer);
  return iree_ok_status();
}

static void iree_uk_benchmark_register_softmax(iree_uk_uint32_t flags,
                                            int M, int N,
                                            const char* cpu_features) {
  iree_uk_softmax_type_t type = iree_uk_softmax_type(flags);
  char type_str[32];
  iree_uk_type_str(type_str, sizeof type_str, type);
  iree_uk_softmax_params_t params = {.M = M,
                                  .N = N};

  char name[128];
  snprintf(name, sizeof name, "softmax_%s_tile_%dx%d", type_str,
           M, N);
  params.flags = flags;
  iree_uk_benchmark_register(name, iree_uk_benchmark_softmax, &params,
                             sizeof params, cpu_features);
}

int main(int argc, char** argv) {
  iree_flags_set_usage("softmax_benchmark", "");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_uk_benchmark_initialize(&argc, argv);

#if defined(IREE_ARCH_RISCV_64)
  iree_uk_benchmark_register_softmax(IREE_UK_FLAG_SOFTMAX_TYPE_F32, 8, 128, "");
#else
  // Architectures on which we do not have any optimized ukernel code.
  // Benchmark some arbitrary tile shape.
  iree_uk_benchmark_register_softmax(IREE_UK_FLAG_SOFTMAX_TYPE_F32, 8, 128, "");
#endif  // defined(IREE_ARCH_RISCV_64)

  iree_uk_benchmark_run_and_cleanup();
}
