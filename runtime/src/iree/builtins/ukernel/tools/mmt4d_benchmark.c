// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/mmt4d_internal.h"
#include "iree/builtins/ukernel/tools/benchmark.h"
#include "iree/builtins/ukernel/tools/util.h"

IREE_FLAG(int32_t, m_size, 1,
          "M-dimension of mmt4d ops. The overall number of rows of the "
          "accumulator is that times the M0 tile size.");
IREE_FLAG(int32_t, n_size, 1,
          "N-dimension of mmt4d ops. The overall number of columns of the "
          "accumulator is that times the N0 tile size.");
IREE_FLAG(
    int32_t, k_size, 256,
    "K-dimension of mmt4d ops. That's the number of iterations of the inner "
    "loop. The overall accumulation depth is that times the K0 tile size.");
IREE_FLAG(bool, accumulate, false,
          "Whether the kernel should accumulate into the existing accumulator "
          "tile values, or zero the accumulator tile.");

static iree_status_t iree_uk_benchmark_mmt4d(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_uk_benchmark_user_data_t* user_data = benchmark_def->user_data;
  const iree_uk_mmt4d_params_t* src_params =
      iree_uk_benchmark_params(user_data);
  iree_uk_mmt4d_params_t params;
  memcpy(&params, src_params, sizeof params);
  params.cpu_data = iree_uk_benchmark_cpu_data(user_data);
  if (FLAG_accumulate) params.flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE;
  params.M = FLAG_m_size;
  params.N = FLAG_n_size;
  params.K = FLAG_k_size;
  params.lhs_stride0 = params.K * params.M0 * params.K0;
  params.rhs_stride0 = params.K * params.N0 * params.K0;
  params.out_stride0 = params.N * params.M0 * params.N0;
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params.flags);
  iree_uk_type_t lhs_type = iree_uk_mmt4d_lhs_type(mmt4d_type);
  iree_uk_type_t rhs_type = iree_uk_mmt4d_rhs_type(mmt4d_type);
  iree_uk_type_t out_type = iree_uk_mmt4d_out_type(mmt4d_type);
  iree_uk_index_t lhs_buffer_size =
      iree_uk_2d_buffer_length(lhs_type, params.M, params.lhs_stride0);
  iree_uk_index_t rhs_buffer_size =
      iree_uk_2d_buffer_length(rhs_type, params.N, params.rhs_stride0);
  iree_uk_index_t out_buffer_size =
      iree_uk_2d_buffer_length(out_type, params.M, params.out_stride0);
  void* lhs_buffer = malloc(lhs_buffer_size);
  void* rhs_buffer = malloc(rhs_buffer_size);
  void* out_buffer = malloc(out_buffer_size);
  iree_uk_random_engine_t* engine = iree_uk_benchmark_random_engine(user_data);
  // It's just about plausible that on some platform, for some number type,
  // performance might be different on zero buffers vs random buffers. But it
  // shouldn't matter that we recreate the random engine every time, getting
  // the same random values again.
  iree_uk_write_random_buffer(lhs_buffer, lhs_buffer_size, lhs_type, engine);
  iree_uk_write_random_buffer(rhs_buffer, rhs_buffer_size, rhs_type, engine);
  iree_uk_write_random_buffer(out_buffer, out_buffer_size, out_type, engine);
  params.lhs_buffer = lhs_buffer;
  params.rhs_buffer = rhs_buffer;
  params.out_buffer = out_buffer;
  int64_t total_iterations = 0;
  int64_t batch_count = 1;
  while (iree_benchmark_keep_running(benchmark_state, batch_count)) {
    for (int i = 0; i < batch_count; ++i) {
      iree_uk_mmt4d(&params);
    }
    total_iterations += batch_count;
    batch_count *= 2;
  }
  iree_benchmark_set_items_processed(
      benchmark_state, total_iterations * 2 * params.M * params.N * params.K *
                           params.M0 * params.N0 * params.K0);
  free(lhs_buffer);
  free(rhs_buffer);
  free(out_buffer);
  return iree_ok_status();
}

static void iree_uk_benchmark_register_mmt4d_impl(
    iree_uk_uint32_t flags, int M0, int N0, int K0, const char* cpu_features,
    const char* code_path_suffix) {
  char type_str[32];
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(flags);
  iree_uk_type_triple_str(type_str, sizeof type_str, mmt4d_type);
  char name[128];
  snprintf(name, sizeof name, "mmt4d_%s_tile_%dx%dx%d%s", type_str, M0, N0, K0,
           code_path_suffix);
  iree_uk_mmt4d_params_t params = {
      .flags = flags | IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS,
      .M0 = M0,
      .N0 = N0,
      .K0 = K0};
  iree_uk_benchmark_register(name, iree_uk_benchmark_mmt4d, &params,
                             sizeof params, cpu_features);
}

static void iree_uk_benchmark_register_mmt4d(iree_uk_uint32_t flags, int M0,
                                             int N0, int K0,
                                             const char* cpu_features) {
  // Test narrowed, power-of-two values of M0, as mmt4d kernels tend to have
  // narrow variants for handling these cases.
  for (int narrowM0 = 1; narrowM0 < M0; narrowM0 *= 2) {
    iree_uk_benchmark_register_mmt4d_impl(flags, narrowM0, N0, K0, cpu_features,
                                          "");
  }
  iree_uk_benchmark_register_mmt4d_impl(flags, M0, N0, K0, cpu_features, "");
}

int main(int argc, char** argv) {
  iree_flags_set_usage("mmt4d_benchmark", "");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_uk_benchmark_initialize(&argc, argv);

#if defined(IREE_ARCH_ARM_64)
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 8, 8, 1,
                                   "");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 8, 8, 1,
                                   "");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 8, 8, 1,
                                   "fp16fml");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 8, 8, 1,
                                   "");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 8, 8, 1,
                                   "fullfp16");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32, 8, 8, 4,
                                   "bf16");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16, 8, 8,
                                   4, "bf16");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 1,
                                   "");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 4,
                                   "dotprod");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 8,
                                   "i8mm");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S4S32, 4, 16, 2,
                                   "");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S4S32, 8, 4, 8,
                                   "dotprod");
#elif defined(IREE_ARCH_X86_64)
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 8, 8, 1,
                                   "avx2_fma");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 16, 16, 1,
                                   "avx512_base");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 8, 8, 1,
                                   "avx2_fma");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 16, 16, 1,
                                   "avx512_base");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 8, 8, 1,
                                   "avx2_fma");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 16, 16, 1,
                                   "avx512_base");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32, 16, 16,
                                   2, "avx512_bf16");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16, 16, 16,
                                   2, "avx512_bf16");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 2,
                                   "avx2_fma");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 16, 16, 2,
                                   "avx512_base");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 16, 16, 2,
                                   "avx512_vnni");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S16S32, 8, 8, 2,
                                   "avx2_fma");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S16S32, 16, 16, 2,
                                   "avx512_base");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S16S32, 16, 16, 2,
                                   "avx512_vnni");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16U4S32, 1, 32, 8,
                                   "avx512_vnni");
#else   // defined(IREE_ARCH_ARM_64)
  // Architectures on which we do not have any optimized ukernel code.
  // Benchmark some arbitrary tile shape.
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 8, 8, 1,
                                   "");
  iree_uk_benchmark_register_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 1,
                                   "");
#endif  // defined(IREE_ARCH_ARM_64)

  iree_uk_benchmark_run_and_cleanup();
}
