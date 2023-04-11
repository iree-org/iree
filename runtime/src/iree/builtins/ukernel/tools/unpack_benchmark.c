// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/tools/benchmark.h"
#include "iree/builtins/ukernel/tools/memcpy_benchmark.h"
#include "iree/builtins/ukernel/tools/util.h"

IREE_FLAG(int64_t, batch_min_traversal_size, 10000000,
          "Minimum number of bytes to be traversed in each batch.");

IREE_FLAG(
    int64_t, working_set_size, 10000,
    "Number of bytes to be traversed by the benchmark workload (input and "
    "output buffers together). Matrix shapes are computed accordingly.");
IREE_FLAG(
    int32_t, padding_size, 0,
    "Padding size (same value used for both dimensions, 0 means no padding)");

static iree_status_t iree_uk_benchmark_unpack(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_uk_benchmark_user_data_t* user_data = benchmark_def->user_data;
  const iree_uk_unpack_params_t* src_params =
      iree_uk_benchmark_params(user_data);
  iree_uk_unpack_params_t params;
  memcpy(&params, src_params, sizeof params);
  params.cpu_data = iree_uk_benchmark_cpu_data(user_data);
  iree_uk_type_t in_type = iree_uk_unpack_in_type(params.type);
  iree_uk_type_t out_type = iree_uk_unpack_out_type(params.type);
  iree_uk_ssize_t in_type_size = iree_uk_type_size(in_type);
  iree_uk_ssize_t out_type_size = iree_uk_type_size(out_type);

  // The inner dims 2, 3 are given to us as part of the benchmark user_data.
  // The outer dims 0, 1 are to be determined based on FLAG_working_set_size.
  iree_uk_ssize_t in_size0 = 1;
  iree_uk_ssize_t in_size1 = 1;
  iree_uk_ssize_t in_size2 = params.in_size2;
  iree_uk_ssize_t in_size3 = params.in_size3;
  int target_matrix_size_in_elems =
      FLAG_working_set_size / (in_type_size + out_type_size);
  int target_product_of_outer_sizes_0_1 =
      target_matrix_size_in_elems / (in_size2 * in_size3);
  while (target_product_of_outer_sizes_0_1 >= 4) {
    target_product_of_outer_sizes_0_1 /= 4;
    in_size0 *= 2;
    in_size1 *= 2;
  }
  in_size1 *= target_product_of_outer_sizes_0_1;
  params.in_size0 = in_size0;
  params.in_size1 = in_size1;
  if (params.flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER) {
    iree_uk_ssize_swap(&in_size0, &in_size1);
  }
  if (params.flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER) {
    iree_uk_ssize_swap(&in_size2, &in_size3);
  }
  params.out_size0 = iree_max(0, in_size0 * in_size2 - FLAG_padding_size);
  params.out_size1 = iree_max(0, in_size1 * in_size3 - FLAG_padding_size);
  params.out_stride0 = params.out_size1;
  params.in_stride0 = params.in_size1 * params.in_size2 * params.in_size3;
  iree_uk_ssize_t in_buffer_size =
      iree_uk_2d_buffer_length(in_type, params.in_size0, params.in_stride0);
  iree_uk_ssize_t out_buffer_size =
      iree_uk_2d_buffer_length(out_type, params.out_size0, params.out_stride0);
  void* in_buffer = malloc(in_buffer_size);
  void* out_buffer = malloc(out_buffer_size);
  iree_uk_random_engine_t* engine = iree_uk_benchmark_random_engine(user_data);
  // It's just about plausible that on some platform, for some number type,
  // performance might be different on zero buffers vs random buffers. But it
  // shouldn't matter that we recreate the random engine every time, getting
  // the same random values again.
  iree_uk_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  iree_uk_write_random_buffer(out_buffer, out_buffer_size, out_type, engine);
  params.in_buffer = in_buffer;
  params.out_buffer = out_buffer;
  int64_t total_iterations = 0;
  int64_t batch_count =
      (FLAG_batch_min_traversal_size + FLAG_working_set_size - 1) /
      FLAG_working_set_size;
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/batch_count)) {
    for (int i = 0; i < batch_count; ++i) {
      iree_uk_unpack(&params);
    }
    total_iterations += batch_count;
  }
  // Report bytes per second, so that can be easily compared to known memory
  // system performance metrics (e.g. RAM bandwidth, to tell whether this is
  // memory-bound).
  iree_benchmark_set_items_processed(benchmark_state,
                                     total_iterations * out_buffer_size);
  free(in_buffer);
  free(out_buffer);
  return iree_ok_status();
}

static void iree_uk_benchmark_register_unpack(iree_uk_unpack_type_t type,
                                              int tile_size0, int tile_size1,
                                              const char* cpu_features) {
  char type_str[32];
  iree_uk_type_pair_str(type_str, sizeof type_str, type);
  iree_uk_unpack_params_t params = {
      .type = type, .in_size2 = tile_size0, .in_size3 = tile_size1};
  typedef struct unpack_variant_t {
    const char* label;
    iree_uk_uint32_t flags;
  } unpack_variant_t;
  const unpack_variant_t variants[] = {
      {"trnone", 0},
      {"trinner", IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER},
      {"trouter", IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER},
      {"trboth", IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER |
                     IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER},
  };
  for (int i = 0; i < IREE_ARRAYSIZE(variants); ++i) {
    unpack_variant_t variant = variants[i];
    char name[128];
    snprintf(name, sizeof name, "unpack_%s_tile_%dx%d_%s_wss_%" PRIi64,
             type_str, tile_size0, tile_size1, variant.label,
             FLAG_working_set_size);
    params.flags = variant.flags;
    iree_uk_benchmark_register(name, iree_uk_benchmark_unpack, &params,
                               sizeof params, cpu_features);
  }
}

int main(int argc, char** argv) {
  iree_flags_set_usage("unpack_benchmark", "");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_uk_benchmark_initialize(&argc, argv);

  // The memcpy benchmark provides a useful comparison point, as pack is fairly
  // close to memory-bound.
  iree_uk_benchmark_register_memcpy(FLAG_working_set_size,
                                    FLAG_batch_min_traversal_size);

#if defined(IREE_UK_ARCH_ARM_64)
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_f32f32, 8, 8, NULL);
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_i32i32, 8, 8, NULL);
#elif defined(IREE_UK_ARCH_X86_64)
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_f32f32, 8, 8,
                                    "avx2_fma");
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_i32i32, 8, 8,
                                    "avx2_fma");
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_f32f32, 16, 16,
                                    "avx512_base");
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_i32i32, 16, 16,
                                    "avx512_base");
#else   // defined(IREE_UK_ARCH_ARM_64)
  // Architectures on which we do not have any optimized ukernel code.
  // Benchmark some arbitrary tile shape.
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_f32f32, 8, 8, NULL);
  iree_uk_benchmark_register_unpack(iree_uk_unpack_type_i32i32, 8, 8, NULL);
#endif  // defined(IREE_UK_ARCH_ARM_64)

  iree_uk_benchmark_run_and_cleanup();
}
