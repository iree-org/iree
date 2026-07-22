// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/conv_nchwc.h"
#include "iree/builtins/ukernel/conv_nchwc_internal.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/tools/benchmark.h"
#include "iree/builtins/ukernel/tools/util.h"

IREE_FLAG(int32_t, n_size, 1, "Batch (N) of the conv.");
IREE_FLAG(int32_t, oc_size, 8, "Output-channel tiles (OC / k0).");
IREE_FLAG(int32_t, ic_size, 8, "Input-channel tiles (IC / c0).");
IREE_FLAG(int32_t, oh_size, 28, "Output height (OH).");
IREE_FLAG(int32_t, ow_size, 28, "Output width (OW).");
IREE_FLAG(int32_t, fh_size, 3, "Filter height (FH).");
IREE_FLAG(int32_t, fw_size, 3, "Filter width (FW).");
IREE_FLAG(bool, accumulate, false,
          "Whether the kernel should accumulate into the existing output, or "
          "overwrite it.");

static iree_status_t iree_uk_benchmark_conv_nchwc(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_uk_benchmark_user_data_t* user_data = benchmark_def->user_data;
  const iree_uk_conv_nchwc_params_t* src_params =
      iree_uk_benchmark_params(user_data);
  iree_uk_conv_nchwc_params_t params;
  memcpy(&params, src_params, sizeof params);
  params.cpu_data = iree_uk_benchmark_cpu_data(user_data);
  if (FLAG_accumulate) params.flags |= IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE;
  params.N = FLAG_n_size;
  params.OC_outer = FLAG_oc_size;
  params.IC_outer = FLAG_ic_size;
  params.OH = FLAG_oh_size;
  params.OW = FLAG_ow_size;
  params.FH = FLAG_fh_size;
  params.FW = FLAG_fw_size;

  iree_uk_index_t H = (params.OH - 1) * params.stride_h + params.FH;
  iree_uk_index_t W = (params.OW - 1) * params.stride_w + params.FW;
  params.input_stride_h = W * params.c0;
  params.input_stride_ic_outer = H * params.input_stride_h;
  params.input_stride_n = params.IC_outer * params.input_stride_ic_outer;
  params.filter_stride_fw = (iree_uk_index_t)params.c0 * params.k0;
  params.filter_stride_fh = params.FW * params.filter_stride_fw;
  params.filter_stride_ic_outer = params.FH * params.filter_stride_fh;
  params.filter_stride_oc_outer =
      params.IC_outer * params.filter_stride_ic_outer;
  params.output_stride_oh = params.OW * params.k0;
  params.output_stride_oc_outer = params.OH * params.output_stride_oh;
  params.output_stride_n = params.OC_outer * params.output_stride_oc_outer;

  iree_uk_conv_nchwc_type_t type = iree_uk_conv_nchwc_type(params.flags);
  iree_uk_type_t in_type = iree_uk_conv_nchwc_input_type(type);
  iree_uk_type_t f_type = iree_uk_conv_nchwc_filter_type(type);
  iree_uk_type_t out_type = iree_uk_conv_nchwc_output_type(type);
  iree_uk_index_t in_size =
      iree_uk_2d_buffer_length(in_type, params.N, params.input_stride_n);
  iree_uk_index_t f_size = iree_uk_2d_buffer_length(
      f_type, params.OC_outer, params.filter_stride_oc_outer);
  iree_uk_index_t out_size =
      iree_uk_2d_buffer_length(out_type, params.N, params.output_stride_n);
  void* in_buffer = malloc(in_size);
  void* f_buffer = malloc(f_size);
  void* out_buffer = malloc(out_size);
  iree_uk_random_engine_t* engine = iree_uk_benchmark_random_engine(user_data);
  iree_uk_write_random_buffer(in_buffer, in_size, in_type, engine);
  iree_uk_write_random_buffer(f_buffer, f_size, f_type, engine);
  iree_uk_write_random_buffer(out_buffer, out_size, out_type, engine);
  params.input_buffer = in_buffer;
  params.filter_buffer = f_buffer;
  params.output_buffer = out_buffer;

  int64_t total_iterations = 0;
  int64_t batch_count = 1;
  while (iree_benchmark_keep_running(benchmark_state, batch_count)) {
    for (int i = 0; i < batch_count; ++i) {
      iree_uk_conv_nchwc_p(&params);
    }
    total_iterations += batch_count;
    batch_count *= 2;
  }
  // FLOPs: 2 (multiply-add) per (output element x reduction element).
  iree_benchmark_set_items_processed(
      benchmark_state, total_iterations * 2 * params.N * params.OC_outer *
                           params.k0 * params.OH * params.OW * params.IC_outer *
                           params.c0 * params.FH * params.FW);
  free(in_buffer);
  free(f_buffer);
  free(out_buffer);
  return iree_ok_status();
}

static void iree_uk_benchmark_register_conv_nchwc(iree_uk_uint32_t flags,
                                                  int k0, int c0,
                                                  const char* cpu_features) {
  char type_str[32];
  iree_uk_conv_nchwc_type_t type = iree_uk_conv_nchwc_type(flags);
  iree_uk_type_triple_str(type_str, sizeof type_str, type);
  char name[128];
  iree_snprintf(name, sizeof name, "conv_nchwc_%s_tile_%dx%d", type_str, k0,
                c0);
  iree_uk_conv_nchwc_params_t params = {
      .flags =
          flags | IREE_UK_FLAG_CONV_NCHWC_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION,
      .k0 = k0,
      .c0 = c0,
      .stride_h = 1,
      .stride_w = 1};
  iree_uk_benchmark_register(name, iree_uk_benchmark_conv_nchwc, &params,
                             sizeof params, cpu_features);
}

int main(int argc, char** argv) {
  iree_flags_set_usage("conv_nchwc_benchmark", "");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_uk_benchmark_initialize(&argc, argv);

#if defined(IREE_ARCH_X86_64)
  iree_uk_benchmark_register_conv_nchwc(IREE_UK_FLAG_CONV_NCHWC_TYPE_F32F32F32,
                                        16, 16, "avx512_base");
#endif
  // Generic fallback (no CPU features), for comparison against the arch tile.
  iree_uk_benchmark_register_conv_nchwc(IREE_UK_FLAG_CONV_NCHWC_TYPE_F32F32F32,
                                        16, 16, "");

  iree_uk_benchmark_run_and_cleanup();
}
