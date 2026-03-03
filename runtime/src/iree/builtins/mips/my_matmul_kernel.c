// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Naive triple-loop matmul kernel exposed as an IREE executable plugin.
//
// The IREE-compiled dispatch calls @my_matmul_kernel through the IREE import
// mechanism (not direct dynamic linking). At runtime the plugin is registered
// via:
//   iree-run-module --executable_plugin=libmy_matmul_kernel.dylib ...
//
// IREE import calling convention (for all imports):
//   int fn(void* params_ptr, void* context, void* reserved)
//
// The params_ptr points to a packed struct whose fields mirror the
// func.call arguments emitted by LowerMIPSToFuncCall, in order:
//   float* A, int64_t A_off, A_s0, A_s1,
//   float* B, int64_t B_off, B_s0, B_s1,
//   float* C, int64_t C_off, C_s0, C_s1,
//   int64_t M, int64_t N, int64_t K

#include <stdint.h>
#include <stddef.h>

// IREE standalone plugin header — only requires C99 standard headers.
#include "iree/hal/local/executable_plugin.h"

//===----------------------------------------------------------------------===//
// Kernel implementation
//===----------------------------------------------------------------------===//

// Packed argument struct mirroring the func.call arguments from
// LowerMIPSToFuncCall.cpp.
typedef struct {
  float *A;
  int64_t A_off, A_s0, A_s1;
  float *B;
  int64_t B_off, B_s0, B_s1;
  float *C;
  int64_t C_off, C_s0, C_s1;
  int64_t M, N, K;
} my_matmul_kernel_args_t;

// Import thunk wrapper — called by the IREE runtime with the packed args.
static int my_matmul_kernel_import(void *params_ptr, void *context,
                                   void *reserved) {
  (void)context;
  (void)reserved;
  const my_matmul_kernel_args_t *a = (const my_matmul_kernel_args_t *)params_ptr;

  float *A = a->A + a->A_off;
  float *B = a->B + a->B_off;
  float *C = a->C + a->C_off;
  int64_t M = a->M, N = a->N, K = a->K;
  int64_t A_s0 = a->A_s0, A_s1 = a->A_s1;
  int64_t B_s0 = a->B_s0, B_s1 = a->B_s1;
  int64_t C_s0 = a->C_s0, C_s1 = a->C_s1;

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += A[m * A_s0 + k * A_s1] * B[k * B_s0 + n * B_s1];
      C[m * C_s0 + n * C_s1] = acc;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// IREE Executable Plugin interface
//===----------------------------------------------------------------------===//

static iree_hal_executable_plugin_status_t plugin_load(
    const iree_hal_executable_plugin_environment_v0_t *environment,
    size_t param_count,
    const iree_hal_executable_plugin_string_pair_t *params, void **out_self) {
  (void)environment;
  (void)param_count;
  (void)params;
  *out_self = NULL;  // stateless plugin
  return iree_hal_executable_plugin_ok_status();
}

static void plugin_unload(void *self) { (void)self; }

static iree_hal_executable_plugin_status_t plugin_resolve(
    void *self, const iree_hal_executable_plugin_resolve_params_v0_t *params,
    iree_hal_executable_plugin_resolution_t *out_resolution) {
  (void)self;
  *out_resolution = 0;
  bool any_required_not_found = false;

  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;  // already resolved
    const char *name = params->symbol_names[i];
    bool optional = iree_hal_executable_plugin_import_is_optional(name);
    if (optional) ++name;  // skip the leading '?'

    if (iree_hal_executable_plugin_strcmp(name, "my_matmul_kernel") == 0) {
      params->out_fn_ptrs[i] = my_matmul_kernel_import;
      params->out_fn_contexts[i] = NULL;
    } else {
      if (optional) {
        *out_resolution |=
            IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
      } else {
        any_required_not_found = true;
      }
    }
  }

  return any_required_not_found
             ? iree_hal_executable_plugin_status_from_code(
                   IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
             : iree_hal_executable_plugin_ok_status();
}

// Exported entry point queried by the IREE runtime (via dlsym).
IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t **
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void *reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      .name = "mips_matmul",
      .description = "MIPS custom matmul kernel plugin",
      .features = IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_STANDALONE,
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = plugin_load,
      .unload = plugin_unload,
      .resolve = plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t **)&plugin
             : NULL;
}
