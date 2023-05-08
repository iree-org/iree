// Copyright 2023 The IREE Authors
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

IREE_FLAG(string, type, "f32f32f32",
          "Element types triple (LHS, RHS, OUT). Valid values include: "
          "f32f32f32, i8i8i32.");
IREE_FLAG(int32_t, M, 256, "M dimension size (number of rows of LHS and OUT)");
IREE_FLAG(
    int32_t, K, 256,
    "K dimension size (number of columns of LHS and number of rows of RHS)");
IREE_FLAG(int32_t, N, 256,
          "N dimension size (number of columns of RHS and OUT)");
IREE_FLAG(bool, accumulate, false,
          "If true, benchmark a matmul accumulating into existing accumulator "
          "(OUT += LHS * RHS). If false, benchmark just a matmul overwriting "
          "the accumulator (OUT = LHS * RHS)");
IREE_FLAG(
    string, cpu_features, "host",
    "Name of standard CPU features set to enable, or \"host\" to detect the "
    "host CPU capabilities. Other values are like in other benchmarks, e.g. "
    "\"avx2_fma\", \"avx512_base\". The empty string \"\" means the "
    "architecture baseline (e.g. on x86-64 that would be SSE2).");

typedef struct iree_uk_benchmark_e2e_matmul_params_t {
  iree_uk_uint32_t mmt4d_flags;
  int M;
  int K;
  int N;
} iree_uk_benchmark_e2e_matmul_params_t;

static iree_uk_uint32_t iree_uk_qts_op_flag(iree_uk_mmt4d_type_t type) {
  if (type == iree_uk_mmt4d_type_f32f32f32)
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32;
  if (type == iree_uk_mmt4d_type_i8i8i32)
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32;
  iree_abort();
  return 0;
}

static void iree_uk_query_tile_sizes_for_one_operand(
    iree_uk_uint32_t flags, int size0, int size1,
    const iree_uk_uint64_t* cpu_data, int* tile_size0, int* tile_size1) {
  iree_uk_query_tile_sizes_2d_params_t qts_params = {
      .flags = flags, .size0 = size0, .size1 = size1, .cpu_data = cpu_data};
  iree_uk_query_tile_sizes_2d_out_params_t qts_out_params = {0};
  iree_uk_query_tile_sizes_2d(&qts_params, &qts_out_params);
  *tile_size0 = qts_out_params.tile_size0;
  *tile_size1 = qts_out_params.tile_size1;
}

static void iree_uk_query_tile_sizes_for_all_operands(
    const iree_uk_benchmark_e2e_matmul_params_t* params,
    const iree_uk_uint64_t* cpu_data, int* M0, int* K0, int* N0) {
  int M0_lhs = 0, M0_out = 0, K0_lhs = 0, K0_rhs = 0, N0_rhs = 0, N0_out = 0;
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params->mmt4d_flags);
  iree_uk_uint32_t qts_op_flag = iree_uk_qts_op_flag(mmt4d_type);
  iree_uk_query_tile_sizes_for_one_operand(
      qts_op_flag | IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS, params->M,
      params->K, cpu_data, &M0_lhs, &K0_lhs);
  iree_uk_query_tile_sizes_for_one_operand(
      qts_op_flag | IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS, params->K,
      params->N, cpu_data, &N0_rhs, &K0_rhs);
  iree_uk_query_tile_sizes_for_one_operand(
      qts_op_flag | IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT,
      params->M, params->N, cpu_data, &M0_out, &N0_out);
  if (M0_lhs != M0_out || K0_lhs != K0_rhs || N0_rhs != N0_out) {
    fprintf(stderr, "query_tile_sizes mismatch\n");
    iree_abort();
  }
  *M0 = M0_lhs;
  *K0 = K0_lhs;
  *N0 = N0_rhs;
}

static int iree_uk_ceildiv(int a, int b) {
  IREE_UK_ASSERT(a > 0 && b > 0);
  return (a + b - 1) / b;
}

static void iree_uk_reference_rowmajor_matmul_f32f32f32(
    const iree_uk_benchmark_e2e_matmul_params_t* params, const float* lhs,
    const float* rhs, float* out) {
  bool accumulate = params->mmt4d_flags & IREE_UK_FLAG_MMT4D_ACCUMULATE;
  for (int i = 0; i < params->M; ++i) {
    for (int j = 0; j < params->N; ++j) {
      float* out_ptr = out + i * params->N + j;
      float acc = accumulate ? *out_ptr : 0.f;
      for (int k = 0; k < params->K; ++k) {
        acc += lhs[i * params->K + k] * rhs[k * params->N + j];
      }
      *out_ptr = acc;
    }
  }
}

static void iree_uk_reference_rowmajor_matmul_i8i8i32(
    const iree_uk_benchmark_e2e_matmul_params_t* params,
    const iree_uk_int8_t* lhs, const iree_uk_int8_t* rhs,
    iree_uk_int32_t* out) {
  bool accumulate = params->mmt4d_flags & IREE_UK_FLAG_MMT4D_ACCUMULATE;
  for (int i = 0; i < params->M; ++i) {
    for (int j = 0; j < params->N; ++j) {
      iree_uk_int32_t* out_ptr = out + i * params->N + j;
      iree_uk_int32_t acc = accumulate ? *out_ptr : 0;
      for (int k = 0; k < params->K; ++k) {
        iree_uk_int32_t lhs_val = lhs[i * params->K + k];
        iree_uk_int32_t rhs_val = rhs[k * params->N + j];
        acc += lhs_val * rhs_val;
      }
      *out_ptr = acc;
    }
  }
}

static void iree_uk_reference_rowmajor_matmul(
    const iree_uk_benchmark_e2e_matmul_params_t* params, const void* lhs,
    const void* rhs, void* out) {
  switch (params->mmt4d_flags & IREE_UK_FLAG_MMT4D_TYPE_MASK) {
    case IREE_UK_FLAG_MMT4D_TYPE_F32F32F32:
      iree_uk_reference_rowmajor_matmul_f32f32f32(
          params, (const float*)lhs, (const float*)rhs, (float*)out);
      break;
    case IREE_UK_FLAG_MMT4D_TYPE_I8I8I32:
      iree_uk_reference_rowmajor_matmul_i8i8i32(
          params, (const iree_uk_int8_t*)lhs, (const iree_uk_int8_t*)rhs,
          (iree_uk_int32_t*)out);
      break;
    default:
      IREE_UK_ASSERT(false);
  }
}

static uint32_t iree_uk_pack_flags(iree_uk_type_t type) {
  switch (type) {
    case IREE_UK_TYPE_FLOAT_32:
      return IREE_UK_FLAG_PACK_TYPE_F32F32;
    case IREE_UK_TYPE_INT_32:
      return IREE_UK_FLAG_PACK_TYPE_I32I32;
    case IREE_UK_TYPE_INT_8:
      return IREE_UK_FLAG_PACK_TYPE_I8I8;
    default:
      IREE_UK_ASSERT(false);
      return IREE_UK_FLAG_PACK_TYPE_NONE;
  }
}

static uint32_t iree_uk_unpack_flags(iree_uk_type_t type) {
  switch (type) {
    case IREE_UK_TYPE_FLOAT_32:
      return IREE_UK_FLAG_UNPACK_TYPE_F32F32;
    case IREE_UK_TYPE_INT_32:
      return IREE_UK_FLAG_UNPACK_TYPE_I32I32;
    default:
      IREE_UK_ASSERT(false);
      return IREE_UK_FLAG_UNPACK_TYPE_NONE;
  }
}

static void iree_uk_e2e_matmul(
    const iree_uk_pack_params_t* pack_lhs_params,
    const iree_uk_pack_params_t* pack_rhs_params,
    const iree_uk_pack_params_t* pack_out_params,
    const iree_uk_mmt4d_params_t* mmt4d_params,
    const iree_uk_unpack_params_t* unpack_out_params) {
  iree_uk_pack(pack_lhs_params);
  iree_uk_pack(pack_rhs_params);
  if (mmt4d_params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    iree_uk_pack(pack_out_params);
  }
  iree_uk_mmt4d(mmt4d_params);
  iree_uk_unpack(unpack_out_params);
}

static iree_status_t iree_uk_benchmark_e2e_matmul(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_uk_benchmark_user_data_t* user_data = benchmark_def->user_data;
  const iree_uk_uint64_t* cpu_data = iree_uk_benchmark_cpu_data(user_data);
  const iree_uk_benchmark_e2e_matmul_params_t* params =
      iree_uk_benchmark_params(user_data);

  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params->mmt4d_flags);
  iree_uk_type_t lhs_type = iree_uk_mmt4d_lhs_type(mmt4d_type);
  iree_uk_type_t rhs_type = iree_uk_mmt4d_rhs_type(mmt4d_type);
  iree_uk_type_t out_type = iree_uk_mmt4d_out_type(mmt4d_type);

  int M0 = 0, K0 = 0, N0 = 0;
  iree_uk_query_tile_sizes_for_all_operands(params, cpu_data, &M0, &K0, &N0);

  int M1 = iree_uk_ceildiv(params->M, M0);
  int K1 = iree_uk_ceildiv(params->K, K0);
  int N1 = iree_uk_ceildiv(params->N, N0);

  iree_uk_mmt4d_params_t mmt4d_params = {
      .flags = params->mmt4d_flags,
      .cpu_data = cpu_data,
      .M = M1,
      .K = K1,
      .N = N1,
      .M0 = M0,
      .K0 = K0,
      .N0 = N0,
      .lhs_stride0 = K1 * M0 * K0,
      .rhs_stride0 = K1 * N0 * K0,
      .out_stride0 = N1 * M0 * N0,
  };

  iree_uk_pack_params_t pack_lhs_params = {
      .cpu_data = cpu_data,
      .flags = iree_uk_pack_flags(lhs_type),
      .in_size0 = params->M,
      .in_size1 = params->K,
      .out_size0 = M1,
      .out_size1 = K1,
      .out_size2 = M0,
      .out_size3 = K0,
      .in_stride0 = params->K,
      .out_stride0 = mmt4d_params.lhs_stride0,
      .padding_value = 0,
  };

  iree_uk_pack_params_t pack_rhs_params = {
      .cpu_data = cpu_data,
      .flags = iree_uk_pack_flags(rhs_type) |
               IREE_UK_FLAG_PACK_TRANSPOSE_INNER |
               IREE_UK_FLAG_PACK_TRANSPOSE_OUTER,
      .in_size0 = params->K,
      .in_size1 = params->N,
      .out_size0 = N1,
      .out_size1 = K1,
      .out_size2 = N0,
      .out_size3 = K0,
      .in_stride0 = params->N,
      .out_stride0 = mmt4d_params.rhs_stride0,
      .padding_value = 0,
  };

  iree_uk_pack_params_t pack_out_params = {
      .cpu_data = cpu_data,
      .flags = iree_uk_pack_flags(out_type),
      .in_size0 = params->M,
      .in_size1 = params->N,
      .out_size0 = M1,
      .out_size1 = N1,
      .out_size2 = M0,
      .out_size3 = N0,
      .in_stride0 = params->N,
      .out_stride0 = mmt4d_params.out_stride0,
      .padding_value = 0,
  };

  iree_uk_unpack_params_t unpack_out_params = {
      .cpu_data = cpu_data,
      .flags = iree_uk_unpack_flags(out_type),
      .out_size0 = params->M,
      .out_size1 = params->N,
      .in_size0 = M1,
      .in_size1 = N1,
      .in_size2 = M0,
      .in_size3 = N0,
      .out_stride0 = params->N,
      .in_stride0 = mmt4d_params.out_stride0,
  };

  iree_uk_ssize_t rowmajor_lhs_buffer_size =
      iree_uk_2d_buffer_length(lhs_type, params->M, params->K);
  iree_uk_ssize_t rowmajor_rhs_buffer_size =
      iree_uk_2d_buffer_length(rhs_type, params->K, params->N);
  iree_uk_ssize_t rowmajor_out_buffer_size =
      iree_uk_2d_buffer_length(out_type, params->M, params->N);
  iree_uk_ssize_t packed_lhs_buffer_size =
      iree_uk_2d_buffer_length(lhs_type, M1, mmt4d_params.lhs_stride0);
  iree_uk_ssize_t packed_rhs_buffer_size =
      iree_uk_2d_buffer_length(rhs_type, N1, mmt4d_params.rhs_stride0);
  iree_uk_ssize_t packed_out_buffer_size =
      iree_uk_2d_buffer_length(out_type, M1, mmt4d_params.out_stride0);
  void* rowmajor_lhs_buffer = malloc(rowmajor_lhs_buffer_size);
  void* rowmajor_rhs_buffer = malloc(rowmajor_rhs_buffer_size);
  void* rowmajor_init_out_buffer = malloc(rowmajor_out_buffer_size);
  void* rowmajor_out_buffer = malloc(rowmajor_out_buffer_size);
  void* packed_lhs_buffer = malloc(packed_lhs_buffer_size);
  void* packed_rhs_buffer = malloc(packed_rhs_buffer_size);
  void* packed_out_buffer = malloc(packed_out_buffer_size);
  iree_uk_random_engine_t* engine = iree_uk_benchmark_random_engine(user_data);
  // It's just about plausible that on some platform, for some number type,
  // performance might be different on zero buffers vs random buffers. But it
  // shouldn't matter that we recreate the random engine every time, getting
  // the same random values again.
  iree_uk_write_random_buffer(rowmajor_lhs_buffer, rowmajor_lhs_buffer_size,
                              lhs_type, engine);
  iree_uk_write_random_buffer(rowmajor_rhs_buffer, rowmajor_rhs_buffer_size,
                              rhs_type, engine);
  iree_uk_write_random_buffer(rowmajor_init_out_buffer,
                              rowmajor_out_buffer_size, out_type, engine);
  iree_uk_write_random_buffer(rowmajor_out_buffer, rowmajor_out_buffer_size,
                              out_type, engine);
  iree_uk_write_random_buffer(packed_lhs_buffer, packed_lhs_buffer_size,
                              lhs_type, engine);
  iree_uk_write_random_buffer(packed_rhs_buffer, packed_rhs_buffer_size,
                              rhs_type, engine);
  iree_uk_write_random_buffer(packed_out_buffer, packed_out_buffer_size,
                              out_type, engine);
  mmt4d_params.lhs_buffer = packed_lhs_buffer;
  mmt4d_params.rhs_buffer = packed_rhs_buffer;
  mmt4d_params.out_buffer = packed_out_buffer;
  pack_lhs_params.in_buffer = rowmajor_lhs_buffer;
  pack_lhs_params.out_buffer = packed_lhs_buffer;
  pack_rhs_params.in_buffer = rowmajor_rhs_buffer;
  pack_rhs_params.out_buffer = packed_rhs_buffer;
  pack_out_params.in_buffer = rowmajor_init_out_buffer;
  pack_out_params.out_buffer = packed_out_buffer;
  unpack_out_params.in_buffer = packed_out_buffer;
  unpack_out_params.out_buffer = rowmajor_out_buffer;

  int64_t num_mul_adds =
      (int64_t)params->M * (int64_t)params->N * (int64_t)params->K;
  // For small problem sizes we check results against reference code.
  if (num_mul_adds <= 512 * 512 * 512) {
    // Run once before the benchmark loop to check numerical correctness.
    iree_uk_e2e_matmul(&pack_lhs_params, &pack_rhs_params, &pack_out_params,
                       &mmt4d_params, &unpack_out_params);
    // Get the reference results to compare against.
    void* rowmajor_reference_out_buffer = malloc(rowmajor_out_buffer_size);
    memcpy(rowmajor_reference_out_buffer, rowmajor_init_out_buffer,
           rowmajor_out_buffer_size);
    iree_uk_reference_rowmajor_matmul(params, rowmajor_lhs_buffer,
                                      rowmajor_rhs_buffer,
                                      rowmajor_reference_out_buffer);
    // Rationale for bit-exact compare: same as in mmt4d_test.
    if (memcmp(rowmajor_out_buffer, rowmajor_reference_out_buffer,
               rowmajor_out_buffer_size)) {
      fprintf(stderr, "❌❌❌ Numerical error! ❌❌❌\n");
      iree_abort();
    }
    free(rowmajor_reference_out_buffer);
  }

  // The benchmark loop.
  int64_t batch_count = 1;
  int64_t total_iterations = 0;
  while (iree_benchmark_keep_running(benchmark_state, batch_count)) {
    for (int i = 0; i < batch_count; ++i) {
      iree_uk_e2e_matmul(&pack_lhs_params, &pack_rhs_params, &pack_out_params,
                         &mmt4d_params, &unpack_out_params);
    }
    total_iterations += batch_count;
    batch_count *= 2;
  }
  iree_benchmark_set_items_processed(benchmark_state,
                                     total_iterations * 2 * num_mul_adds);

  free(rowmajor_lhs_buffer);
  free(rowmajor_rhs_buffer);
  free(rowmajor_out_buffer);
  free(rowmajor_init_out_buffer);
  free(packed_lhs_buffer);
  free(packed_rhs_buffer);
  free(packed_out_buffer);
  return iree_ok_status();
}

iree_uk_uint32_t iree_uk_mmt4d_parse_type_into_flag(const char* type) {
  if (!strcmp(type, "f32f32f32")) {
    return IREE_UK_FLAG_MMT4D_TYPE_F32F32F32;
  }
  if (!strcmp(type, "i8i8i32")) {
    return IREE_UK_FLAG_MMT4D_TYPE_I8I8I32;
  }
  fprintf(stderr, "Unhandled type: %s\n", type);
  iree_abort();
  return (iree_uk_mmt4d_type_t)0;
}

static void iree_uk_benchmark_register_e2e_matmul(const char* type_str, int M,
                                                  int K, int N, bool accumulate,
                                                  const char* cpu_features) {
  char name[128];
  snprintf(name, sizeof name, "e2e_matmul_%s_%dx%dx%d", type_str, M, K, N);
  iree_uk_uint32_t mmt4d_flags = iree_uk_mmt4d_parse_type_into_flag(type_str);
  if (accumulate) mmt4d_flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE;
  iree_uk_benchmark_e2e_matmul_params_t params = {
      .mmt4d_flags = mmt4d_flags, .M = M, .K = K, .N = N};
  iree_uk_benchmark_register(name, iree_uk_benchmark_e2e_matmul, &params,
                             sizeof params, cpu_features);
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "e2e_matmul_benchmark",
      "Benchmark an end-to-end matmul by chaining together multiple ukernels: "
      "query_tile_sizes, pack, mmt4d, unpack.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_uk_benchmark_initialize(&argc, argv);
  iree_uk_benchmark_register_e2e_matmul(FLAG_type, FLAG_M, FLAG_K, FLAG_N,
                                        FLAG_accumulate, FLAG_cpu_features);
  iree_uk_benchmark_run_and_cleanup();
}
