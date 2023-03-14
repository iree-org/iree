// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/query_tile_sizes.h"

#if defined(IREE_UK_ARCH_ARM_64)
#include "iree/builtins/ukernel/arch/arm_64/query_tile_sizes_arm_64.h"
#endif

static bool iree_uk_query_tile_sizes_operation_is_matmul(
    iree_uk_uint32_t flags) {
  iree_uk_uint32_t op = iree_uk_query_tile_sizes_operation(flags);
  return op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32 ||
         op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32;
}

static void iree_uk_query_tile_sizes_2d_validate(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
#ifdef IREE_UK_ENABLE_ASSERTS
  IREE_UK_ASSERT(iree_uk_query_tile_sizes_operation_is_matmul(params->flags));
  iree_uk_uint32_t role = iree_uk_query_tile_sizes_operand_role(params->flags);
  IREE_UK_ASSERT(role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS ||
                 role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS ||
                 role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT);
  const iree_uk_int64_t kDynamic = IREE_UK_INT64_MIN;
  IREE_UK_ASSERT((params->size0 >= 0 || params->size0 == kDynamic) ||
                 (params->size1 >= 0 || params->size1 == kDynamic));
#endif  // IREE_UK_ENABLE_ASSERTS
}

static iree_uk_matmul_tile_sizes_t iree_uk_query_matmul_tile_sizes_generic(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
  // Dummy values, originally taken from what was used on ARM_64 +dotprod for
  // i8i8i32. Not particularly meaningful outside of that case, just is what
  // some tests have been written against.
  (void)params;
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 4, .N = 8};
}

static bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes) {
#if defined(IREE_UK_ARCH_ARM_64)
  return iree_uk_query_matmul_tile_sizes_arm_64(params, out_matmul_tile_sizes);
#endif
  return false;
}

static void iree_uk_query_tile_sizes_2d_matmul(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_query_tile_sizes_2d_out_params_t* out_params) {
  iree_uk_matmul_tile_sizes_t matmul_tile_sizes;
  if (!iree_uk_query_matmul_tile_sizes_arch(params, &matmul_tile_sizes)) {
    matmul_tile_sizes = iree_uk_query_matmul_tile_sizes_generic(params);
  }
  iree_uk_uint32_t role = iree_uk_query_tile_sizes_operand_role(params->flags);
  if (role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS) {
    out_params->tile_size0 = matmul_tile_sizes.M;
    out_params->tile_size1 = matmul_tile_sizes.K;
  } else if (role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS) {
    out_params->tile_size0 = matmul_tile_sizes.N;
    out_params->tile_size1 = matmul_tile_sizes.K;
  } else if (role == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT) {
    out_params->tile_size0 = matmul_tile_sizes.M;
    out_params->tile_size1 = matmul_tile_sizes.N;
  } else {
    // Can't happen, validated earlier.
    IREE_UK_ASSUME_UNREACHABLE;
  }
}

IREE_UK_EXPORT void iree_uk_query_tile_sizes_2d(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_query_tile_sizes_2d_out_params_t* out_params) {
  iree_uk_query_tile_sizes_2d_validate(params);

  if (iree_uk_query_tile_sizes_operation_is_matmul(params->flags)) {
    iree_uk_query_tile_sizes_2d_matmul(params, out_params);
  } else {
    // Can't happen, validated earlier.
    IREE_UK_ASSUME_UNREACHABLE;
  }
}
