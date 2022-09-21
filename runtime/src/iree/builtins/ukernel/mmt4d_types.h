// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_TYPES_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_TYPES_H_

#include "iree/builtins/ukernel/common.h"

// Supported combinations of data types (order: LHS, RHS, OUT).
enum iree_ukernel_mmt4d_type_t {
  iree_ukernel_mmt4d_type_none = 0,
  iree_ukernel_mmt4d_type_f32f32f32,
  iree_ukernel_mmt4d_type_i8i8i32,
};

typedef enum iree_ukernel_mmt4d_type_t iree_ukernel_mmt4d_type_t;

// Parameters for a mmt4d operation.
struct iree_ukernel_mmt4d_params_t {
  iree_ukernel_mmt4d_type_t type;
  uint32_t flags;
  const void* lhs_buffer;
  const void* rhs_buffer;
  void* out_buffer;
  iree_ukernel_ssize_t lhs_stride;
  iree_ukernel_ssize_t rhs_stride;
  iree_ukernel_ssize_t out_stride;
  iree_ukernel_ssize_t M;
  iree_ukernel_ssize_t N;
  iree_ukernel_ssize_t K;
  int32_t M0;
  int32_t N0;
  int32_t K0;
  const uint64_t* cpu_data;
};

typedef struct iree_ukernel_mmt4d_params_t iree_ukernel_mmt4d_params_t;

// Status codes returned by a mmt4d operation.
enum iree_ukernel_mmt4d_status_t {
  iree_ukernel_mmt4d_status_ok = 0,
  iree_ukernel_mmt4d_status_bad_type,
  iree_ukernel_mmt4d_status_bad_flags,
  iree_ukernel_mmt4d_status_unsupported_huge_or_negative_dimension,
  iree_ukernel_mmt4d_status_unsupported_generic_tile_size,
};

typedef enum iree_ukernel_mmt4d_status_t iree_ukernel_mmt4d_status_t;

// TODO: move these flags to a header file shared with compiler/.
#define IREE_VMVX_MATMUL_FLAG_ACCUMULATE 1

#define IREE_UKERNEL_MMT4D_RETURN_IF_ERROR(X)     \
  do {                                            \
    iree_ukernel_mmt4d_status_t status = (X);     \
    if (status != iree_ukernel_mmt4d_status_ok) { \
      return status;                              \
    }                                             \
  } while (0)

// Function pointer type for tile functions, i.e. typically architecture
// specific functions computing one M0xN0 tile of the output matrix, i.e.
// the inner-most loop of the matmul, i.e. the thing that we should actually
// be calling "micro kernel" except that the name is already taken by the
// higher-level builtin name.
//
// The 'params' argument is only used by generic kernels. Actual optimized
// kernels are already specialized for a given tile shape (M0xN0xK0), so the
// five first arguments here are the only information that they need. Not having
// to address 'params' struct fields in the middle of assembly kernels is
// good, because it's hard to get the struct field offsets right in assembly
// and keep that in sync with future struct changes.
typedef void (*iree_ukernel_mmt4d_tile_func_t)(
    void* /*out_tile*/, const void* /*lhs_panel*/, const void* /*rhs_panel*/,
    int32_t /*K*/, uint32_t /*flags*/,
    const iree_ukernel_mmt4d_params_t* /*params*/);

// Tile kernel declarations. Prototype matches iree_ukernel_mmt4d_tile_func_t.
#define IREE_UKERNEL_MMT4D_TILE_FUNC_DECL(NAME)                           \
  void NAME(void* out_tile, const void* lhs_panel, const void* rhs_panel, \
            int32_t K, uint32_t flags,                                    \
            const iree_ukernel_mmt4d_params_t* params);

// Log2 of size of LHS matrix element type, e.g. f32 --> size=4 --> log2=2
static inline int iree_ukernel_mmt4d_lhs_elem_size_log2(
    iree_ukernel_mmt4d_type_t type) {
  switch (type) {
    case iree_ukernel_mmt4d_type_f32f32f32:
      return 2;
    default:
      return 0;
  }
}

static inline int iree_ukernel_mmt4d_lhs_elem_size(
    iree_ukernel_mmt4d_type_t type) {
  return 1 << iree_ukernel_mmt4d_lhs_elem_size_log2(type);
}

// Log2 of size of RHS matrix element type, e.g. f32 --> size=4 --> log2=2
static inline int iree_ukernel_mmt4d_rhs_elem_size_log2(
    iree_ukernel_mmt4d_type_t type) {
  return iree_ukernel_mmt4d_lhs_elem_size_log2(type);  // for now it's the same
}

static inline int iree_ukernel_mmt4d_rhs_elem_size(
    iree_ukernel_mmt4d_type_t type) {
  return 1 << iree_ukernel_mmt4d_rhs_elem_size_log2(type);
}

// Log2 of size of OUT matrix element type, e.g. f32 --> size=4 --> log2=2
static inline int iree_ukernel_mmt4d_out_elem_size_log2(
    iree_ukernel_mmt4d_type_t type) {
  switch (type) {
    case iree_ukernel_mmt4d_type_f32f32f32:
    case iree_ukernel_mmt4d_type_i8i8i32:
      return 2;
    default:
      return 0;
  }
}

static inline int iree_ukernel_mmt4d_out_elem_size(
    iree_ukernel_mmt4d_type_t type) {
  return 1 << iree_ukernel_mmt4d_out_elem_size_log2(type);
}

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_TYPES_H_
