// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_uk_mmt4d_type_t {
  iree_uk_mmt4d_type_f32f32f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_32, FLOAT_32, FLOAT_32),
  iree_uk_mmt4d_type_i8i8i32 =
      IREE_UK_TIE_3_TYPES_LITERAL(INT_8, INT_8, INT_32),
} iree_uk_mmt4d_type_t;

static inline iree_uk_type_t iree_uk_mmt4d_lhs_type(iree_uk_mmt4d_type_t type) {
  return iree_uk_untie_type(0, type);
}

static inline iree_uk_type_t iree_uk_mmt4d_rhs_type(iree_uk_mmt4d_type_t type) {
  return iree_uk_untie_type(1, type);
}

static inline iree_uk_type_t iree_uk_mmt4d_out_type(iree_uk_mmt4d_type_t type) {
  return iree_uk_untie_type(2, type);
}

// Parameters for a mmt4d operation.
typedef struct iree_uk_mmt4d_params_t {
  iree_uk_mmt4d_type_t type;
  iree_uk_uint32_t flags;
  iree_uk_ssize_t lhs_stride;
  iree_uk_ssize_t rhs_stride;
  iree_uk_ssize_t out_stride;
  iree_uk_ssize_t M;
  iree_uk_ssize_t N;
  iree_uk_ssize_t K;
  iree_uk_int32_t M0;
  iree_uk_int32_t N0;
  iree_uk_int32_t K0;
  const void* lhs_buffer;
  const void* rhs_buffer;
  void* out_buffer;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_mmt4d_params_t;

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
typedef void (*iree_uk_mmt4d_tile_func_t)(
    void* /*out_tile*/, const void* /*lhs_panel*/, const void* /*rhs_panel*/,
    iree_uk_int32_t /*K*/, iree_uk_uint32_t /*flags*/,
    const iree_uk_mmt4d_params_t* /*params*/);

// Tile kernel declarations. Prototype matches iree_uk_mmt4d_tile_func_t.
#define IREE_UK_MMT4D_TILE_FUNC_DECL(NAME)                                \
  void NAME(void* out_tile, const void* lhs_panel, const void* rhs_panel, \
            iree_uk_int32_t K, iree_uk_uint32_t flags,                    \
            const iree_uk_mmt4d_params_t* params);

// In order to be helpful as a reference for future architecture-specific
// kernels, the generic kernels are structured like an actual optimized kernel,
// using an "accumulator tile" that in this case is a stack array (which would
// become a group of SIMD registers in an actual optimized kernel). The downside
// of this approach is that we have to set a fixed max size for the accumulator
// tile, but for now all known cases are comfortably far below where trouble
// would happen. For reference:
// - On ARM NEON, the entire register space is 512 bytes, so the accumulator
//   tile is less than that, typically 256 to 384 bytes.
// - On ARM SME, we will be working with an accumulator tile as large as 4096
//   bytes (IIUC).
// - The smallest stack frame size limit that we know we may have to deal with
//   on certain targets is 16 kilobytes.
// The size or architecture-specific tiles is relevant here because this
// generic code is what will be run as a fallback if the device is found not to
// support the CPU feature that the tile sizes were picked to target.
enum { iree_uk_mmt4d_tile_generic_max_bytes = 4096 };

// Main entry point.
IREE_UK_EXPORT void iree_uk_mmt4d(const iree_uk_mmt4d_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_H_
