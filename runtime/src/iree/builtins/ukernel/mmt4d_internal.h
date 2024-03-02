// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_INTERNAL_H_

#include "iree/builtins/ukernel/mmt4d.h"

// While the iree_uk_mmt4d public entry point takes separate parameters,
// internally the implementation functions pass parameters as this struct.
typedef struct iree_uk_mmt4d_params_t {
  const void* lhs_buffer;
  iree_uk_index_t lhs_offset;
  iree_uk_index_t lhs_stride0;
  const void* rhs_buffer;
  iree_uk_index_t rhs_offset;
  iree_uk_index_t rhs_stride0;
  void* out_buffer;
  iree_uk_index_t out_offset;
  iree_uk_index_t out_stride0;
  iree_uk_index_t M;
  iree_uk_index_t N;
  iree_uk_index_t K;
  iree_uk_int32_t M0;
  iree_uk_int32_t N0;
  iree_uk_int32_t K0;
  iree_uk_uint32_t flags;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_mmt4d_params_t;

// Same as the iree_uk_mmt4d public entry point, but taking the struct.
void iree_uk_mmt4d_p(const iree_uk_mmt4d_params_t* params);

// Same as the iree_uk_mmt4d_info public entry point, but taking the struct.
// Only the struct fields corresponding to iree_uk_mmt4d_info parameters are
// used.
iree_uk_uint32_t iree_uk_mmt4d_info_p(const iree_uk_mmt4d_params_t* params);

typedef enum iree_uk_mmt4d_type_t {
  iree_uk_mmt4d_type_f32f32f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_32, FLOAT_32, FLOAT_32),
  iree_uk_mmt4d_type_s8s8s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_8, SINT_8, SINT_32),
  iree_uk_mmt4d_type_s8s4s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_8, SINT_4, SINT_32),
  iree_uk_mmt4d_type_s16s16s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_16, SINT_16, SINT_32),
  iree_uk_mmt4d_type_s16u4s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_16, UINT_4, SINT_32),
  iree_uk_mmt4d_type_s16s8s32 =
      IREE_UK_TIE_3_TYPES_LITERAL(SINT_16, SINT_8, SINT_32),
  iree_uk_mmt4d_type_f16f16f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_16, FLOAT_16, FLOAT_32),
  iree_uk_mmt4d_type_f16f16f16 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_16, FLOAT_16, FLOAT_16),
  iree_uk_mmt4d_type_bf16bf16f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(BFLOAT_16, BFLOAT_16, FLOAT_32),
  iree_uk_mmt4d_type_bf16bf16bf16 =
      IREE_UK_TIE_3_TYPES_LITERAL(BFLOAT_16, BFLOAT_16, BFLOAT_16),
} iree_uk_mmt4d_type_t;

static inline iree_uk_mmt4d_type_t iree_uk_mmt4d_type(iree_uk_uint32_t flags) {
  switch (flags & IREE_UK_FLAG_MMT4D_TYPE_MASK) {
    case IREE_UK_FLAG_MMT4D_TYPE_F32F32F32:
      return iree_uk_mmt4d_type_f32f32f32;
    case IREE_UK_FLAG_MMT4D_TYPE_S8S8S32:
      return iree_uk_mmt4d_type_s8s8s32;
    case IREE_UK_FLAG_MMT4D_TYPE_S8S4S32:
      return iree_uk_mmt4d_type_s8s4s32;
    case IREE_UK_FLAG_MMT4D_TYPE_S16S16S32:
      return iree_uk_mmt4d_type_s16s16s32;
    case IREE_UK_FLAG_MMT4D_TYPE_S16U4S32:
      return iree_uk_mmt4d_type_s16u4s32;
    case IREE_UK_FLAG_MMT4D_TYPE_S16S8S32:
      return iree_uk_mmt4d_type_s16s8s32;
    case IREE_UK_FLAG_MMT4D_TYPE_F16F16F32:
      return iree_uk_mmt4d_type_f16f16f32;
    case IREE_UK_FLAG_MMT4D_TYPE_F16F16F16:
      return iree_uk_mmt4d_type_f16f16f16;
    case IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32:
      return iree_uk_mmt4d_type_bf16bf16f32;
    case IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16:
      return iree_uk_mmt4d_type_bf16bf16bf16;
    default:
      // Work around a LLVM/riscv32 miscompile. Without the unreachable here,
      // returning (iree_uk_mmt4d_type_t)0 causes this whole switch statement to
      // be miscompiled by LLVM/riscv32 as if it were UB, as the 0 was passed to
      // `iree_uk_type_bit_count(x)`, which evaluates to `1<<(x - 3)`, which is
      // UB if x<3.
#if defined(IREE_UK_COMPILER_CLANG) && defined(IREE_UK_ARCH_RISCV_32)
      __builtin_unreachable();
#endif
      // Shouldn't happen, validated earlier.
      return (iree_uk_mmt4d_type_t)0;
  }
}

static inline iree_uk_type_t iree_uk_mmt4d_lhs_type(iree_uk_mmt4d_type_t type) {
  return iree_uk_untie_type(0, type);
}

static inline iree_uk_type_t iree_uk_mmt4d_rhs_type(iree_uk_mmt4d_type_t type) {
  return iree_uk_untie_type(1, type);
}

static inline iree_uk_type_t iree_uk_mmt4d_out_type(iree_uk_mmt4d_type_t type) {
  return iree_uk_untie_type(2, type);
}

// Function pointer type for tile functions, i.e. typically architecture
// specific functions computing one M0xN0 tile of the output matrix, i.e.
// the inner-most loop of the matmul, i.e. the thing that we should actually
// be calling "micro kernel" except that the name is already taken by the
// higher-level builtin name.
typedef void (*iree_uk_mmt4d_tile_func_t)(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params);

// Tile kernel declarations. Prototype matches iree_uk_mmt4d_tile_func_t.
#define IREE_UK_MMT4D_TILE_FUNC_DECL(NAME)          \
  void NAME(void* IREE_UK_RESTRICT out_tile,        \
            const void* IREE_UK_RESTRICT lhs_panel, \
            const void* IREE_UK_RESTRICT rhs_panel, \
            const iree_uk_mmt4d_params_t* params);

#define IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(GENERIC_FUNC, FUNC, M0) \
  void FUNC(void* IREE_UK_RESTRICT out_tile,                        \
            const void* IREE_UK_RESTRICT lhs_panel,                 \
            const void* IREE_UK_RESTRICT rhs_panel,                 \
            const iree_uk_mmt4d_params_t* params) {                 \
    GENERIC_FUNC(out_tile, lhs_panel, rhs_panel, params, M0);       \
  }

#define IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4(G, F1, F2, F4) \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F1, 1)                  \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F2, 2)                  \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F4, 4)

#define IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(G, F1, F2, F4, F8) \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F1, 1)                        \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F2, 2)                        \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F4, 4)                        \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F8, 8)

#define IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8_16(G, F1, F2, F4, F8, F16) \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(G, F1, F2, F4, F8)               \
  IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(G, F16, 16)

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

// Architecture-specific implementation, or generic fallback returning null.
iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params);

// Generic fallback.
iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_generic(
    const iree_uk_mmt4d_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_INTERNAL_H_
