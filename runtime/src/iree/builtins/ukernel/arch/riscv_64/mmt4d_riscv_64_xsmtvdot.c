// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SpaceMiT IME (XSMTVDot) s8s8s32 mmt4d.
//
// One `smt.vmadot` (X60, VLEN=256, SEW=8) is a 4x4x8 int8->int32 MAC atom.
// Macro-tile is fixed 12x16x8 = 3x4 atoms (12 vmadot/K panel); K1 panels in
// the inner loop. Accumulators stay in {vd,vd+1} across K via one asm block
// (`vsetvli` hoisted per phase).
//
// Panels (vmadot-native):
//   lhs : int8 [K1][M0][K0]   K0-contiguous
//   rhs : int8 [K1][N0][K0]   transposed (N0,K0), K0-contiguous
//   out : int32 [M0][N0]
//
// Requires +xsmtvdot (common_riscv_64.h) and -march=...xsmtvdot.

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

#define IME_M_ATOM 4
#define IME_N_ATOM 4
#define IME_K_ATOM 8
#define IME_K0 8
// int32 elements in one atom's {vd,vd+1} accumulator group.
#define IME_ATOM_ELEMS (IME_M_ATOM * IME_N_ATOM)

// Gather the strided (M0,N0) int32 output tile into a contiguous per-atom
// scratch layout: atom (mt,nt) occupies scratch[(mt*NT+nt)*IME_ATOM_ELEMS] as a
// row-major 4x4 block, which is exactly the layout the {vd,vd+1} accumulator
// group holds.
static inline void ime_gather_acc(const iree_uk_int32_t* out,
                                  iree_uk_int32_t* scratch, int MT, int NT,
                                  int N0) {
  for (int mt = 0; mt < MT; mt++) {
    for (int nt = 0; nt < NT; nt++) {
      iree_uk_int32_t* dst = scratch + (mt * NT + nt) * IME_ATOM_ELEMS;
      const iree_uk_int32_t* src =
          out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
      for (int r = 0; r < IME_M_ATOM; r++)
        for (int c = 0; c < IME_N_ATOM; c++)
          dst[r * IME_N_ATOM + c] = src[r * N0 + c];
    }
  }
}

// Inverse of ime_gather_acc: scatter the contiguous per-atom scratch tiles back
// into the strided (M0,N0) output.
static inline void ime_scatter_acc(iree_uk_int32_t* out,
                                   const iree_uk_int32_t* scratch, int MT,
                                   int NT, int N0) {
  for (int mt = 0; mt < MT; mt++) {
    for (int nt = 0; nt < NT; nt++) {
      const iree_uk_int32_t* src = scratch + (mt * NT + nt) * IME_ATOM_ELEMS;
      iree_uk_int32_t* dst = out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
      for (int r = 0; r < IME_M_ATOM; r++)
        for (int c = 0; c < IME_N_ATOM; c++)
          dst[r * N0 + c] = src[r * IME_N_ATOM + c];
    }
  }
}

void iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  IREE_UK_ASSERT(params->M0 == 12);
  IREE_UK_ASSERT(params->N0 == 16);
  IREE_UK_ASSERT(params->K0 == IME_K0);

  const iree_uk_int8_t* IREE_UK_RESTRICT lhs = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out = out_tile;

  const int K1 = (int)(params->K);

  // Fixed 3x4 atom grid over the asserted 12x16 output.
  enum { MT = 3, NT = 4, N0 = 16 };
  const int accumulate = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) != 0;

  iree_uk_int32_t acc_scratch[MT * NT * IME_ATOM_ELEMS];
  if (accumulate) ime_gather_acc(out, acc_scratch, MT, NT, N0);

  // Accumulators: v8,v10,...,v30 (12 groups spanning v8..v31 -- max register
  // occupancy, 24 acc regs + 7 operand regs = 31 <= 32).
  // Operands: A0=v0,A1=v2,A2=v4; B0=v1,B1=v3,B2=v5,B3=v6.
  // The scratch strides of 64 below are one atom = IME_M_ATOM*IME_N_ATOM*4B.
  __asm__ volatile(
      "    beqz        %[acc], .Lzero%=                 \n\t"
      "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
      "    mv          t1, %[scr]                       \n\t"
      "    vle32.v     v8,  (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v10, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v12, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v14, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v16, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v18, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v20, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v22, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v24, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v26, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v28, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vle32.v     v30, (t1)                        \n\t"
      "    j           .Lloop_setup%=                   \n\t"
      ".Lzero%=:                                        \n\t"
      "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
      "    vmv.v.i     v8,  0                           \n\t"
      "    vmv.v.i     v10, 0                           \n\t"
      "    vmv.v.i     v12, 0                           \n\t"
      "    vmv.v.i     v14, 0                           \n\t"
      "    vmv.v.i     v16, 0                           \n\t"
      "    vmv.v.i     v18, 0                           \n\t"
      "    vmv.v.i     v20, 0                           \n\t"
      "    vmv.v.i     v22, 0                           \n\t"
      "    vmv.v.i     v24, 0                           \n\t"
      "    vmv.v.i     v26, 0                           \n\t"
      "    vmv.v.i     v28, 0                           \n\t"
      "    vmv.v.i     v30, 0                           \n\t"
      ".Lloop_setup%=:                                  \n\t"
      "    mv          t1, %[lhs]                       \n\t"
      "    mv          t2, %[rhs]                       \n\t"
      "    mv          t3, %[K]                         \n\t"
      "    vsetvli     t0, x0, e8, m1, ta, ma           \n\t"
      "    beqz        t3, .Lstore%=                    \n\t"
      ".Lloop%=:                                        \n\t"
      "    vle8.v      v0, (t1)                         \n\t"
      "    addi        t1, t1, 32                       \n\t"
      "    vle8.v      v2, (t1)                         \n\t"
      "    addi        t1, t1, 32                       \n\t"
      "    vle8.v      v4, (t1)                         \n\t"
      "    addi        t1, t1, 32                       \n\t"
      "    vle8.v      v1, (t2)                         \n\t"
      "    addi        t2, t2, 32                       \n\t"
      "    vle8.v      v3, (t2)                         \n\t"
      "    addi        t2, t2, 32                       \n\t"
      "    vle8.v      v5, (t2)                         \n\t"
      "    addi        t2, t2, 32                       \n\t"
      "    vle8.v      v6, (t2)                         \n\t"
      "    addi        t2, t2, 32                       \n\t"
      "    smt.vmadot  v8,  v0, v1                      \n\t"
      "    smt.vmadot  v10, v0, v3                      \n\t"
      "    smt.vmadot  v12, v0, v5                      \n\t"
      "    smt.vmadot  v14, v0, v6                      \n\t"
      "    smt.vmadot  v16, v2, v1                      \n\t"
      "    smt.vmadot  v18, v2, v3                      \n\t"
      "    smt.vmadot  v20, v2, v5                      \n\t"
      "    smt.vmadot  v22, v2, v6                      \n\t"
      "    smt.vmadot  v24, v4, v1                      \n\t"
      "    smt.vmadot  v26, v4, v3                      \n\t"
      "    smt.vmadot  v28, v4, v5                      \n\t"
      "    smt.vmadot  v30, v4, v6                      \n\t"
      "    addi        t3, t3, -1                       \n\t"
      "    bnez        t3, .Lloop%=                     \n\t"
      ".Lstore%=:                                       \n\t"
      "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
      "    mv          t1, %[scr]                       \n\t"
      "    vse32.v     v8,  (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v10, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v12, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v14, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v16, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v18, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v20, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v22, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v24, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v26, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v28, (t1)                        \n\t"
      "    addi        t1, t1, 64                       \n\t"
      "    vse32.v     v30, (t1)                        \n\t"
      :
      : [lhs] "r"(lhs), [rhs] "r"(rhs), [scr] "r"(acc_scratch), [K] "r"(K1),
        [acc] "r"(accumulate)
      : "memory", "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
  ime_scatter_acc(out, acc_scratch, MT, NT, N0);
}
