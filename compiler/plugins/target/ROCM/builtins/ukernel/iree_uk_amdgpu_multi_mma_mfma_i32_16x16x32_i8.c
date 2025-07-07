// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/builtins/ukernel/common.h"

// Encodes some information about a A/B fragment tile;
typedef struct ab_tile_info_t {
  // Terminology: "_vecs":
  // We will be counting in units of "vectors", meaning, for each A/B fragment
  // the corresponding operand type of this particular MFMA intrinsic.
  // For A and B, that type is i64, used as <8 x i8>.

  // Number of vectors in the tile.
  int num_vecs;
  // Number of vectors that we store in shared memory. That is typically equal
  // to num_vecs if using shared memory for the tile, or 0 otherwise.
  int num_shared_vecs;
} ab_tile_info_t;

static ab_tile_info_t get_ab_tile_info(int tile_intrinsics, int tile_subgroups,
                                       int opposite_tile_subgroups) {
  ab_tile_info_t info;
  info.num_vecs = /*subgroup size*/ 64 * tile_intrinsics * tile_subgroups;
  // Use shared memory if the opposite tile has more than 1 subgroup, so that
  // using shared memory would amortize loads from global memory.
  info.num_shared_vecs = opposite_tile_subgroups > 1 ? info.num_vecs : 0;
  return info;
}

static int32_t get_shared_memory_bytes(ab_tile_info_t a_tile,
                                       ab_tile_info_t b_tile) {
  // For this MFMA intrinsic, the A and B vector types are 8 bytes.
  return 8 * (a_tile.num_shared_vecs + b_tile.num_shared_vecs);
}

// The bitcode of this function is interpreted during IREE compilation to
// determine the exact shared_memory_bytes to pass to the ukernel.
int32_t iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8_query_shared_memory_bytes(
    int32_t intrinsics_m, int32_t subgroups_m, int32_t intrinsics_n,
    int32_t subgroups_n, int32_t intrinsics_k) {
  ab_tile_info_t a_tile =
      get_ab_tile_info(intrinsics_m * intrinsics_k, subgroups_m, subgroups_n);
  ab_tile_info_t b_tile =
      get_ab_tile_info(intrinsics_n * intrinsics_k, subgroups_n, subgroups_m);
  return get_shared_memory_bytes(a_tile, b_tile);
}

// Microkernel for iree_codegen.inner_tiled with DataTiledMMAAttr with
// intrinsic = MFMA_I32_16x16x32_I8 and a shape with outer M and N dimensions
// equal to 1 (so that this is just doing the inner loop on the K dimension).
//
// This microkernel uses a shared memory workspace buffer provided by the
// caller. It is used to copy tiles of the A and/or B matrices, depending on
// which ones are reused by multiple subgroups.
//
// Note that the A, B, C matrix pointers are all after thread-distribution.
// When the pointer before thread-distribution is needed (when copying data
// into shared memory), care is taken to subtract the thread-relative offset,
// which is computed from the thread id.
//
// Each MFMA intrinsic produces a vector which is contiguous within in the C
// matrix, but there may be strides between the vectors produced by each
// intrinsic. This inter-intrinsic dimension stride is passed to the ukernel
// in order to compute the correct offsets into the C matrix for each intrinsic
// result. As an example of why this could be needed, the compiler may pass
// global memory or thread-local, private memory for the C matrix. For private
// memory, the thead gets a contiguous block of memory, so the stride would just
// be the size of the vector produced by each instruction. For global memory,
// the C matrix is shared by many threads in the same subgroup, and the subgroup
// collectively produces the C matrix tile, so the stride would be the size of
// the full tile produced by the subgroup.
//
// As this function is always_inline, some of its parameters are actually
// constant values after inlining, so some for() loops and if() branches here
// are actually unrolled/resolved at compile time, making this microkernel
// a generic "template". This is summarized in the below table.
//
// Parameters                  | Constant?  | Description
// --------------------------- | ---------- | -----------
// a_base, a_offset            | No         | A-matrix pointer (thread-distrib.)
// b_base, b_offset            | No         | B-matrix pointer (thread-distrib.)
// c_base, c_offset            | No         | C-matrix pointer (thread-distrib.)
// shared_memory_{base,offset} | No         | Shared memory workspace pointer
// shared_memory_bytes         | Yes        | Shared memory workspace size
// k_size                      | From shape | Size of outer K dimension
// intrinsics_m, subgroups_m   | Yes        | See DataTiledMMAAttr
// intrinsics_n, subgroups_n   | Yes        | See DataTiledMMAAttr
// intrinsics_k                | Yes        | See DataTiledMMAAttr
// c_inter_intrinsic_stride    | Yes        | C-matrix stride of inter-intrinsic
//                             |            | dimension
//
// TODO(bjacob): Better scheduling via either barrier intrinsics or inline asm.
[[clang::always_inline]] void iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8(
    const int8_t *a_base, int64_t a_offset, const int8_t *b_base,
    int64_t b_offset, int32_t *c_base, int64_t c_offset,
    int64_t c_inter_intrinsic_stride, int8_t *shared_memory_base,
    int64_t shared_memory_offset, int32_t shared_memory_bytes, int32_t k_size,
    int32_t intrinsics_m, int32_t subgroups_m, int32_t intrinsics_n,
    int32_t subgroups_n, int32_t intrinsics_k) {
  ab_tile_info_t a_tile =
      get_ab_tile_info(intrinsics_m * intrinsics_k, subgroups_m, subgroups_n);
  ab_tile_info_t b_tile =
      get_ab_tile_info(intrinsics_n * intrinsics_k, subgroups_n, subgroups_m);

  // shared_memory_bytes should match exactly, as the value ultimately comes
  // from the ..._query_shared_memory_bytes function defined just above.
  if (shared_memory_bytes != get_shared_memory_bytes(a_tile, b_tile)) {
    __builtin_trap();
  }
  // Set up our pointers to shared memory for A and B tiles.
  int64_t *restrict a_shared =
      (int64_t *)(shared_memory_base + shared_memory_offset);
  int64_t *restrict b_shared = a_shared + a_tile.num_shared_vecs;

  // Determine our thread id and the range for it.
  int tid = __builtin_amdgcn_workitem_id_x();
  int numthreads = 64 * subgroups_m * subgroups_n;
  __builtin_assume(tid < numthreads);

  // Compute the thread-relative data offsets.
  int lane_id = tid % 64;
  int subgroup_id = tid / 64;
  int subgroup_n_idx = subgroup_id % subgroups_n;
  int subgroup_m_idx = subgroup_id / subgroups_n;
  int a_thread_relative_offset =
      intrinsics_k * (lane_id + 64 * intrinsics_m * subgroup_m_idx);
  int b_thread_relative_offset =
      intrinsics_k * (lane_id + 64 * intrinsics_n * subgroup_n_idx);

  // Set up pointers to global memory.
  const int64_t *restrict a_global = (const int64_t *)(a_base + a_offset);
  const int64_t *restrict b_global = (const int64_t *)(b_base + b_offset);
  int32x4_t *restrict c_global = ((int32x4_t *)(c_base + c_offset));

  // Load existing accumulators from global memory into registers.
  // The VLA becomes a normal array after inlining.
  int32x4_t c_regs[intrinsics_m][intrinsics_n];
  // Divide the c_inter_intrinsic_stride by 4, because c_global contains vectors
  // of 4 elements.
  int c_inner_stride = c_inter_intrinsic_stride / 4;
  for (int m = 0; m < intrinsics_m; ++m) {
    for (int n = 0; n < intrinsics_n; ++n) {
      c_regs[m][n] = c_global[c_inner_stride * (m * intrinsics_n + n)];
    }
  }

  // Arithmetic loop.
  for (int k_outer = 0; k_outer < k_size; ++k_outer) {
    // Pointers to A/B data to feed MFMA, based on whether shared memory is
    // used.
    const int64_t *restrict a_mfma_vecs =
        a_tile.num_shared_vecs ? a_shared + a_thread_relative_offset : a_global;
    const int64_t *restrict b_mfma_vecs =
        b_tile.num_shared_vecs ? b_shared + b_thread_relative_offset : b_global;

    // If needed, load data from global to shared memory.
    if (tid < a_tile.num_shared_vecs) { // Benefits from above __builtin_assume.
      for (int i = 0; i < a_tile.num_shared_vecs; i += numthreads) {
        a_shared[i + tid] = a_global[i + tid - a_thread_relative_offset];
      }
    }
    if (tid < b_tile.num_shared_vecs) { // Benefits from above __builtin_assume.
      for (int i = 0; i < b_tile.num_shared_vecs; i += numthreads) {
        b_shared[i + tid] = b_global[i + tid - b_thread_relative_offset];
      }
    }
    // Thread barrier if any shared memory is used.
    if (a_tile.num_shared_vecs || b_tile.num_shared_vecs) {
      __syncthreads();
    }
    // Load data from shared memory and perform arithmetic.
    for (int m = 0; m < intrinsics_m; ++m) {
      for (int n = 0; n < intrinsics_n; ++n) {
        for (int k = 0; k < intrinsics_k; ++k) {
          c_regs[m][n] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              a_mfma_vecs[64 * intrinsics_k * m + k],
              b_mfma_vecs[64 * intrinsics_k * n + k], c_regs[m][n], 0, 0, 0);
        }
      }
    }
    a_global += a_tile.num_vecs;
    b_global += b_tile.num_vecs;
    // Thread barrier if any shared memory is used.
    if (a_tile.num_shared_vecs || b_tile.num_shared_vecs) {
      __syncthreads();
    }
  }

  // Store accumulators.
  for (int m = 0; m < intrinsics_m; ++m) {
    for (int n = 0; n < intrinsics_n; ++n) {
      c_global[c_inner_stride * (m * intrinsics_n + n)] = c_regs[m][n];
    }
  }
}
