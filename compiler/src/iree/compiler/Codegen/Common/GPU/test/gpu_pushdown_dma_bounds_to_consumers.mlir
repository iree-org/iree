// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-pushdown-dma-bounds-to-consumers))" --split-input-file %s | FileCheck %s


// Test 1: in_bounds=[true, true] — pass is a no-op.


func.func @no_op_all_in_bounds(
    %src : tensor<64x16xf16>,
    %init : tensor<64x16xf16>,
    %lane : index) -> tensor<64x16xf16> {
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<64x16xf16> {
    %inner = scf.forall (%l) in (64) shared_outs(%inn = %outer)
        -> tensor<64x16xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
            in_bounds [true, true]
          : tensor<64x16xf16>, tensor<64x16xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [64, 16] [1, 1]
        : tensor<64x16xf16> into tensor<64x16xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<64x16xf16>
}
// CHECK-LABEL: func.func @no_op_all_in_bounds
// CHECK-NOT:   iree_gpu.buffer_resource_cast
// CHECK-NOT:   tensor.pad
// CHECK-NOT:   tensor.extract_slice

// -----


// Test 2: in_bounds=[false, true] — outer-only OOB; innermost is true.
//         Pass skips because in_bounds[innermost]=true.


func.func @no_op_outer_oob_only(
    %src : tensor<64x16xf16>,
    %init : tensor<64x16xf16>,
    %lane : index) -> tensor<64x16xf16> {
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<64x16xf16> {
    %inner = scf.forall (%l) in (64) shared_outs(%inn = %outer)
        -> tensor<64x16xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
            in_bounds [false, true]
          : tensor<64x16xf16>, tensor<64x16xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [64, 16] [1, 1]
        : tensor<64x16xf16> into tensor<64x16xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<64x16xf16>
}
// CHECK-LABEL: func.func @no_op_outer_oob_only
// CHECK-NOT:   iree_gpu.buffer_resource_cast
// CHECK-NOT:   tensor.pad
// CHECK-NOT:   tensor.extract_slice

// -----


// Test 3: in_bounds=[true, false] — innermost OOB; this is the target case.
//         Pass must insert extract_slice + tensor.pad after the outer forall.


func.func @pushdown_innermost_oob(
    %src : tensor<64x?xf16>,
    %init : tensor<64x16xf16>,
    %lane : index) -> tensor<64x16xf16> {
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<64x16xf16> {
    %inner = scf.forall (%l) in (64) shared_outs(%inn = %outer)
        -> tensor<64x16xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
            in_bounds [true, false]
          : tensor<64x?xf16>, tensor<64x16xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [64, 16] [1, 1]
        : tensor<64x16xf16> into tensor<64x16xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<64x16xf16>
}
// The pass walks from the DMA's init (%inn) up the shared_outs chain AND
// across parallel_insert_slice links until it reaches the outermost forall
// result that has external consumers. It inserts the slice+pad after that
// forall and RAUWs uses to the padded result.
// CHECK-LABEL: func.func @pushdown_innermost_oob
// CHECK:         %[[VB:.+]] = affine.apply
// CHECK:         %[[CAST:.+]] = iree_gpu.buffer_resource_cast %arg0 validBytes(%[[VB]])
// CHECK-SAME:        : tensor<64x?xf16>
// CHECK:         %[[OUTER:.+]] = scf.forall
// CHECK:           iree_gpu.coalesced_gather_dma %[[CAST]]
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[DIM:.+]] = tensor.dim %[[CAST]], %[[C1]]
// CHECK:         %[[C16:.+]] = arith.constant 16 : index
// CHECK:         %[[PAD_AMT:.+]] = arith.subi %[[C16]], %[[DIM]]
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[OUTER]][0, 0] [64, %[[DIM]]] [1, 1]
// CHECK-SAME:        : tensor<64x16xf16> to tensor<64x?xf16>
// CHECK:         %[[PADDED:.+]] = tensor.pad %[[SLICE]] low[0, 0] high[0, %[[PAD_AMT]]]
// CHECK:         return %[[PADDED]]

// -----


// Test 4: in_bounds=[false, false] — both dims OOB.
//         Pass inserts pad only on innermost; outer dim left to HW OOB.


func.func @pushdown_both_oob(
    %src : tensor<?x?xf16>,
    %init : tensor<64x16xf16>,
    %lane : index) -> tensor<64x16xf16> {
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<64x16xf16> {
    %inner = scf.forall (%l) in (64) shared_outs(%inn = %outer)
        -> tensor<64x16xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
            in_bounds [false, false]
          : tensor<?x?xf16>, tensor<64x16xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [64, 16] [1, 1]
        : tensor<64x16xf16> into tensor<64x16xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<64x16xf16>
}
// CHECK-LABEL: func.func @pushdown_both_oob
// CHECK:       iree_gpu.buffer_resource_cast %arg0 validBytes(%{{.+}}) : tensor<?x?xf16>
// CHECK:       tensor.extract_slice
// CHECK-SAME:      [0, 0] [64, %{{.*}}] [1, 1]
// CHECK:       tensor.pad
// CHECK-SAME:  low[0, 0] high[0, %{{.*}}]

// -----


// Test 5: No in_bounds attribute — pass is a no-op.


func.func @no_op_no_in_bounds(
    %src : tensor<64x16xf16>,
    %init : tensor<64x16xf16>,
    %lane : index) -> tensor<64x16xf16> {
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<64x16xf16> {
    %inner = scf.forall (%l) in (64) shared_outs(%inn = %outer)
        -> tensor<64x16xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
          : tensor<64x16xf16>, tensor<64x16xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [64, 16] [1, 1]
        : tensor<64x16xf16> into tensor<64x16xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<64x16xf16>
}
// CHECK-LABEL: func.func @no_op_no_in_bounds
// CHECK-NOT:   iree_gpu.buffer_resource_cast
// CHECK-NOT:   tensor.pad

// -----

// Test 6: single-level forall with an external linalg.matmul consumer.
// This is the primary motivated pattern. The DMA fills a single forall
// (no outer wrapper), so walkUpSharedOuts reaches the forall result directly.
// The pass inserts slice+pad between the forall result and the matmul.
func.func @single_forall_matmul_consumer(
    %src  : tensor<4x?xf16>,
    %init : tensor<4x16xf16>,
    %rhs  : tensor<16x8xf16>,
    %acc  : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %lhs = scf.forall (%l) in (4) shared_outs(%s = %init)
      -> tensor<4x16xf16> {
    scf.forall.in_parallel {
      iree_gpu.coalesced_gather_dma %src into %s lane(%l)
          in_bounds [true, false]
        : tensor<4x?xf16>, tensor<4x16xf16>, index
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<4x16xf16>, tensor<16x8xf16>)
      outs(%acc : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}
// CHECK-LABEL: func.func @single_forall_matmul_consumer
// CHECK:       %[[VB:.+]] = affine.apply
// CHECK:       %[[CAST:.+]] = iree_gpu.buffer_resource_cast %arg0 validBytes(%[[VB]])
// CHECK-SAME:      : tensor<4x?xf16>
// CHECK:       %[[LHS:.+]] = scf.forall
// CHECK:         iree_gpu.coalesced_gather_dma %[[CAST]]
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[DIM:.+]] = tensor.dim %[[CAST]], %[[C1]]
// CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0] [4, %[[DIM]]] [1, 1]
// CHECK:       %[[PADDED:.+]] = tensor.pad %[[SLICE]] low[0, 0] high[0, %{{.+}}]
// CHECK:       linalg.matmul ins(%[[PADDED]], %arg2

// -----

// Test 7: dynamic-shape source whose ROOT's innermost row is statically
// DWORD-aligned. The pass must skip the validBytes wrap because the
// underlying buffer's end is naturally DWORD-aligned (no partial-DWORD
// straddle is possible). Models the 4000x4000 f16 matmul case where the
// K-block tile produces tensor<32x?xf16> from a tensor<4000x4000xf16>
// root: dynamic source dim, but the root's 4000-element row * 2 bytes =
// 8000 bytes is divisible by 4.

func.func @skip_validbytes_when_root_innermost_aligned(
    %root : tensor<4000x4000xf16>,
    %init : tensor<32x16xf16>,
    %dyn  : index,
    %lane : index) -> tensor<32x16xf16> {
  %c0 = arith.constant 0 : index
  %src = tensor.extract_slice %root[%c0, %c0] [32, %dyn] [1, 1]
      : tensor<4000x4000xf16> to tensor<32x?xf16>
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<32x16xf16> {
    %inner = scf.forall (%l) in (32) shared_outs(%inn = %outer)
        -> tensor<32x16xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
            in_bounds [true, false]
          : tensor<32x?xf16>, tensor<32x16xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [32, 16] [1, 1]
        : tensor<32x16xf16> into tensor<32x16xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<32x16xf16>
}
// CHECK-LABEL: func.func @skip_validbytes_when_root_innermost_aligned
// CHECK-NOT:   iree_gpu.buffer_resource_cast

// -----

// Test 8: dynamic-shape source whose innermost extent's static UB matches
// the LDS inner tile size. The pass must skip the consumer-side
// extract_slice + tensor.pad chain. AMDGPULowerCoalescedDMA's OOB clamping
// zeros LDS positions for OOB lanes, so the pad would only redundantly
// rewrite the same zeros at the cost of materializing a private alloca.
// Models the 4000x4000 f16 RHS DMA case where N_tile = 128 and the source
// is tensor<32x?xf16> with the dynamic dim derived from
// affine.min<...,128> on the workgroup index.

#min128 = affine_map<(d0) -> (-d0 + 4000, 128)>
func.func @skip_consumer_pad_when_inner_ub_matches_tile(
    %root : tensor<4000x4000xf16>,
    %init : tensor<32x128xf16>,
    %off  : index,
    %lane : index) -> tensor<32x128xf16> {
  %c0 = arith.constant 0 : index
  %dyn = affine.min #min128(%off)
  %src = tensor.extract_slice %root[%c0, %off] [32, %dyn] [1, 1]
      : tensor<4000x4000xf16> to tensor<32x?xf16>
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<32x128xf16> {
    %inner = scf.forall (%l) in (32) shared_outs(%inn = %outer)
        -> tensor<32x128xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
            in_bounds [true, false]
          : tensor<32x?xf16>, tensor<32x128xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [32, 128] [1, 1]
        : tensor<32x128xf16> into tensor<32x128xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<32x128xf16>
}
// CHECK-LABEL: func.func @skip_consumer_pad_when_inner_ub_matches_tile
// Skip #1 fires here too (root 4000 * 2 = 8000 is DWORD-aligned), so no
// validBytes wrap. Skip #2 fires for the consumer-side pad chain.
// CHECK-NOT:   iree_gpu.buffer_resource_cast
// CHECK-NOT:   tensor.pad

// -----

// Test 9: same as Test 8 but with a DWORD-misaligned root (4000*2=8000
// aligned -> swap to 4001 to break alignment). Skip #1 must NOT fire
// (validBytes wrap is needed to guard against the partial-DWORD straddle
// at the buffer end), but Skip #2 must still fire because the inner
// extent's UB still matches the LDS tile.

#min128b = affine_map<(d0) -> (-d0 + 4001, 128)>
func.func @consumer_pad_skipped_even_when_root_misaligned(
    %root : tensor<4000x4001xf16>,
    %init : tensor<32x128xf16>,
    %off  : index,
    %lane : index) -> tensor<32x128xf16> {
  %c0 = arith.constant 0 : index
  %dyn = affine.min #min128b(%off)
  %src = tensor.extract_slice %root[%c0, %off] [32, %dyn] [1, 1]
      : tensor<4000x4001xf16> to tensor<32x?xf16>
  %filled = scf.forall (%w) in (1) shared_outs(%outer = %init)
      -> tensor<32x128xf16> {
    %inner = scf.forall (%l) in (32) shared_outs(%inn = %outer)
        -> tensor<32x128xf16> {
      scf.forall.in_parallel {
        iree_gpu.coalesced_gather_dma %src into %inn lane(%l)
            in_bounds [true, false]
          : tensor<32x?xf16>, tensor<32x128xf16>, index
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inner into %outer [0, 0] [32, 128] [1, 1]
        : tensor<32x128xf16> into tensor<32x128xf16>
    }
  } {mapping = [#gpu.warp<linear_dim_0>]}
  return %filled : tensor<32x128xf16>
}
// CHECK-LABEL: func.func @consumer_pad_skipped_even_when_root_misaligned
// CHECK:       iree_gpu.buffer_resource_cast %arg0 validBytes
// CHECK-NOT:   tensor.pad
