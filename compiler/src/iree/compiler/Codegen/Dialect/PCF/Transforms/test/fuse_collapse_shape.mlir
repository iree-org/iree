// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-fuse-consumers)" --split-input-file | FileCheck %s

// Test: Fuse tensor.collapse_shape into pcf.generic with tied init.
// The 2D result is collapsed to 1D, and the write_slice offsets/sizes are
// linearized accordingly. The constant source also gets collapsed.

func.func @fuse_collapse_shape_into_generic(%arg0: tensor<8x10xi32>) -> tensor<80xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x10xi32>
    pcf.write_slice %cst into %ref[%id0, 0] [4, 10] [1, 1] : tensor<4x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-LABEL: @fuse_collapse_shape_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>

// The constant gets collapsed by canonicalization.
//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<40xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} : tensor<8x10xi32> into tensor<80xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    %[[FLAT_OFF:.+]] = affine.apply #[[$MAP0]](%[[ID0]])
//       CHECK:    pcf.write_slice %[[CST]] into %[[REF]][%[[FLAT_OFF]]] [40] [1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Test: Fuse tensor.collapse_shape into pcf.loop.

func.func @fuse_collapse_shape_into_loop(%arg0: tensor<8x10xi32>, %n0: index, %n1: index) -> tensor<80xi32> {
  %0 = pcf.loop scope(#pcf.test_scope) count(%n0, %n1)
    execute(%ref = %arg0)[%id0: index, %id1: index]
            : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
           -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x10xi32>
    pcf.write_slice %cst into %ref[%id0, 0] [4, 10] [1, 1] : tensor<4x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-LABEL: @fuse_collapse_shape_into_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<40xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} : tensor<8x10xi32> into tensor<80xi32>
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    %[[FLAT_OFF:.+]] = affine.apply #[[$MAP0]](%[[ID0]])
//       CHECK:    pcf.write_slice %[[CST]] into %[[REF]][%[[FLAT_OFF]]] [40] [1]
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Test: Fuse tensor.collapse_shape with multiple write_slices.

func.func @fuse_collapse_shape_multiple_write_slices(%arg0: tensor<8x10xi32>) -> tensor<80xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    %cst1 = arith.constant dense<5> : tensor<3x10xi32>
    %cst2 = arith.constant dense<7> : tensor<5x10xi32>
    pcf.write_slice %cst1 into %ref[%id0, 0] [3, 10] [1, 1] : tensor<3x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.write_slice %cst2 into %ref[%id1, 0] [5, 10] [1, 1] : tensor<5x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-LABEL: @fuse_collapse_shape_multiple_write_slices
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>

// Constants are collapsed by canonicalization.
//   CHECK-DAG:  %[[CST1:.+]] = arith.constant dense<5> : tensor<30xi32>
//   CHECK-DAG:  %[[CST2:.+]] = arith.constant dense<7> : tensor<50xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}}
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    %[[OFF0:.+]] = affine.apply #[[$MAP0]](%[[ID0]])
//       CHECK:    pcf.write_slice %[[CST1]] into %[[REF]][%[[OFF0]]] [30] [1]
//       CHECK:    %[[OFF1:.+]] = affine.apply #[[$MAP0]](%[[ID1]])
//       CHECK:    pcf.write_slice %[[CST2]] into %[[REF]][%[[OFF1]]] [50] [1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Negative test: producer result is directly returned (multiple uses prevent
// fusion). Using a direct return of %0 ensures the second use cannot be fused
// by other patterns.

func.func @no_fuse_collapse_shape_multiple_uses(%arg0: tensor<8x10xi32>) -> (tensor<80xi32>, tensor<8x10xi32>) {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    %cst = arith.constant dense<5> : tensor<4x10xi32>
    pcf.write_slice %cst into %ref[%id0, 0] [4, 10] [1, 1] : tensor<4x10xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1, %0 : tensor<80xi32>, tensor<8x10xi32>
}

// CHECK-LABEL: @no_fuse_collapse_shape_multiple_uses

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//       CHECK:  %[[COLLAPSE:.+]] = tensor.collapse_shape %[[GENERIC]] {{\[}}[0, 1]{{\]}}
//       CHECK:  return %[[COLLAPSE]], %[[GENERIC]]

// -----

// Test: Inner dimension partially covered -> emits scf.forall loop.
// write_slice covers [4, 5] at [%id0, %id1] in shape [8, 10].
// Inner dim (dim 1) has size 5 != shape 10, so dim 0 becomes a loop dim.
// Uses non-constant source to verify extract_slice + collapse_shape generation.

func.func @fuse_collapse_shape_partial_inner_dim(%arg0: tensor<8x10xi32>,
    %src: tensor<4x5xi32>) -> tensor<80xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    pcf.write_slice %src into %ref[%id0, %id1] [4, 5] [1, 1] : tensor<4x5xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

// Dim 0 is a loop dim (size 4), dim 1 is retained (size 5).
// Linearized offset: id1 + id0 * 10 + iv * 10.
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 10 + d2 * 10)>
// CHECK-LABEL: @fuse_collapse_shape_partial_inner_dim
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<4x5xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}}
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    scf.forall (%[[IV:.+]]) in (4) {
//       CHECK:      %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][%[[IV]], 0] [1, 5] [1, 1] : tensor<4x5xi32> to tensor<1x5xi32>
//       CHECK:      %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SLICE]] {{\[}}[0, 1]{{\]}} : tensor<1x5xi32> into tensor<5xi32>
//       CHECK:      %[[OFF:.+]] = affine.apply #[[$MAP0]](%[[ID1]], %[[ID0]], %[[IV]])
//       CHECK:      pcf.write_slice %[[COLLAPSED]] into %[[REF]][%[[OFF]]] [5] [1]
//       CHECK:    }
//       CHECK:    pcf.return
//       CHECK:  return

// -----

// Test: 3D group with full static retention from inner dim coverage.
// write_slice covers [2, 4, 8] at [%id0, 0, 0] in shape [6, 4, 8].
// dim 2 (innermost): size 8 == shape 8, offset 0 -> retained.
// dim 1: size 4 == shape 4, offset 0 -> dim 2 fully covered, retained.
// dim 0: dim 1 fully covered -> retained.
// All dims retained. No loops. Single write of 2*4*8 = 64.

func.func @fuse_collapse_shape_3d_full_retention(%arg0: tensor<6x4x8xi32>) -> tensor<192xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<6x4x8xi32, sync(#pcf.test_scope)>)
        -> (tensor<6x4x8xi32>) {
    %cst = arith.constant dense<5> : tensor<2x4x8xi32>
    pcf.write_slice %cst into %ref[%id0, 0, 0] [2, 4, 8] [1, 1, 1] : tensor<2x4x8xi32> into !pcf.sref<6x4x8xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<6x4x8xi32> into tensor<192xi32>
  return %1 : tensor<192xi32>
}

// All 3 dims retained. Linearized offset: id0 * 4 * 8 = id0 * 32.
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: @fuse_collapse_shape_3d_full_retention
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<6x4x8xi32>
//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<64xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]{{\]}}
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//   CHECK-NOT:  scf.forall
//       CHECK:  %[[OFF:.+]] = affine.apply #[[$MAP0]](%[[ID0]])
//       CHECK:  pcf.write_slice %[[CST]] into %[[REF]][%[[OFF]]] [64] [1]
//       CHECK:  pcf.return

// -----

// Test: 3D group where only innermost dim is retained.
// write_slice covers [2, 3, 5] at [%id0, %id1, %n0] in shape [6, 4, 8].
// dim 2 (innermost): offset %n0 (not const 0) -> only innermost retained.
// dim 1: cannot extend (dim 2 not fully covered) -> loop dim.
// dim 0: cannot extend -> loop dim.
// Result: 2D forall loop over dims 0,1 (bounds 2,3), write 5 elements each.

func.func @fuse_collapse_shape_3d_two_loop_dims(%arg0: tensor<6x4x8xi32>,
    %src: tensor<2x3x5xi32>) -> tensor<192xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<6x4x8xi32, sync(#pcf.test_scope)>)
        -> (tensor<6x4x8xi32>) {
    pcf.write_slice %src into %ref[%id0, %id1, %n0] [2, 3, 5] [1, 1, 1] : tensor<2x3x5xi32> into !pcf.sref<6x4x8xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<6x4x8xi32> into tensor<192xi32>
  return %1 : tensor<192xi32>
}

// Dims 0,1 are loop dims (bounds 2,3). Dim 2 retained (size 5).
// Linearized offset: n0 + id0 * 4*8 + iv0 * 4*8 + id1 * 8 + iv1 * 8.
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d1 * 32 + d2 * 32 + d3 * 8 + d4 * 8)>
// CHECK-LABEL: @fuse_collapse_shape_3d_two_loop_dims
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<6x4x8xi32>
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<2x3x5xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]{{\]}}
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index, %[[N0:[A-Za-z0-9_]+]]: index
//       CHECK:    scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (2, 3) {
//       CHECK:      %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], %[[IV1]], 0] [1, 1, 5] [1, 1, 1]
//       CHECK:      %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SLICE]] {{\[}}[0, 1, 2]{{\]}} : tensor<1x1x5xi32> into tensor<5xi32>
//       CHECK:      %[[OFF:.+]] = affine.apply #[[$MAP0]](%[[N0]], %[[ID0]], %[[IV0]], %[[ID1]], %[[IV1]])
//       CHECK:      pcf.write_slice %[[COLLAPSED]] into %[[REF]][%[[OFF]]] [5] [1]
//       CHECK:    }
//       CHECK:    pcf.return

// -----

// Test: Multiple reassociation groups [[0,1], [2,3]].
// 4D shape [6, 4, 8, 5] collapses to 2D [24, 40].
// Group [0,1]: dim 1 fully covers (4==4, off=0), both retained, no loop.
// Group [2,3]: dim 3 fully covers (5==5, off=0), both retained, no loop.

func.func @fuse_collapse_shape_two_groups(%arg0: tensor<6x4x8x5xi32>) -> tensor<24x40xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<6x4x8x5xi32, sync(#pcf.test_scope)>)
        -> (tensor<6x4x8x5xi32>) {
    %cst = arith.constant dense<5> : tensor<3x4x2x5xi32>
    pcf.write_slice %cst into %ref[%id0, 0, %id1, 0] [3, 4, 2, 5] [1, 1, 1, 1] : tensor<3x4x2x5xi32> into !pcf.sref<6x4x8x5xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]] : tensor<6x4x8x5xi32> into tensor<24x40xi32>
  return %1 : tensor<24x40xi32>
}

// Group [0,1]: offset id0*4, size 3*4=12. Group [2,3]: offset id1*5, size 2*5=10.
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 5)>
// CHECK-LABEL: @fuse_collapse_shape_two_groups
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<6x4x8x5xi32>
//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<12x10xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}}
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<24x40xi32>)
//   CHECK-NOT:  scf.forall
//       CHECK:  %[[OFF0:.+]] = affine.apply #[[$MAP0]](%[[ID0]])
//       CHECK:  %[[OFF1:.+]] = affine.apply #[[$MAP1]](%[[ID1]])
//       CHECK:  pcf.write_slice %[[CST]] into %[[REF]][%[[OFF0]], %[[OFF1]]] [12, 10] [1, 1]
//       CHECK:  pcf.return

// -----

// Test: Multiple groups with mixed loop/no-loop.
// 4D shape [6, 4, 8, 5] collapses to 2D [24, 40].
// Group [0,1]: dim 1 has dynamic offset %id1 -> cannot extend. dim 0 is loop.
// Group [2,3]: dim 3 size 5 == shape 5, offset 0 -> both retained, no loop.

func.func @fuse_collapse_shape_mixed_groups(%arg0: tensor<6x4x8x5xi32>,
    %src: tensor<2x3x3x5xi32>) -> tensor<24x40xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<6x4x8x5xi32, sync(#pcf.test_scope)>)
        -> (tensor<6x4x8x5xi32>) {
    pcf.write_slice %src into %ref[%id0, %id1, %n0, 0] [2, 3, 3, 5] [1, 1, 1, 1] : tensor<2x3x3x5xi32> into !pcf.sref<6x4x8x5xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]] : tensor<6x4x8x5xi32> into tensor<24x40xi32>
  return %1 : tensor<24x40xi32>
}

// Group [0,1]: dim 0 loops (bound 2), dim 1 retained.
//   Offset: id1 + id0 * 4 + iv * 4. Size: 3.
// Group [2,3]: both retained. Offset: n0 * 5. Size: 3*5=15.
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 4 + d2 * 4)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 5)>
// CHECK-LABEL: @fuse_collapse_shape_mixed_groups
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<6x4x8x5xi32>
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<2x3x3x5xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}}
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index, %[[N0:[A-Za-z0-9_]+]]: index
//       CHECK:    scf.forall (%[[IV:.+]]) in (2) {
//       CHECK:      %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][%[[IV]], 0, 0, 0] [1, 3, 3, 5] [1, 1, 1, 1]
//       CHECK:      %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SLICE]] {{\[}}[0, 1], [2, 3]{{\]}} : tensor<1x3x3x5xi32> into tensor<3x15xi32>
//       CHECK:      %[[OFF0:.+]] = affine.apply #[[$MAP0]](%[[ID1]], %[[ID0]], %[[IV]])
//       CHECK:      %[[OFF1:.+]] = affine.apply #[[$MAP1]](%[[N0]])
//       CHECK:      pcf.write_slice %[[COLLAPSED]] into %[[REF]][%[[OFF0]], %[[OFF1]]] [3, 15] [1, 1]
//       CHECK:    }
//       CHECK:    pcf.return

// -----

// Test: Dynamic producer shape (dim 0 dynamic, dim 1 static).
// tensor<?x10xi32> collapsed to tensor<?xi32>.
// Both dims are in group [0,1]. Dim 1 is static with full coverage
// (offset 0, size 10 == shape 10), so both dims are retained despite
// dim 0 being dynamic. No loop needed.

func.func @fuse_collapse_shape_dynamic_producer(%arg0: tensor<?x10xi32>,
    %src: tensor<4x10xi32>) -> tensor<?xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<?x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<?x10xi32>) {
    pcf.write_slice %src into %ref[%id0, 0] [4, 10] [1, 1] : tensor<4x10xi32> into !pcf.sref<?x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<?x10xi32> into tensor<?xi32>
  return %1 : tensor<?xi32>
}

// Dynamic producer: dim 0 is dynamic but dim 1 (innermost) is static with
// full coverage (offset 0, size 10 == shape 10). Both dims retained.
// Linearized offset: id0 * 10, size: 4 * 10 = 40.
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-LABEL: @fuse_collapse_shape_dynamic_producer
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<?x10xi32>
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<4x10xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} : tensor<?x10xi32> into tensor<?xi32>
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<?xi32>)
//       CHECK:    %[[OFF:.+]] = affine.apply #[[$MAP0]](%[[ID0]])
//       CHECK:    %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SRC]] {{\[}}[0, 1]{{\]}} : tensor<4x10xi32> into tensor<40xi32>
//       CHECK:    pcf.write_slice %[[COLLAPSED]] into %[[REF]][%[[OFF]]] [40] [1]
//   CHECK-NOT:    scf.forall
//       CHECK:    pcf.return

// -----

// Test: Dynamic write_slice offsets and sizes with static producer shape.
// Producer is static 8x10, but the write_slice uses dynamic offsets/sizes.
// The inner dim has dynamic offset, so it's not provably full-coverage.
// This means dim 0 becomes a loop dim.

func.func @fuse_collapse_shape_dynamic_write(%arg0: tensor<8x10xi32>,
    %off0: index, %off1: index, %sz0: index, %sz1: index,
    %src: tensor<?x?xi32>) -> tensor<80xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x10xi32, sync(#pcf.test_scope)>)
        -> (tensor<8x10xi32>) {
    pcf.write_slice %src into %ref[%off0, %off1] [%sz0, %sz1] [1, 1] : tensor<?x?xi32> into !pcf.sref<8x10xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<8x10xi32> into tensor<80xi32>
  return %1 : tensor<80xi32>
}

// Dynamic offsets/sizes: dim 0 loops (bound sz0), dim 1 retained (size sz1).
// Linearized offset: iv * 10 + off0 * 10 + off1.
//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0)[s0, s1] -> (d0 * 10 + s0 * 10 + s1)>
// CHECK-LABEL: @fuse_collapse_shape_dynamic_write
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x10xi32>,
//  CHECK-SAME:   %[[OFF0:[A-Za-z0-9_]+]]: index, %[[OFF1:[A-Za-z0-9_]+]]: index, %[[SZ0:[A-Za-z0-9_]+]]: index, %[[SZ1:[A-Za-z0-9_]+]]: index,
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}}
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])
//       CHECK:    -> (tensor<80xi32>)
//       CHECK:    scf.forall (%[[IV:.+]]) in (%[[SZ0]]) {
//       CHECK:      %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][%[[IV]], 0] [1, %[[SZ1]]] [1, 1] : tensor<?x?xi32> to tensor<1x?xi32>
//       CHECK:      %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SLICE]] {{\[}}[0, 1]{{\]}} : tensor<1x?xi32> into tensor<?xi32>
//       CHECK:      %[[LOFF:.+]] = affine.apply #[[$MAP0]](%[[IV]])[%[[OFF0]], %[[OFF1]]]
//       CHECK:      pcf.write_slice %[[COLLAPSED]] into %[[REF]][%[[LOFF]]] [%[[SZ1]]] [1]
//       CHECK:    }
//       CHECK:    pcf.return

// -----

// Test: 3D static group fully retained (all dims fully cover).
// write_slice covers [6, 4, 8] at [0, 0, 0] in shape [6, 4, 8].
// All dims at offset 0 with full size. No loops; single write of 192.
// Offset is constant 0 (folded away), no affine.apply needed.

func.func @fuse_collapse_shape_3d_fully_retained(%arg0: tensor<6x4x8xi32>) -> tensor<192xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<6x4x8xi32, sync(#pcf.test_scope)>)
        -> (tensor<6x4x8xi32>) {
    %cst = arith.constant dense<5> : tensor<6x4x8xi32>
    pcf.write_slice %cst into %ref[0, 0, 0] [6, 4, 8] [1, 1, 1] : tensor<6x4x8xi32> into !pcf.sref<6x4x8xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<6x4x8xi32> into tensor<192xi32>
  return %1 : tensor<192xi32>
}

// CHECK-LABEL: @fuse_collapse_shape_3d_fully_retained
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<6x4x8xi32>
//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<192xi32>
//       CHECK:  %[[INIT:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]{{\]}}
//       CHECK:  pcf.generic
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])
//   CHECK-NOT:  scf.forall
//   CHECK-NOT:  affine.apply
//       CHECK:  pcf.write_slice %[[CST]] into %[[REF]][0] [192] [1]
//       CHECK:  pcf.return
