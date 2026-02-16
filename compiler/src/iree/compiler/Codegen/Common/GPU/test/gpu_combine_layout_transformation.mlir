// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-combine-layout-transformation,canonicalize,cse))" -split-input-file %s | FileCheck %s

func.func @fold_pad_op(%source : tensor<250xf32>, %result : memref<256xf32>) {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %source low[2] high[4] {
  ^bb0(%arg0: index):
    tensor.yield %cst : f32
  } : tensor<250xf32> to tensor<256xf32>
  iree_codegen.store_to_buffer %padded, %result : tensor<256xf32> into memref<256xf32>
  return
}
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (2, d0 + 1)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (256, d0 + 1)>
// CHECK-LABEL: @fold_pad_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<256xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[TRUE]]
//       CHECK:   } : tensor<250xf32> into tensor<256xf32> -> tensor<256xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<256xf32> into memref<256xf32>

// Low padding

//       CHECK:   scf.forall ({{.+}}) = (0) to (2) step (64) {
//       CHECK:     scf.forall (%[[THREAD_LOOP0_IV:.+]]) in (2) {
//       CHECK:       %[[THREAD_TILE0_UB:.+]] = affine.min #[[$MAP]](%[[THREAD_LOOP0_IV]])
//       CHECK:       scf.for %[[LOW_IDX:.+]] = %[[THREAD_LOOP0_IV]] to %[[THREAD_TILE0_UB]] step %[[C1]] {
//  CHECK-NEXT:         memref.store %[[PAD_VAL]], %[[RESULT]][%[[LOW_IDX]]] : memref<256xf32>
//      CHECK:        }
//      CHECK:      } {mapping = [#gpu.thread<linear_dim_0>]}
//      CHECK:    } {mapping = [#iree_codegen.workgroup_mapping<x>]}

// High padding

//       CHECK:   scf.forall ({{.+}}) = (252) to (256) step (64) {
//       CHECK:     scf.forall (%[[THREAD_LOOP1_IV:.+]]) = (252) to (256) step (1) {
//       CHECK:       %[[THREAD_TILE1_UB:.+]] = affine.min #[[$MAP1]](%[[THREAD_LOOP1_IV]])
//       CHECK:       scf.for %[[HIGH_IDX:.+]] = %[[THREAD_LOOP1_IV]] to %[[THREAD_TILE1_UB]] step %[[C1]] {
//  CHECK-NEXT:         memref.store %[[PAD_VAL]], %[[RESULT]][%[[HIGH_IDX]]] : memref<256xf32>
//      CHECK:        }
//      CHECK:      } {mapping = [#gpu.thread<linear_dim_0>]}
//      CHECK:    } {mapping = [#iree_codegen.workgroup_mapping<x>]}

// -----

func.func @no_fold_simple_relayout_op_chain(%source : tensor<256x128xf32>, %result : memref<120x250xf32>) {
  %empty0 = tensor.empty() : tensor<256x128xf32>
  %copy = linalg.copy ins(%source : tensor<256x128xf32>) outs(%empty0 : tensor<256x128xf32>) -> tensor<256x128xf32>
  %empty1 = tensor.empty() : tensor<128x256xf32>
  %transpose = linalg.transpose ins(%copy : tensor<256x128xf32>) outs(%empty1 : tensor<128x256xf32>) permutation = [1, 0]
  %extract_slice = tensor.extract_slice %transpose [0, 0][120, 250][1, 1] : tensor<128x256xf32> to tensor<120x250xf32>
  iree_codegen.store_to_buffer %extract_slice, %result : tensor<120x250xf32> into memref<120x250xf32>
  return
}

// CHECK-LABEL: @no_fold_simple_relayout_op_chain
//   CHECK-NOT:   iree_linalg_ext.map_store
//       CHECK:   linalg.copy
//       CHECK:   linalg.transpose
//       CHECK:   tensor.extract_slice
//       CHECK:   iree_codegen.store_to_buffer

// -----

func.func @fold_pack_op(%source : tensor<256x128xf32>, %result : memref<2x2x128x64xf32>) {
  %dest = tensor.empty() : tensor<2x2x128x64xf32>
  %pack = linalg.pack %source
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 64]
      into %dest : tensor<256x128xf32> -> tensor<2x2x128x64xf32>
  iree_codegen.store_to_buffer %pack, %result : tensor<2x2x128x64xf32> into memref<2x2x128x64xf32>
  return
}

// CHECK-LABEL: @fold_pack_op
//   CHECK-NOT:   linalg.pack
//       CHECK:   iree_linalg_ext.map_store

// -----

func.func @fold_unpack_op(%source : tensor<2x2x128x64xf32>, %result : memref<256x128xf32>) {
  %dest = tensor.empty() : tensor<256x128xf32>
  %unpack = linalg.unpack %source
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 64]
      into %dest : tensor<2x2x128x64xf32> -> tensor<256x128xf32>
  iree_codegen.store_to_buffer %unpack, %result : tensor<256x128xf32> into memref<256x128xf32>
  return
}

// CHECK-LABEL: @fold_unpack_op
//   CHECK-NOT:   linalg.unpack
//       CHECK:   iree_linalg_ext.map_store

// -----

func.func @fold_expand_shape_op(%source : tensor<8x16xf32>, %result : memref<2x4x16xf32>) {
  %expand = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  iree_codegen.store_to_buffer %expand, %result : tensor<2x4x16xf32> into memref<2x4x16xf32>
  return
}

// CHECK-LABEL: @fold_expand_shape_op
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   iree_linalg_ext.map_store

// -----

func.func @fold_collapse_shape_op(%source : tensor<2x4x16xf32>, %result : memref<8x16xf32>) {
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  iree_codegen.store_to_buffer %collapse, %result : tensor<8x16xf32> into memref<8x16xf32>
  return
}

// CHECK-LABEL: @fold_collapse_shape_op
//   CHECK-NOT:   tensor.collapse_shape
//       CHECK:   iree_linalg_ext.map_store
