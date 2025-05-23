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
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (256, d0 + 64)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (256, d1 + 64, d0 + 1)>
// CHECK-LABEL: @fold_pad_op
//  CHECK-SAME:   %[[SOURCE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[RESULT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[TRUE:.+]] = arith.constant true
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C252:.+]] = arith.constant 252 : index
//       CHECK:   %[[MAP_SCATTER_DEST:.+]] = tensor.empty() : tensor<256xf32>
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[SOURCE]] into %[[MAP_SCATTER_DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[TRUE]]
//       CHECK:   } : tensor<250xf32> into tensor<256xf32> -> tensor<256xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[RESULT]] : tensor<256xf32> into memref<256xf32>

//       CHECK:   scf.forall (%[[WG_IV:.+]]) = (0) to (256) step (64) {
//       CHECK:     %[[WG_TILE_UB:.+]] = affine.min #[[$MAP]](%[[WG_IV]])
//       CHECK:     scf.forall (%[[THREAD_IV:.+]]) = (%[[WG_IV]]) to (%[[WG_TILE_UB]]) step (1) {
//       CHECK:       %[[THREAD_TILE_UB:.+]] = affine.min #[[$MAP1]](%[[THREAD_IV]], %[[WG_IV]])
//       CHECK:       scf.for %[[IDX:.+]] = %[[THREAD_IV]] to %[[THREAD_TILE_UB]] step %[[C1]] {
//   CHECK-DAG:         %[[IS_LOW_PAD:.+]] = arith.cmpi ult, %[[IDX]], %[[C2]] : index
//   CHECK-DAG:         %[[IS_HIGH_PAD:.+]] = arith.cmpi uge, %[[IDX]], %[[C252]] : index
//   CHECK-DAG:         %[[IS_PAD:.+]] = arith.ori %[[IS_LOW_PAD]], %[[IS_HIGH_PAD]] : i1
//       CHECK:         scf.if %[[IS_PAD]] {
//  CHECK-NEXT:           memref.store %[[PAD_VAL]], %[[RESULT]][%[[IDX]]] : memref<256xf32>
//  CHECK-NEXT:         }
//  CHECK:            }
//  CHECK:          } {mapping = [#gpu.thread<linear_dim_0>]}
//  CHECK:        } {mapping = [#iree_codegen.workgroup_mapping<x>]}
