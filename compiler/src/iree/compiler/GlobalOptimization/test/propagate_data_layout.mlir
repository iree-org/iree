// RUN: iree-opt --pass-pipeline="builtin.module(iree-global-opt-propagate-data-layout,cse)" --split-input-file %s | FileCheck %s

module @pack_propagation {
  util.global private mutable @_global_state.global = #util.uninitialized : tensor<256x32x128xf32>
  util.func public @pack_across_globals(%arg0: tensor<32x1x64x1x2xf32>) -> tensor<32x1x16x1x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %_global_state.global = util.global.load @_global_state.global : tensor<256x32x128xf32>
    %0 = tensor.empty() : tensor<32x16x64x16x2xf32>
    %pack = tensor.pack %_global_state.global outer_dims_perm = [1, 0, 2] inner_dims_pos = [0, 2] inner_tiles = [16, 2] into %0 : tensor<256x32x128xf32> -> tensor<32x16x64x16x2xf32>
    %1 = tensor.empty() : tensor<32x1x16x1x16xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x1x16x1x16xf32>) -> tensor<32x1x16x1x16xf32>
    %3 = linalg.batch_mmt4d ins(%arg0, %pack : tensor<32x1x64x1x2xf32>, tensor<32x16x64x16x2xf32>) outs(%2 : tensor<32x1x16x1x16xf32>) -> tensor<32x1x16x1x16xf32>
    util.return %3 : tensor<32x1x16x1x16xf32>
  }
}

// //       CHECK: module @pack_propagation
// //       CHECK:   private mutable @[[PACKED_GLOBAL:.+]] = #util.uninitialized : tensor<32x16x64x16x2xf32>
// //       CHECK:   util.initializer {
// //       CHECK:     %[[SPLAT:.+]] = flow.tensor.splat {{.+}} tensor<32x16x64x16x2xf32>
// //       CHECK:     util.global.store %[[SPLAT]], @[[PACKED_GLOBAL]] : tensor<32x16x64x16x2xf32>
// //       CHECK:   util.func public @pack_across_globals
// //  CHECK-SAME:     %[[ARG0:.+]]: tensor<32x1x64x1x2xf32>
// //   CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// //       CHECK:     %[[PACKED_STATE:.+]] = util.global.load @[[PACKED_GLOBAL]] : tensor<32x16x64x16x2xf32>
// //       CHECK:     %[[EMPTY:.+]] = tensor.empty() : tensor<32x1x16x1x16xf32>
// //       CHECK:     %[[MMT4D_DST:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32x1x16x1x16xf32>) -> tensor<32x1x16x1x16xf32>
// //       CHECK:     %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[ARG0]], %[[PACKED_STATE]] : tensor<32x1x64x1x2xf32>, tensor<32x16x64x16x2xf32>) outs(%[[MMT4D_DST]] : tensor<32x1x16x1x16xf32>) -> tensor<32x1x16x1x16xf32>
// //       CHECK:     util.return %[[MMT4D]] : tensor<32x1x16x1x16xf32>
