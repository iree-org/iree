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
// //       CHECK:     %[[PACKED_STATE:.+]] = util.global.load @[[PACKED_GLOBAL]]{{.+}} : tensor<32x16x64x16x2xf32>
// //       CHECK:     %[[EMPTY:.+]] = tensor.empty() : tensor<32x1x16x1x16xf32>
// //       CHECK:     %[[MMT4D_DST:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32x1x16x1x16xf32>) -> tensor<32x1x16x1x16xf32>
// //       CHECK:     %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[ARG0]], %[[PACKED_STATE]] : tensor<32x1x64x1x2xf32>, tensor<32x16x64x16x2xf32>) outs(%[[MMT4D_DST]] : tensor<32x1x16x1x16xf32>) -> tensor<32x1x16x1x16xf32>
// //       CHECK:     util.return %[[MMT4D]] : tensor<32x1x16x1x16xf32>

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 16)>
module @pack_propagation_extract_slice {
  util.global private mutable @_global_seq_step.global {noinline} = 0 : index
  util.global private mutable @_global_state.global {noinline} = #util.uninitialized : tensor<4095x32x128xf32>
  util.func @pack_across_globals(%arg0: tensor<32x1x64x1x2xf32>) -> tensor<32x1x?x1x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %_global_state.global = util.global.load @_global_state.global : tensor<4095x32x128xf32>
    %_global_seq_step.global = util.global.load @_global_seq_step.global : index
    %0 = affine.apply #map()[%_global_seq_step.global]
    %extracted_slice = tensor.extract_slice %_global_state.global[0, 0, 0] [%0, 32, 128] [1, 1, 1] : tensor<4095x32x128xf32> to tensor<?x32x128xf32>
    %2 = affine.apply #map1()[%_global_seq_step.global]
    %3 = tensor.empty(%2) : tensor<32x?x64x16x2xf32>
    %pack = tensor.pack %extracted_slice padding_value(%cst_0 : f32) outer_dims_perm = [1, 0, 2] inner_dims_pos = [0, 2] inner_tiles = [16, 2] into %3 : tensor<?x32x128xf32> -> tensor<32x?x64x16x2xf32>
    %4 = tensor.empty(%2) : tensor<32x1x?x1x16xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
    %6 = linalg.batch_mmt4d ins(%arg0, %pack : tensor<32x1x64x1x2xf32>, tensor<32x?x64x16x2xf32>) outs(%5 : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
    %7 = arith.addi %_global_seq_step.global, %c1 : index
    util.global.store %_global_state.global, @_global_state.global : tensor<4095x32x128xf32>
    util.global.store %7, @_global_seq_step.global : index
    util.return %6 : tensor<32x1x?x1x16xf32>
  }
}

// //   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> ((s0 + 1) ceildiv 16)>
// //   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// //       CHECK: module @pack_propagation_extract_slice
// //       CHECK:   private mutable @[[PACKED_GLOBAL:.+]] {noinline} = #util.uninitialized : tensor<32x256x64x16x2xf32>
// //       CHECK:   util.initializer {
// //       CHECK:     %[[SPLAT:.+]] = flow.tensor.splat {{.+}} tensor<32x256x64x16x2xf32>
// //       CHECK:     util.global.store %[[SPLAT]], @[[PACKED_GLOBAL]] : tensor<32x256x64x16x2xf32>
// //       CHECK:   util.func public @pack_across_globals
// //  CHECK-SAME:     %[[ARG0:.+]]: tensor<32x1x64x1x2xf32>
// //   CHECK-DAG:     %[[ONE:.+]] = arith.constant 1 : index
// //   CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// //       CHECK:     %[[PACKED_STATE:.+]] = util.global.load @[[PACKED_GLOBAL]]{{.+}} : tensor<32x256x64x16x2xf32>
// //       CHECK:     util.global.store %[[PACKED_STATE]], @[[PACKED_GLOBAL]] : tensor<32x256x64x16x2xf32>
// //       CHECK:     %[[SEQ_STEP:.+]] = util.global.load @_global_seq_step.global : index
// //       CHECK:     %[[NEXT_SEQ_STEP_DIV16:.+]] = affine.apply #[[MAP]]()[%[[SEQ_STEP]]]
// //       CHECK:     %[[PACKED_RHS:.+]] = tensor.extract_slice %[[PACKED_STATE]][0, 0, 0, 0, 0] [32, %[[NEXT_SEQ_STEP_DIV16]], 64, 16, 2] [1, 1, 1, 1, 1] : tensor<32x256x64x16x2xf32> to tensor<32x?x64x16x2xf32>
// //       CHECK:     %[[PACKED_N:.+]] = affine.apply #[[MAP1]]()[%[[SEQ_STEP]]]
// //       CHECK:     %[[EMPTY:.+]] = tensor.empty(%[[PACKED_N]]) : tensor<32x1x?x1x16xf32>
// //       CHECK:     %[[MMT4D_DST:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
// //       CHECK:     %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[ARG0]], %[[PACKED_RHS]] : tensor<32x1x64x1x2xf32>, tensor<32x?x64x16x2xf32>) outs(%[[MMT4D_DST]] : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
// //       CHECK:     %[[NEXT_SEQ_STEP:.+]] = arith.addi %[[SEQ_STEP]], %[[ONE]] : index
// //       CHECK:     util.global.store %[[NEXT_SEQ_STEP]], @_global_seq_step.global : index
// //       CHECK:     util.return %[[MMT4D]] : tensor<32x1x?x1x16xf32>

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 16)>
module @pack_propagation_insert_extract_slice {
  util.global private mutable @_global_seq_step.global {noinline} = 0 : index
  util.global private mutable @_global_state.global {noinline} = #util.uninitialized : tensor<4095x32x128xf32>
  util.func @pack_across_globals(%arg0: tensor<32x1x64x1x2xf32>, %arg1: tensor<32x128xf32>) -> tensor<32x1x?x1x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %_global_state.global = util.global.load @_global_state.global : tensor<4095x32x128xf32>
    %_global_seq_step.global = util.global.load @_global_seq_step.global : index
    %0 = affine.apply #map()[%_global_seq_step.global]
    %1 = tensor.empty(%0) : tensor<?x32x128xf32>
    %inserted_slice = tensor.insert_slice %arg1 into %_global_state.global[%_global_seq_step.global, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<32x128xf32> into tensor<4095x32x128xf32>
    %extracted_slice = tensor.extract_slice %inserted_slice[0, 0, 0] [%0, 32, 128] [1, 1, 1] : tensor<4095x32x128xf32> to tensor<?x32x128xf32>
    %2 = affine.apply #map1()[%_global_seq_step.global]
    %3 = tensor.empty(%2) : tensor<32x?x64x16x2xf32>
    %pack = tensor.pack %extracted_slice padding_value(%cst_0 : f32) outer_dims_perm = [1, 0, 2] inner_dims_pos = [0, 2] inner_tiles = [16, 2] into %3 : tensor<?x32x128xf32> -> tensor<32x?x64x16x2xf32>
    %4 = tensor.empty(%2) : tensor<32x1x?x1x16xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
    %6 = linalg.batch_mmt4d ins(%arg0, %pack : tensor<32x1x64x1x2xf32>, tensor<32x?x64x16x2xf32>) outs(%5 : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
    %7 = arith.addi %_global_seq_step.global, %c1 : index
    util.global.store %inserted_slice, @_global_state.global : tensor<4095x32x128xf32>
    util.global.store %7, @_global_seq_step.global : index
    util.return %6 : tensor<32x1x?x1x16xf32>
  }
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 16)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 mod 16)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> ((s0 + 1) ceildiv 16)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//       CHECK: module @pack_propagation_insert_extract_slice
//       CHECK:   private mutable @[[PACKED_GLOBAL:.+]] {noinline} = #util.uninitialized : tensor<32x256x64x16x2xf32>
//       CHECK:   util.initializer {
//       CHECK:     %[[SPLAT:.+]] = flow.tensor.splat {{.+}} tensor<32x256x64x16x2xf32>
//       CHECK:     util.global.store %[[SPLAT]], @[[PACKED_GLOBAL]] : tensor<32x256x64x16x2xf32>
//       CHECK:   util.func public @pack_across_globals
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<32x1x64x1x2xf32>, %[[ARG1:.+]]: tensor<32x128xf32>
//   CHECK-DAG:     %[[ONE:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[PACKED_STATE:.+]] = util.global.load @[[PACKED_GLOBAL]]{{.+}} : tensor<32x256x64x16x2xf32>
//       CHECK:     %[[SEQ_STEP:.+]] = util.global.load @_global_seq_step.global : index
//   CHECK-DAG:     %[[SEQ_STEP_DIV16:.+]] = affine.apply #[[MAP]]()[%[[SEQ_STEP]]]
//   CHECK-DAG:     %[[SEQ_STEP_MOD16:.+]] = affine.apply #[[MAP1]]()[%[[SEQ_STEP]]]
//       CHECK:     %[[PACKED_SLICE_DST:.+]] = tensor.empty() : tensor<32x64x2xf32>
//       CHECK:     %[[PACKED_SLICE:.+]] = tensor.pack %[[ARG1]] padding_value(%[[ZERO]] : f32) outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [2] into %[[PACKED_SLICE_DST]] {{.+}} : tensor<32x128xf32> -> tensor<32x64x2xf32>
//       CHECK:     %[[EXPANDED_SLICE:.+]] = tensor.expand_shape %[[PACKED_SLICE]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x64x2xf32> into tensor<32x1x64x1x2xf32>
//       CHECK:     %[[INSERT_SLICE:.+]] = tensor.insert_slice %[[EXPANDED_SLICE]] into %[[PACKED_STATE]][0, %[[SEQ_STEP_DIV16]], 0, %[[SEQ_STEP_MOD16]], 0] [32, 1, 64, 1, 2] [1, 1, 1, 1, 1] : tensor<32x1x64x1x2xf32> into tensor<32x256x64x16x2xf32>
//       CHECK:     util.global.store %[[INSERT_SLICE]], @[[PACKED_GLOBAL]] : tensor<32x256x64x16x2xf32>
//       CHECK:     %[[NEXT_SEQ_STEP_DIV16:.+]] = affine.apply #[[MAP2]]()[%[[SEQ_STEP]]]
//       CHECK:     %[[PACKED_RHS:.+]] = tensor.extract_slice %[[INSERT_SLICE]][0, 0, 0, 0, 0] [32, %[[NEXT_SEQ_STEP_DIV16]], 64, 16, 2] [1, 1, 1, 1, 1] : tensor<32x256x64x16x2xf32> to tensor<32x?x64x16x2xf32>
//       CHECK:     %[[PACKED_N:.+]] = affine.apply #[[MAP3]]()[%[[SEQ_STEP]]]
//       CHECK:     %[[EMPTY:.+]] = tensor.empty(%[[PACKED_N]]) : tensor<32x1x?x1x16xf32>
//       CHECK:     %[[MMT4D_DST:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
//       CHECK:     %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[ARG0]], %[[PACKED_RHS]] : tensor<32x1x64x1x2xf32>, tensor<32x?x64x16x2xf32>) outs(%[[MMT4D_DST]] : tensor<32x1x?x1x16xf32>) -> tensor<32x1x?x1x16xf32>
//       CHECK:     %[[NEXT_SEQ_STEP:.+]] = arith.addi %[[SEQ_STEP]], %[[ONE]] : index
//       CHECK:     util.global.store %[[NEXT_SEQ_STEP]], @_global_seq_step.global : index
//       CHECK:     util.return %[[MMT4D]] : tensor<32x1x?x1x16xf32>
