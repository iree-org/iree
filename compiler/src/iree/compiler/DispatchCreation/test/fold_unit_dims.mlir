// RUN: iree-opt --pass-pipeline="builtin.module(iree-dispatch-creation-fold-unit-extent-dims)" %s --split-input-file --mlir-print-local-scope | FileCheck %s

util.func public @no_fold_unit_dims_in_dispatches(%arg0 : tensor<1x1x10xf32>) -> tensor<1x1x10xf32> {
  %0 = tensor.empty() : tensor<1x1x10xf32>
  %1 = flow.dispatch.region[] -> (tensor<1x1x10xf32>) {
    %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x1x10xf32>) outs(%0 : tensor<1x1x10xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %3 = arith.addf %b0, %b0 : f32
      linalg.yield %3 : f32
    } -> tensor<1x1x10xf32>
    flow.return %2 : tensor<1x1x10xf32>
  }
  util.return %1 : tensor<1x1x10xf32>
}
//      CHECK: util.func public @no_fold_unit_dims_in_dispatches(%[[ARG0:.+]]: tensor<1x1x10xf32>)
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[ARG0]] : tensor<1x1x10xf32>)
//      CHECK:     flow.return %[[GENERIC]]
//      CHECK:   util.return %[[DISPATCH]]


// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
module @fold_unit_dims {
  util.global private mutable @global {inlining_policy = #util.inline.never} = #util.uninitialized : tensor<1x32x1x1x64xf32>
  util.global private mutable @unit_global = #util.uninitialized : tensor<1x1xf32>
  util.func public @fold_global_unit_dims() -> tensor<32x64xf32> {
    %global = util.global.load @global : tensor<1x32x1x1x64xf32>
    %unit_global = util.global.load @unit_global : tensor<1x1xf32>
    %collapsed = tensor.collapse_shape %global [[0, 1], [2, 3, 4]] : tensor<1x32x1x1x64xf32> into tensor<32x64xf32>
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %unit_global : tensor<32x64xf32>, tensor<1x1xf32>) outs(%0 : tensor<32x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<32x64xf32>
    %expanded = tensor.expand_shape %1 [[0, 1], [2, 3, 4]] output_shape[1, 32, 1, 1, 64] : tensor<32x64xf32> into tensor<1x32x1x1x64xf32>
    util.global.store %expanded, @global : tensor<1x32x1x1x64xf32>
    util.return %1 : tensor<32x64xf32>
  }
}

//      CHECK: module @fold_unit_dims
//      CHECK:   util.global private mutable @[[GLOBAL:.+]] {inlining_policy = #util.inline.never} = #util.uninitialized : tensor<32x64xf32>
//      CHECK:   util.global private mutable @[[UNIT_GLOBAL:.+]] = #util.uninitialized : tensor<f32>
//      CHECK:   util.func public @fold_global_unit_dims
//      CHECK:     %[[LOAD0:.+]] = util.global.load @[[GLOBAL]] : tensor<32x64xf32>
//      CHECK:     %[[LOAD1:.+]] = util.global.load @[[UNIT_GLOBAL]] : tensor<f32>
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:     ins(%[[LOAD0]], %[[LOAD1]]
//      CHECK:     util.global.store %[[GENERIC]], @[[GLOBAL]] : tensor<32x64xf32>
//      CHECK:     util.return %[[GENERIC]]

// -----

module @no_fold_immutable {
  util.global private @global : tensor<1x32x1x1x64xf32>
  util.func public @no_fold_global_unit_dims() -> tensor<32x64xf32> {
    %global = util.global.load @global : tensor<1x32x1x1x64xf32>
    %collapsed = tensor.collapse_shape %global [[0, 1], [2, 3, 4]] : tensor<1x32x1x1x64xf32> into tensor<32x64xf32>
    util.return %collapsed : tensor<32x64xf32>
  }
}

//      CHECK: module @no_fold_immutable
//      CHECK:   util.global private @[[GLOBAL:.+]] : tensor<1x32x1x1x64xf32>
//      CHECK:   util.func public @no_fold_global_unit_dims
//      CHECK:     %[[LOAD:.+]] = util.global.load @[[GLOBAL]] : tensor<1x32x1x1x64xf32>
//      CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[LOAD]]
//      CHECK:     util.return %[[COLLAPSE]]

// -----

module @no_fold_public {
  util.global public mutable @global : tensor<1x32x1x1x64xf32>
  util.func public @no_fold_global_unit_dims() -> tensor<32x64xf32> {
    %global = util.global.load @global : tensor<1x32x1x1x64xf32>
    %collapsed = tensor.collapse_shape %global [[0, 1], [2, 3, 4]] : tensor<1x32x1x1x64xf32> into tensor<32x64xf32>
    util.return %collapsed : tensor<32x64xf32>
  }
}

//      CHECK: module @no_fold_public
//      CHECK:   util.global public mutable @[[GLOBAL:.+]] : tensor<1x32x1x1x64xf32>
//      CHECK:   util.func public @no_fold_global_unit_dims
//      CHECK:     %[[LOAD:.+]] = util.global.load @[[GLOBAL]] : tensor<1x32x1x1x64xf32>
//      CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[LOAD]]

// -----

module @fold_stream_parameter {
  util.global private mutable @global = #stream.parameter.named<"module"::"global"> : tensor<1x1x10xf32>
  util.func public @fold_stream_parameter() -> tensor<1x1x10xf32> {
    %global = util.global.load @global : tensor<1x1x10xf32>
    util.return %global : tensor<1x1x10xf32>
  }
}

//      CHECK: module @fold_stream_parameter
//      CHECK:   util.global private mutable @[[GLOBAL:.+]] = #stream.parameter.named<"module"::"global"> : tensor<10xf32>
//      CHECK:   util.func public @fold_stream_parameter
//      CHECK:     %[[LOAD:.+]] = util.global.load @[[GLOBAL]] : tensor<10xf32>

// -----

util.func public @scatter(%arg0 : tensor<4xi64>, %arg1 : tensor<4x1xi32>, %arg2 : tensor<4xi64>) -> tensor<4xi64> {
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%arg0, %arg1: tensor<4xi64>, tensor<4x1xi32>) outs(%arg2 : tensor<4xi64>) {
  ^bb0(%arg3: i64, %arg4: i64):
    %16 = arith.addi %arg4, %arg3 : i64
    iree_linalg_ext.yield %16 : i64
  } -> tensor<4xi64>
  util.return %0 : tensor<4xi64>
}
// CHECK-LABEL: func public @scatter
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]]
//  CHECK-SAME:     tensor<4x1xi32> into tensor<4xi32>
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     ins(%[[ARG0]], %[[COLLAPSED]]
//  CHECK-SAME:     outs(%[[ARG2]]
//       CHECK:   util.return %[[SCATTER]]

// -----

util.func public @attention_mask_multi_m_dims(%arg0: tensor<8x4x1x128xf32>, %arg1: tensor<?x32x8x128xf32>, %arg2: tensor<?x32x8x128xf32>, %arg3: f32, %arg4: tensor<8x4x1x?x32xf32>) -> tensor<8x4x1x128xf32> {
  %0 = tensor.empty() : tensor<8x4x1x128xf32>
  %1 = iree_linalg_ext.attention {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d0, d4)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d0, d3)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d6)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>]}
    ins(%arg0, %arg1, %arg2, %arg3, %arg4 : tensor<8x4x1x128xf32>, tensor<?x32x8x128xf32>, tensor<?x32x8x128xf32>, f32, tensor<8x4x1x?x32xf32>)
    outs(%0 : tensor<8x4x1x128xf32>) {
  ^bb0(%arg5: f32):
    iree_linalg_ext.yield %arg5 : f32
  } -> tensor<8x4x1x128xf32>
  util.return %1 : tensor<8x4x1x128xf32>
}
// CHECK-LABEL: func public @attention_mask_multi_m_dims
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG4]]
//  CHECK-SAME:     tensor<8x4x1x?x32xf32> into tensor<8x4x?x32xf32>
//       CHECK:   %[[ATTN:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d0, d4)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d0, d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d5, d6)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     ins({{.+}}, %[[COLLAPSED]]
//       CHECK:   util.return %[[ATTN]]


// -----

util.func public @attention_mask_single_m_dim(%arg0 : tensor<32x1x128xf16>, %arg1 : tensor<32x?x128xf16>, %arg2 : tensor<32x128x?xf16>, %arg3 : f16, %arg4 : tensor<32x1x?xf16>) -> tensor<32x1x128xf16> {
  %0 = tensor.empty() : tensor<32x1x128xf16>
  %1 = iree_linalg_ext.attention {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> ()>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]}
    ins(%arg0, %arg1, %arg2, %arg3, %arg4 : tensor<32x1x128xf16>, tensor<32x?x128xf16>, tensor<32x128x?xf16>, f16, tensor<32x1x?xf16>)
    outs(%0 : tensor<32x1x128xf16>) {
  ^bb0(%arg7: f32):
    iree_linalg_ext.yield %arg7 : f32
  } -> tensor<32x1x128xf16>
  util.return %1 : tensor<32x1x128xf16>
}
// CHECK-LABEL: func public @attention_mask_single_m_dim
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG4]]
//  CHECK-SAME:     tensor<32x1x?xf16> into tensor<32x?xf16>
//       CHECK:   %[[ATTN:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> ()>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d4)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//  CHECK-SAME:     ins({{.+}}, %[[COLLAPSED]]
//       CHECK:   util.return %[[ATTN]]
