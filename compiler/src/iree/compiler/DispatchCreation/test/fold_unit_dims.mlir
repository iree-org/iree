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

module @fold_flow_parameter {
  util.global private mutable @global = #flow.parameter.named<"module"::"global"> : tensor<1x1x10xf32>
  util.func public @fold_flow_parameter() -> tensor<1x1x10xf32> {
    %global = util.global.load @global : tensor<1x1x10xf32>
    util.return %global : tensor<1x1x10xf32>
  }
}

//      CHECK: module @fold_flow_parameter
//      CHECK:   util.global private mutable @[[GLOBAL:.+]] = #flow.parameter.named<"module"::"global"> : tensor<10xf32>
//      CHECK:   util.func public @fold_flow_parameter
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
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5, d0, d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5, d0, d2)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
//  CHECK-SAME:     ins({{.+}}, %[[COLLAPSED]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ATTN]]
//  CHECK-SAME:     tensor<8x4x128xf32> into tensor<8x4x1x128xf32>
//       CHECK:   util.return %[[EXPAND]]


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
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> ()>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//  CHECK-SAME:     ins({{.+}}, %[[COLLAPSED]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ATTN]]
//  CHECK-SAME:     tensor<32x128xf16> into tensor<32x1x128xf16>
//       CHECK:   util.return %[[EXPAND]]

// -----

util.func @collapse_of_expand_0(%arg0: tensor<?x128xf16>, %arg1: index) -> tensor<4x?x128xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3, 4]] output_shape [4, %arg1, 1, 1, 128] : tensor<?x128xf16> into tensor<4x?x1x1x128xf16>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3], [4]] : tensor<4x?x1x1x128xf16> into tensor<4x?x128xf16>
  util.return %collapsed : tensor<4x?x128xf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_0
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x128xf16>, %[[ARG1:.+]]: index
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<?x128xf16> into tensor<4x?x128xf16>
//       CHECK:   util.return %[[EXPAND]] : tensor<4x?x128xf16>

// -----

util.func @collapse_of_expand_1(%arg0: tensor<?x128xf16>, %arg1: index) -> tensor<4x?x64xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3, 4]] output_shape [4, %arg1, 1, 2, 64] : tensor<?x128xf16> into tensor<4x?x1x2x64xf16>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3], [4]] : tensor<4x?x1x2x64xf16> into tensor<4x?x64xf16>
  util.return %collapsed : tensor<4x?x64xf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_1
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x128xf16>, %[[ARG1:.+]]: index
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<?x128xf16> into tensor<4x?x2x64xf16>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]]
//  CHECK-SAME:     tensor<4x?x2x64xf16> into tensor<4x?x64xf16>
//       CHECK:   util.return %[[COLLAPSE]] : tensor<4x?x64xf16>

// -----

util.func @collapse_of_expand_to_expand(%arg0: tensor<?x1xf16>, %arg1: index) -> tensor<4x?x1xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3, 4]] output_shape [4, %arg1, 1, 1, 1] : tensor<?x1xf16> into tensor<4x?x1x1x1xf16>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3], [4]] : tensor<4x?x1x1x1xf16> into tensor<4x?x1xf16>
  util.return %collapsed : tensor<4x?x1xf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_to_expand
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x1xf16>, %[[ARG1:.+]]: index
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<?x1xf16> into tensor<4x?x1xf16>
//       CHECK:   util.return %[[EXPAND]] : tensor<4x?x1xf16>

// -----

util.func @collapse_of_expand_fully_dynamic(%arg0: tensor<?x?xf16>, %arg1: index, %arg2: index) -> tensor<?x?xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [%arg1, 1, 1, %arg2] : tensor<?x?xf16> into tensor<?x1x1x?xf16>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3]] : tensor<?x1x1x?xf16> into tensor<?x?xf16>
  util.return %collapsed : tensor<?x?xf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_fully_dynamic
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf16>
//       CHECK:   util.return %[[ARG0]] : tensor<?x?xf16>

// -----

util.func @collapse_of_expand_all_unit_dim_groups(%arg0: tensor<1x1xf16>, %arg1: index, %arg2: index) -> tensor<1xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [%arg1, 1, 1, %arg2] : tensor<1x1xf16> into tensor<1x1x1x1xf16>
  %collapsed = tensor.collapse_shape %expanded [[0, 1, 2, 3]] : tensor<1x1x1x1xf16> into tensor<1xf16>
  util.return %collapsed : tensor<1xf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_all_unit_dim_groups
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1xf16>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]]
//  CHECK-SAME:     tensor<1x1xf16> into tensor<1xf16>
//       CHECK:   util.return %[[COLLAPSED]] : tensor<1xf16>

// -----

util.func @collapse_of_expand_to_collapse(%arg0: tensor<1x?x4x32xf16>, %arg1: index) -> tensor<?x4x32xf16> {
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [1, %arg1, 4, 1, 32] : tensor<1x?x4x32xf16> into tensor<1x?x4x1x32xf16>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2, 3], [4]] : tensor<1x?x4x1x32xf16> into tensor<?x4x32xf16>
  util.return %collapsed : tensor<?x4x32xf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_to_collapse
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x4x32xf16>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]]
//  CHECK-SAME:     tensor<1x?x4x32xf16> into tensor<?x4x32xf16>
//       CHECK:   util.return %[[COLLAPSED]] : tensor<?x4x32xf16>

// -----

util.func @collapse_of_expand_to_scalar(%arg0: tensor<1x1xf16>, %arg1: index, %arg2: index) -> tensor<f16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [%arg1, 1, 1, %arg2] : tensor<1x1xf16> into tensor<1x1x1x1xf16>
  %collapsed = tensor.collapse_shape %expanded [] : tensor<1x1x1x1xf16> into tensor<f16>
  util.return %collapsed : tensor<f16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_to_scalar
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1xf16>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]]
//  CHECK-SAME:     tensor<1x1xf16> into tensor<f16>
//       CHECK:   util.return %[[COLLAPSED]] : tensor<f16>

// -----

util.func @collapse_of_expand_trailing_unit_dims(%arg0: tensor<23040x1xbf16>) -> tensor<4x5760xbf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [4, 5760, 1, 1] : tensor<23040x1xbf16> into tensor<4x5760x1x1xbf16>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3]] : tensor<4x5760x1x1xbf16> into tensor<4x5760xbf16>
  util.return %collapsed : tensor<4x5760xbf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_trailing_unit_dims
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<23040x1xbf16>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<23040x1xbf16> into tensor<4x5760x1xbf16>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]]
//  CHECK-SAME:     tensor<4x5760x1xbf16> into tensor<4x5760xbf16>
//       CHECK:   util.return %[[COLLAPSE]] : tensor<4x5760xbf16>

// -----

// This test considers the case where we have multiple trailing unit dims but must preserve one for the output,
// as well as an isolated unit dim that must be preserved for the collapse's reassociation dims.
util.func @collapse_of_expand_preserved_trailing_unit_dims(%arg0: tensor<1x23040xbf16>) -> tensor<4x5760x1xbf16> {
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4, 5]] output_shape [1, 4, 5760, 1, 1, 1] : tensor<1x23040xbf16> into tensor<1x4x5760x1x1x1xbf16>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2], [3, 4, 5]] : tensor<1x4x5760x1x1x1xbf16> into tensor<4x5760x1xbf16>
  util.return %collapsed : tensor<4x5760x1xbf16>
}
// CHECK-LABEL: util.func public @collapse_of_expand_preserved_trailing_unit_dims
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1x23040xbf16>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<1x23040xbf16> into tensor<1x4x5760x1xbf16>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]]
//  CHECK-SAME:     tensor<1x4x5760x1xbf16> into tensor<4x5760x1xbf16>
//       CHECK:   util.return %[[COLLAPSE]] : tensor<4x5760x1xbf16>

// -----

util.func @fold_unit_dims_from_extract_leading(%arg0: tensor<1x4x8xf32>, %idx0: index, %idx1: index, %idx2: index) -> f32 {
  %extracted = tensor.extract %arg0[%idx0, %idx1, %idx2] : tensor<1x4x8xf32>
  util.return %extracted : f32
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_leading
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x4x8xf32>
//  CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX2:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}}
//  CHECK-SAME:     tensor<1x4x8xf32> into tensor<4x8xf32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[COLLAPSED]][%[[IDX1]], %[[IDX2]]]
//       CHECK:   util.return %[[EXTRACT]] : f32

// -----

util.func @fold_unit_dims_from_extract_trailing(%arg0: tensor<4x8x1xf32>, %idx0: index, %idx1: index, %idx2: index) -> f32 {
  %extracted = tensor.extract %arg0[%idx0, %idx1, %idx2] : tensor<4x8x1xf32>
  util.return %extracted : f32
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_trailing
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x8x1xf32>
//  CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX2:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2]{{\]}}
//  CHECK-SAME:     tensor<4x8x1xf32> into tensor<4x8xf32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[COLLAPSED]][%[[IDX0]], %[[IDX1]]]
//       CHECK:   util.return %[[EXTRACT]] : f32

// -----

util.func @fold_unit_dims_from_extract_middle(%arg0: tensor<4x1x8xf32>, %idx0: index, %idx1: index, %idx2: index) -> f32 {
  %extracted = tensor.extract %arg0[%idx0, %idx1, %idx2] : tensor<4x1x8xf32>
  util.return %extracted : f32
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_middle
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x1x8xf32>
//  CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX2:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2]{{\]}}
//  CHECK-SAME:     tensor<4x1x8xf32> into tensor<4x8xf32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[COLLAPSED]][%[[IDX0]], %[[IDX2]]]
//       CHECK:   util.return %[[EXTRACT]] : f32

// -----

util.func @fold_unit_dims_from_extract_multiple(%arg0: tensor<1x4x1x8x1xf32>, %idx0: index, %idx1: index, %idx2: index, %idx3: index, %idx4: index) -> f32 {
  %extracted = tensor.extract %arg0[%idx0, %idx1, %idx2, %idx3, %idx4] : tensor<1x4x1x8x1xf32>
  util.return %extracted : f32
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_multiple
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x4x1x8x1xf32>
//  CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX4:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2, 3, 4]{{\]}}
//  CHECK-SAME:     tensor<1x4x1x8x1xf32> into tensor<4x8xf32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[COLLAPSED]]
//       CHECK:   util.return %[[EXTRACT]] : f32

// -----

// Test folding consecutive unit dims from tensor.extract
util.func @fold_unit_dims_from_extract_consecutive(%arg0: tensor<1x1x1x8xf32>, %idx0: index, %idx1: index, %idx2: index, %idx3: index) -> f32 {
  %extracted = tensor.extract %arg0[%idx0, %idx1, %idx2, %idx3] : tensor<1x1x1x8xf32>
  util.return %extracted : f32
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_consecutive
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x1x8xf32>
//  CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX3:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2, 3]{{\]}}
//  CHECK-SAME:     tensor<1x1x1x8xf32> into tensor<8xf32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[COLLAPSED]][%[[IDX3]]]
//       CHECK:   util.return %[[EXTRACT]] : f32

// -----

// Test folding unit dims with dynamic dimensions
util.func @fold_unit_dims_from_extract_dynamic(%arg0: tensor<1x?x1xf32>, %idx0: index, %idx1: index, %idx2: index) -> f32 {
  %extracted = tensor.extract %arg0[%idx0, %idx1, %idx2] : tensor<1x?x1xf32>
  util.return %extracted : f32
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_dynamic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?x1xf32>
//  CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]{{\]}}
//  CHECK-SAME:     tensor<1x?x1xf32> into tensor<?xf32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[COLLAPSED]][%[[IDX1]]]
//       CHECK:   util.return %[[EXTRACT]] : f32

// -----

util.func @fold_unit_dims_from_extract_all_unit(%arg0: tensor<1x1x1xf32>, %idx0: index, %idx1: index, %idx2: index) -> f32 {
  %extracted = tensor.extract %arg0[%idx0, %idx1, %idx2] : tensor<1x1x1xf32>
  util.return %extracted : f32
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_all_unit
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x1xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] []
//  CHECK-SAME:     tensor<1x1x1xf32> into tensor<f32>
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[COLLAPSED]]
//  CHECK-SAME:     tensor<f32>
//       CHECK:   util.return %[[EXTRACT]] : f32

// -----

// Test folding unit dims from tensor.extract_slice - basic case
util.func @fold_unit_dims_from_extract_slice_basic(%arg0: tensor<10x1x20x1x30xf32>, %idx0: index, %idx1: index, %idx2: index, %sz0: index, %sz1: index, %sz2: index) -> tensor<?x1x?x1x?xf32> {
  %slice = tensor.extract_slice %arg0[%idx0, 0, %idx1, 0, %idx2][%sz0, 1, %sz1, 1, %sz2][1, 1, 1, 1, 1]
    : tensor<10x1x20x1x30xf32> to tensor<?x1x?x1x?xf32>
  util.return %slice : tensor<?x1x?x1x?xf32>
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_slice_basic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<10x1x20x1x30xf32>
//  CHECK-SAME:   %[[IDX0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[IDX2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[SZ0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[SZ1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[SZ2:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2], [3, 4]{{\]}}
//  CHECK-SAME:     tensor<10x1x20x1x30xf32> into tensor<10x20x30xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSED]][%[[IDX0]], %[[IDX1]], %[[IDX2]]]
//  CHECK-SAME:     [%[[SZ0]], %[[SZ1]], %[[SZ2]]] [1, 1, 1]
//  CHECK-SAME:     tensor<10x20x30xf32> to tensor<?x?x?xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SLICE]] {{\[}}[0], [1, 2], [3, 4]{{\]}}
//       CHECK:   util.return %[[EXPANDED]]

// -----

// Test folding unit dims from tensor.extract_slice - rank reducing case
util.func @fold_unit_dims_from_extract_slice_rank_reducing(%arg0: tensor<10x1x20x1x30xf32>) -> tensor<5x10xf32> {
  %slice = tensor.extract_slice %arg0[0, 0, 5, 0, 10][5, 1, 1, 1, 10][1, 1, 1, 1, 1]
    : tensor<10x1x20x1x30xf32> to tensor<5x10xf32>
  util.return %slice : tensor<5x10xf32>
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_slice_rank_reducing
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<10x1x20x1x30xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2], [3, 4]{{\]}}
//  CHECK-SAME:     tensor<10x1x20x1x30xf32> into tensor<10x20x30xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSED]][0, 5, 10] [5, 1, 10] [1, 1, 1]
//  CHECK-SAME:     tensor<10x20x30xf32> to tensor<5x10xf32>
//       CHECK:   util.return %[[SLICE]]

// -----

// Test folding unit dims from tensor.extract_slice - leading unit dims
util.func @fold_unit_dims_from_extract_slice_leading(%arg0: tensor<1x1x20x30xf32>, %idx: index, %sz: index) -> tensor<1x1x?x30xf32> {
  %slice = tensor.extract_slice %arg0[0, 0, %idx, 0][1, 1, %sz, 30][1, 1, 1, 1]
    : tensor<1x1x20x30xf32> to tensor<1x1x?x30xf32>
  util.return %slice : tensor<1x1x?x30xf32>
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_slice_leading
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x20x30xf32>
//  CHECK-SAME:   %[[IDX:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[SZ:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]{{\]}}
//  CHECK-SAME:     tensor<1x1x20x30xf32> into tensor<20x30xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSED]][%[[IDX]], 0] [%[[SZ]], 30] [1, 1]
//  CHECK-SAME:     tensor<20x30xf32> to tensor<?x30xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SLICE]] {{\[}}[0, 1, 2], [3]{{\]}}
//       CHECK:   util.return %[[EXPANDED]]

// -----

// Test folding unit dims from tensor.extract_slice - trailing unit dims
util.func @fold_unit_dims_from_extract_slice_trailing(%arg0: tensor<20x30x1x1xf32>, %idx: index, %sz: index) -> tensor<?x30x1x1xf32> {
  %slice = tensor.extract_slice %arg0[%idx, 0, 0, 0][%sz, 30, 1, 1][1, 1, 1, 1]
    : tensor<20x30x1x1xf32> to tensor<?x30x1x1xf32>
  util.return %slice : tensor<?x30x1x1xf32>
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_slice_trailing
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x30x1x1xf32>
//  CHECK-SAME:   %[[IDX:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[SZ:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2, 3]{{\]}}
//  CHECK-SAME:     tensor<20x30x1x1xf32> into tensor<20x30xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSED]][%[[IDX]], 0] [%[[SZ]], 30] [1, 1]
//  CHECK-SAME:     tensor<20x30xf32> to tensor<?x30xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SLICE]] {{\[}}[0], [1, 2, 3]{{\]}}
//       CHECK:   util.return %[[EXPANDED]]

// -----

// Test folding unit dims from tensor.extract_slice - static offsets and sizes
util.func @fold_unit_dims_from_extract_slice_static(%arg0: tensor<4x1x32x1x64xf32>) -> tensor<4x1x16x1x64xf32> {
  %slice = tensor.extract_slice %arg0[0, 0, 8, 0, 0][4, 1, 16, 1, 64][1, 1, 1, 1, 1]
    : tensor<4x1x32x1x64xf32> to tensor<4x1x16x1x64xf32>
  util.return %slice : tensor<4x1x16x1x64xf32>
}
// CHECK-LABEL: util.func public @fold_unit_dims_from_extract_slice_static
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x1x32x1x64xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2], [3, 4]{{\]}}
//  CHECK-SAME:     tensor<4x1x32x1x64xf32> into tensor<4x32x64xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSED]][0, 8, 0] [4, 16, 64] [1, 1, 1]
//  CHECK-SAME:     tensor<4x32x64xf32> to tensor<4x16x64xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SLICE]] {{\[}}[0], [1, 2], [3, 4]{{\]}}
//       CHECK:   util.return %[[EXPANDED]]
