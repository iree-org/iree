// RUN: iree-opt --pass-pipeline="builtin.module(iree-dispatch-creation-fold-unit-extent-dims)" %s --split-input-file | FileCheck %s

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
// CHECK-LABEL: util.func public @no_fold_unit_dims_in_dispatches
//  CHECK-SAME:   (%[[ARG0:.+]]: tensor<1x1x10xf32>)
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0]] : tensor<1x1x10xf32>)
//       CHECK:     flow.return %[[GENERIC]]
//       CHECK:   util.return %[[DISPATCH]]


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

// CHECK-LABEL: module @fold_unit_dims
//       CHECK:   util.global private mutable @[[GLOBAL:.+]] {inlining_policy = #util.inline.never} = #util.uninitialized : tensor<32x64xf32>
//       CHECK:   util.global private mutable @[[UNIT_GLOBAL:.+]] = #util.uninitialized : tensor<f32>
//       CHECK:   util.func public @fold_global_unit_dims
//       CHECK:     %[[LOAD0:.+]] = util.global.load @[[GLOBAL]] : tensor<32x64xf32>
//       CHECK:     %[[LOAD1:.+]] = util.global.load @[[UNIT_GLOBAL]] : tensor<f32>
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[LOAD0]], %[[LOAD1]]
//       CHECK:     util.global.store %[[GENERIC]], @[[GLOBAL]] : tensor<32x64xf32>
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

// CHECK-LABEL: module @no_fold_immutable
//       CHECK:   util.global private @[[GLOBAL:.+]] : tensor<1x32x1x1x64xf32>
//       CHECK:   util.func public @no_fold_global_unit_dims
//       CHECK:     %[[LOAD:.+]] = util.global.load @[[GLOBAL]] : tensor<1x32x1x1x64xf32>
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[LOAD]]
//       CHECK:     util.return %[[COLLAPSE]]

// -----

module @no_fold_public {
  util.global public mutable @global : tensor<1x32x1x1x64xf32>
  util.func public @no_fold_global_unit_dims() -> tensor<32x64xf32> {
    %global = util.global.load @global : tensor<1x32x1x1x64xf32>
    %collapsed = tensor.collapse_shape %global [[0, 1], [2, 3, 4]] : tensor<1x32x1x1x64xf32> into tensor<32x64xf32>
    util.return %collapsed : tensor<32x64xf32>
  }
}

// CHECK-LABEL: module @no_fold_public
//       CHECK:   util.global public mutable @[[GLOBAL:.+]] : tensor<1x32x1x1x64xf32>
//       CHECK:   util.func public @no_fold_global_unit_dims
//       CHECK:     %[[LOAD:.+]] = util.global.load @[[GLOBAL]] : tensor<1x32x1x1x64xf32>
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[LOAD]]

// -----

module @fold_stream_parameter {
  util.global private mutable @global = #stream.parameter.named<"module"::"global"> : tensor<1x1x10xf32>
  util.func public @fold_stream_parameter() -> tensor<1x1x10xf32> {
    %global = util.global.load @global : tensor<1x1x10xf32>
    util.return %global : tensor<1x1x10xf32>
  }
}

// CHECK-LABEL: module @fold_stream_parameter
//       CHECK:   util.global private mutable @[[GLOBAL:.+]] = #stream.parameter.named<"module"::"global"> : tensor<10xf32>
//       CHECK:   util.func public @fold_stream_parameter
//      CHECK:     %[[LOAD:.+]] = util.global.load @[[GLOBAL]] : tensor<10xf32>

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

util.func @collapse_of_expand_2(%arg0: tensor<?x1xf16>, %arg1: index) -> tensor<4x?x1xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3, 4]] output_shape [4, %arg1, 1, 1, 1] : tensor<?x1xf16> into tensor<4x?x1x1x1xf16>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3], [4]] : tensor<4x?x1x1x1xf16> into tensor<4x?x1xf16>
  util.return %collapsed : tensor<4x?x1xf16>
}

// CHECK-LABEL: util.func public @collapse_of_expand_2
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x1xf16>, %[[ARG1:.+]]: index
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<?x1xf16> into tensor<4x?x1xf16>
//       CHECK:   util.return %[[EXPAND]] : tensor<4x?x1xf16>

// -----

util.func @collapse_of_expand_3(%arg0: tensor<?x?xf16>, %arg1: index, %arg2: index) -> tensor<?x?xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [%arg1, 1, 1, %arg2] : tensor<?x?xf16> into tensor<?x1x1x?xf16>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3]] : tensor<?x1x1x?xf16> into tensor<?x?xf16>
  util.return %collapsed : tensor<?x?xf16>
}

// CHECK-LABEL: util.func public @collapse_of_expand_3
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf16>
//       CHECK:   util.return %[[ARG0]] : tensor<?x?xf16>

// -----

util.func @collapse_of_expand_4(%arg0: tensor<1x1xf16>, %arg1: index, %arg2: index) -> tensor<1xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [%arg1, 1, 1, %arg2] : tensor<1x1xf16> into tensor<1x1x1x1xf16>
  %collapsed = tensor.collapse_shape %expanded [[0, 1, 2, 3]] : tensor<1x1x1x1xf16> into tensor<1xf16>
  util.return %collapsed : tensor<1xf16>
}

// CHECK-LABEL: util.func public @collapse_of_expand_4
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1xf16>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]]
//  CHECK-SAME:     tensor<1x1xf16> into tensor<1xf16>
//       CHECK:   util.return %[[COLLAPSED]] : tensor<1xf16>
