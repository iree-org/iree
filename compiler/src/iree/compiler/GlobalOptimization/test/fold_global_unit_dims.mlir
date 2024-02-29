// RUN: iree-opt --iree-global-opt-fold-global-unit-dims --split-input-file %s | FileCheck %s

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
    %expanded = tensor.expand_shape %1 [[0, 1], [2, 3, 4]] : tensor<32x64xf32> into tensor<1x32x1x1x64xf32>
    util.global.store %expanded, @global : tensor<1x32x1x1x64xf32>
    util.return %1 : tensor<32x64xf32>
  }
}

//      CHECK: module @fold_unit_dims
//      CHECK:   util.global private mutable @[[GLOBAL:.+]] {inlining_policy = #util.inline.never} = #util.uninitialized : tensor<32x64xf32>
//      CHECK:   util.global private mutable @[[UNIT_GLOBAL:.+]] = #util.uninitialized : tensor<1x1xf32>
//      CHECK:   util.func public @fold_global_unit_dims
//      CHECK:     %[[LOAD0:.+]] = util.global.load @[[GLOBAL]] : tensor<32x64xf32>
//      CHECK:     %[[LOAD1:.+]] = util.global.load @[[UNIT_GLOBAL]] : tensor<1x1xf32>
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
  util.global public @global : tensor<1x32x1x1x64xf32>
  util.func public @no_fold_global_unit_dims() -> tensor<32x64xf32> {
    %global = util.global.load @global : tensor<1x32x1x1x64xf32>
    %collapsed = tensor.collapse_shape %global [[0, 1], [2, 3, 4]] : tensor<1x32x1x1x64xf32> into tensor<32x64xf32>
    util.return %collapsed : tensor<32x64xf32>
  }
}

//      CHECK: module @no_fold_public
//      CHECK:   util.global public @[[GLOBAL:.+]] : tensor<1x32x1x1x64xf32>
//      CHECK:   util.func public @no_fold_global_unit_dims
//      CHECK:     %[[LOAD:.+]] = util.global.load @[[GLOBAL]] : tensor<1x32x1x1x64xf32>
//      CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[LOAD]]
//      CHECK:     util.return %[[COLLAPSE]]
