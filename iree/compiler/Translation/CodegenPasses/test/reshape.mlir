// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK: func @reshape_collapse_single_dim
// CHECK-SAME: %[[IN:.+]]: memref<1x28x28x1xf32>
// CHECK-SAME: %[[OUT:.+]]: memref<1x784xf32>
func @reshape_collapse_single_dim(%arg0: memref<1x28x28x1xf32>,
                                  %arg1: memref<1x784xf32>)
attributes {iree.dispatch_fn_name = ""} {
  %0 = iree.load_input(%arg0 : memref<1x28x28x1xf32>) : tensor<1x28x28x1xf32>
  %1 = "xla_hlo.reshape"(%0) : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
  iree.store_output(%1 : tensor<1x784xf32>, %arg1 : memref<1x784xf32>)
  return
}
// CHECK: %[[RESULT:.+]] = linalg.reshape %[[IN]] [
// CHECK-SAME: #[[MAP0]], #[[MAP1]]
// CHECK-SAME: ] : memref<{{.+}}xf32> into memref<{{.+}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: func @reshape_collapse
// CHECK-SAME: %[[IN:.+]]: memref<2x2x2x3xf32>
// CHECK-SAME: %[[OUT:.+]]: memref<2x4x3xf32>
func @reshape_collapse(%arg0: memref<2x2x2x3xf32>, %arg1 : memref<2x4x3xf32>)
attributes {iree.dispatch_fn_name = ""} {
    %0 = iree.load_input(%arg0 : memref<2x2x2x3xf32>) : tensor<2x2x2x3xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<2x2x2x3xf32>) -> tensor<2x4x3xf32>
    iree.store_output(%1 : tensor<2x4x3xf32>, %arg1 : memref<2x4x3xf32>)
    return
}
// CHECK: %[[RESULT:.+]] = linalg.reshape %[[IN]] [
// CHECK-SAME: #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME: ] : memref<{{.+}}xf32> into memref<{{.+}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: func @reshape_expand
// CHECK-SAME: %[[IN:.+]]: memref<2x8xf32>
// CHECK-SAME: %[[OUT:.+]]: memref<2x4x2xf32>
func @reshape_expand(%arg0: memref<2x8xf32>, %arg1: memref<2x4x2xf32>)
attributes {iree.dispatch_fn_name = ""} {
    %0 = iree.load_input(%arg0: memref<2x8xf32>) : tensor<2x8xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<2x8xf32>) -> tensor<2x4x2xf32>
    iree.store_output(%1 : tensor<2x4x2xf32>, %arg1 : memref<2x4x2xf32>)
    return
}
// CHECK: %[[RESULT:.+]] = linalg.reshape %[[IN]] [
// CHECK-SAME: #[[MAP0]], #[[MAP1]]
// CHECK-SAME: ] : memref<{{.+}}xf32> into memref<{{.+}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @reshape_single_exapnd
// CHECK-SAME: %[[IN:.+]]: memref<8xf32>
// CHECK-SAME: %[[OUT:.+]]: memref<1x4x2xf32>
func @reshape_single_exapnd(%arg0 : memref<8xf32>, %arg1 : memref<1x4x2xf32>)
attributes {iree.dispatch_fn_name = ""} {
    %0 = iree.load_input(%arg0 : memref<8xf32>) : tensor<8xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<8xf32>) -> tensor<1x4x2xf32>
    iree.store_output(%1 : tensor<1x4x2xf32>, %arg1 : memref<1x4x2xf32>)
    return
}
// CHECK: %[[RESULT:.+]] = linalg.reshape %[[IN]] [
// CHECK-SAME: #[[MAP0]]
// CHECK-SAME: ] : memref<{{.+}}xf32> into memref<{{.+}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])
