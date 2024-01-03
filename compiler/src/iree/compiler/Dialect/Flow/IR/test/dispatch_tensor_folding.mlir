// RUN: iree-opt --allow-unregistered-dialect --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ReuseDispatchTensorLoadShapeDims
func.func @ReuseDispatchTensorLoadShapeDims(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %arg0_tied = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg1, %arg2}
  %c0 = arith.constant 0 : index
  // CHECK: flow.dispatch.tensor.load {{.+}} !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg1, %arg2}
  %0 = flow.dispatch.tensor.load %arg0_tied, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg3, %arg4} -> tensor<256x1024xf32>
  "test.sink"(%0) : (tensor<256x1024xf32>) -> ()
  return
}

// -----

func.func @canonicalizeStaticOperands(%arg0: !flow.dispatch.tensor<readonly:tensor<4x4xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = flow.dispatch.tensor.load %arg0, offsets=[%c0, %c0], sizes=[%c2, %c2], strides=[%c1, %c1] : !flow.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<?x?xf32>
  "test.sink"(%0) : (tensor<?x?xf32>) -> ()
  return
}

// CHECK: @canonicalizeStaticOperand
// CHECK: %[[ARG0:.+]]: !flow.dispatch.tensor<readonly:tensor<4x4xf32>>
// CHECK: flow.dispatch.tensor.load %[[ARG0]]
//      CHECK-SAME: offsets = [0, 0]
//      CHECK-SAME: sizes = [2, 2]
//      CHECK-SAME: strides = [1, 1]
//      CHECK-SAME: !flow.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<2x2xf32>

// -----

func.func @canonicalizePartiallyStaticOperands(%arg0: !flow.dispatch.tensor<readonly:tensor<4x4xf32>>, %offset: index, %size: index, %stride: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = flow.dispatch.tensor.load %arg0, offsets=[%offset, %c0], sizes=[%size, %c2], strides=[%stride, %c1] : !flow.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<?x?xf32>
  "test.sink"(%0) : (tensor<?x?xf32>) -> ()
  return
}

// CHECK: @canonicalizePartiallyStaticOperands
// CHECK: %[[ARG0:.+]]: !flow.dispatch.tensor<readonly:tensor<4x4xf32>>
// CHECK: %[[OFFSET:.+]]: index, %[[SIZE:.+]]: index, %[[STRIDE:.+]]: index
// CHECK: flow.dispatch.tensor.load %[[ARG0]]
//      CHECK-SAME: offsets = [%[[OFFSET]], 0]
//      CHECK-SAME: sizes = [%[[SIZE]], 2]
//      CHECK-SAME: strides = [%[[STRIDE]], 1]
//      CHECK-SAME: !flow.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<?x2xf32>

// -----

func.func @canonicalizeDispatchLoad(%arg0: !flow.dispatch.tensor<readonly:tensor<3x4x1x12x64xf32>>, %arg1 : index, %arg2: index, %arg3 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = flow.dispatch.tensor.load %arg0, offsets = [%arg1, %c0, 0, %arg2, %arg3], sizes = [1, 4, 1, 4, 32], strides = [%c1, %c1, 1, %c1, %c1] : !flow.dispatch.tensor<readonly:tensor<3x4x1x12x64xf32>> -> tensor<1x4x?x32xf32>
  "test.sink"(%0) : (tensor<1x4x?x32xf32>) -> ()
}

// CHECK:      @canonicalizeDispatchLoad
// CHECK-SAME: %[[ARG0:.+]]: !flow.dispatch.tensor<readonly:tensor<3x4x1x12x64xf32>>
// CHECK-SAME: %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index
//             CHECK: %[[LOAD:.+]] = flow.dispatch.tensor.load %arg0, offsets = [%[[ARG1]], 0, 0, %[[ARG2]], %[[ARG3]]], sizes = [1, 4, 1, 4, 32], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x4x1x12x64xf32>> -> tensor<1x4x4x32xf32>
//             CHECK: %[[CAST:.+]] = tensor.cast %[[LOAD]] : tensor<1x4x4x32xf32> to tensor<1x4x?x32xf32>
//             CHECK: "test.sink"(%[[CAST]]) : (tensor<1x4x?x32xf32>) -> ()

// -----

func.func @canonicalizeDimOfTensorTile(%arg0: !flow.dispatch.tensor<readonly:tensor<250x1024xf32>>, %arg1 : index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %0 = affine.min affine_map<(d0) -> (64, -d0 + 250)>(%arg1)
  %1 = flow.dispatch.tensor.load %arg0, offsets = [%arg2, 0], sizes = [%0, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<250x1024xf32>> -> tensor<?x1024xf32>
  %2 = tensor.dim %1, %c0 : tensor<?x1024xf32>
  "test.sink"(%2) : (index) -> ()
  return
}

// CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (-s0 + 250, 64)>
// CHECK: @canonicalizeDimOfTensorTile
// CHECK: %[[ARG0:.+]]: !flow.dispatch.tensor<readonly:tensor<250x1024xf32>>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index
// CHECK: %[[DIM:.+]] = affine.min #[[MAP]]()[%[[ARG1]]]
// CHECK: "test.sink"(%[[DIM]]) : (index) -> ()

// -----

func.func @foldCastIntoStore(%arg0: !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>,
    %arg1 : tensor<3x?xf32>, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index) {
  %c3 = arith.constant 3 : index
  %0 = tensor.cast %arg1 : tensor<3x?xf32> to tensor<?x?xf32>
  flow.dispatch.tensor.store %0, %arg0, offsets = [3, 4, 5], sizes = [%c3, 1, %arg2], strides = [1, 1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%arg3, %arg4, %arg5}
  return
}
//      CHECK: func @foldCastIntoStore
// CHECK-SAME:     %[[ARG0:.+]]: !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<3x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: index
//      CHECK:   flow.dispatch.tensor.store %[[ARG1]], %[[ARG0]]
// CHECK-SAME:       sizes = [3, 1, %[[ARG2]]]
// CHECK-SAME:       tensor<3x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%[[ARG3]], %[[ARG4]], %[[ARG5]]}
