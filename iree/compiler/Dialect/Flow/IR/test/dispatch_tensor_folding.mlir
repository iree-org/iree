// RUN: iree-opt -allow-unregistered-dialect -split-input-file -canonicalize %s | IreeFileCheck %s

func @canonicalizeStaticOperands(%arg0: !flow.dispatch.tensor<readonly:4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = flow.dispatch.tensor.load %arg0, offsets=[%c0, %c0], sizes=[%c2, %c2], strides=[%c1, %c1] : !flow.dispatch.tensor<readonly:4x4xf32> -> tensor<?x?xf32>
    "test.sink"(%0) : (tensor<?x?xf32>) -> ()
    return
}

// CHECK: @canonicalizeStaticOperand
// CHECK: %[[ARG0:.+]]: !flow.dispatch.tensor<readonly:4x4xf32>
// CHECK: flow.dispatch.tensor.load %[[ARG0]]
//      CHECK-SAME: offsets = [0, 0]
//      CHECK-SAME: sizes = [2, 2]
//      CHECK-SAME: strides = [1, 1]
//      CHECK-SAME: !flow.dispatch.tensor<readonly:4x4xf32> -> tensor<2x2xf32>

// -----

func @canonicalizePartiallyStaticOperands(%arg0: !flow.dispatch.tensor<readonly:4x4xf32>, %offset: index, %size: index, %stride: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = flow.dispatch.tensor.load %arg0, offsets=[%offset, %c0], sizes=[%size, %c2], strides=[%stride, %c1] : !flow.dispatch.tensor<readonly:4x4xf32> -> tensor<?x?xf32>
    "test.sink"(%0) : (tensor<?x?xf32>) -> ()
    return
}

// CHECK: @canonicalizePartiallyStaticOperands
// CHECK: %[[ARG0:.+]]: !flow.dispatch.tensor<readonly:4x4xf32>
// CHECK: %[[OFFSET:.+]]: index, %[[SIZE:.+]]: index, %[[STRIDE:.+]]: index
// CHECK: flow.dispatch.tensor.load %[[ARG0]]
//      CHECK-SAME: offsets = [%[[OFFSET]], 0]
//      CHECK-SAME: sizes = [%[[SIZE]], 2]
//      CHECK-SAME: strides = [%[[STRIDE]], 1]
//      CHECK-SAME: !flow.dispatch.tensor<readonly:4x4xf32> -> tensor<?x2xf32>

// -----

func @canonicalizeDimOfTensorTile(%arg0: !flow.dispatch.tensor<readonly:250x1024xf32>, %arg1 : index, %arg2: index) {
    %c0 = arith.constant 0 : index
    %0 = affine.min affine_map<(d0) -> (64, -d0 + 250)>(%arg1)
    %1 = flow.dispatch.tensor.load %arg0, offsets = [%arg2, 0], sizes = [%0, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:250x1024xf32> -> tensor<?x1024xf32>
    %2 = tensor.dim %1, %c0 : tensor<?x1024xf32>
    "test.sink"(%2) : (index) -> ()
    return
}

// CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (64, -s0 + 250)>
// CHECK: @canonicalizeDimOfTensorTile
// CHECK: %[[ARG0:.+]]: !flow.dispatch.tensor<readonly:250x1024xf32>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index
// CHECK: %[[DIM:.+]] = affine.min #[[MAP]]()[%[[ARG1]]]
// CHECK: "test.sink"(%[[DIM]]) : (index) -> ()
