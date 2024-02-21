// RUN: iree-opt --iree-abi-convert-streamable-ops --cse --split-input-file %s --verify-diagnostics | FileCheck %s

// Tests using a shape computation function for computing result dimensions.

// CHECK: util.func private @calculateResultDims
util.func private @calculateResultDims(%arg0: tensor<1x?xi32>, %arg1: i32, %arg2: tensor<?xf32>) -> (index, index) {
  // Could do math here, call other imported host functions, etc. Note that
  // doing anything but tensor.dim on the tensor arguments will cause massive
  // performance penalties and should always be avoided.
  //
  // Note that only dynamic dimensions need to be returned.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %arg0_dim1 = tensor.dim %arg0, %c1 : tensor<1x?xi32>
  %arg2_dim0 = tensor.dim %arg2, %c0 : tensor<?xf32>
  util.return %arg0_dim1, %arg2_dim0 : index, index
}

// CHECK: flow.func private @importCustomResultDims(%arg0: tensor<1x?xi32>, %arg1: i32, %arg2: tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>)
util.func private @importCustomResultDims(%arg0: tensor<1x?xi32>, %arg1: i32, %arg2: tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>) attributes {
  iree.abi.streamable,
  iree.abi.result_dims = @calculateResultDims
}

// CHECK: util.func private @callerCustomResultDims
util.func private @callerCustomResultDims(%arg0: tensor<1x?xi32>, %arg1: i32, %arg2: tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>) {
  // CHECK-DAG: %[[ARG0_DIM1:.+]] = tensor.dim %arg0, %c1
  // CHECK-DAG: %[[ARG2_DIM0:.+]] = tensor.dim %arg2, %c0
  // CHECK: %[[RET_DIMS:.+]]:2 = util.call @calculateResultDims(%arg0, %arg1, %arg2) : (tensor<1x?xi32>, i32, tensor<?xf32>) -> (index, index)
  // CHECK: %[[RETS:.+]]:2 = flow.call @importCustomResultDims(%arg0, %arg1, %arg2) : (tensor<1x?xi32>{%[[ARG0_DIM1]]}, i32, tensor<?xf32>{%[[ARG2_DIM0]]}) -> (tensor<2x?xf32>{%[[RET_DIMS]]#0}, tensor<?xi8>{%[[RET_DIMS]]#1})
  %0:2 = util.call @importCustomResultDims(%arg0, %arg1, %arg2) : (tensor<1x?xi32>, i32, tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>)
  // CHECK: util.return %[[RETS]]#0, %[[RETS]]#1
  util.return %0#0, %0#1 : tensor<2x?xf32>, tensor<?xi8>
}
