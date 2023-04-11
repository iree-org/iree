// RUN: iree-opt --iree-abi-convert-streamable-ops --cse --split-input-file %s --verify-diagnostics | FileCheck %s

// Tests most of the features of the conversion.

// CHECK: flow.func private @import(%arg0: tensor<?x2xi32> {some.arg_attr}, %arg1: tensor<?x4xf32>, %arg2: i32, %arg3: index) -> (%arg0, tensor<?x4xi8> {some.result_attr})
func.func private @import(tensor<?x2xi32> {some.arg_attr}, tensor<?x4xf32>, i32, index) ->
    (tensor<?x2xi32> {iree.abi.tied = 0 : index}, tensor<?x4xi8> {iree.abi.dims = [3 : index], some.result_attr}) attributes {
  iree.abi.streamable
}

// CHECK: func.func private @caller
func.func private @caller(%arg0: tensor<?x2xi32>, %arg1: tensor<?x4xf32>, %arg2: i32, %dim0: index) -> (tensor<?x2xi32>, tensor<?x4xi8>) {
  // CHECK-DAG: %[[ARG0_DIM0:.+]] = tensor.dim %arg0, %c0
  // CHECK-DAG: %[[ARG1_DIM0:.+]] = tensor.dim %arg1, %c0
  // CHECK: %[[RETS:.+]]:2 = flow.call @import(%arg0, %arg1, %arg2, %arg3) : (tensor<?x2xi32>{%[[ARG0_DIM0]]}, tensor<?x4xf32>{%[[ARG1_DIM0]]}, i32, index) -> (%arg0{%[[ARG0_DIM0]]}, tensor<?x4xi8>{%arg3})
  %0:2 = call @import(%arg0, %arg1, %arg2, %dim0) : (tensor<?x2xi32>, tensor<?x4xf32>, i32, index) -> (tensor<?x2xi32>, tensor<?x4xi8>)
  // CHECK: return %[[RETS]]#0, %[[RETS]]#1
  return %0#0, %0#1 : tensor<?x2xi32>, tensor<?x4xi8>
}

// -----

// Verifies if a user doesn't specify untied result dynamic dims we error out.

// expected-error @+1 {{missing dynamic dimensions on result 0}}
func.func private @importMissingResultDims(tensor<?x?xi32>, index, index) -> tensor<?x?xf32> attributes {
  iree.abi.streamable
}

// -----

// Tests that untied results with dynamic dimensions can resolve them.
// Users need to specify in such cases.

// CHECK: flow.func private @importWithResultDims(%arg0: tensor<?x?xi32>, %arg1: index, %arg2: index) -> tensor<?x?xf32>
func.func private @importWithResultDims(tensor<?x?xi32>, index, index) -> (tensor<?x?xf32> {iree.abi.dims = [1 : index, 2 : index]}) attributes {
  iree.abi.streamable
}

// CHECK: func.func private @callerWithResultDims
func.func private @callerWithResultDims(%arg0: tensor<?x?xi32>, %arg1: index, %arg2: index) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[ARG0_DIM0:.+]] = tensor.dim %arg0, %c0
  // CHECK-DAG: %[[ARG0_DIM1:.+]] = tensor.dim %arg0, %c1
  // CHECK: %[[RET:.+]] = flow.call @importWithResultDims(%arg0, %arg1, %arg2) : (tensor<?x?xi32>{%[[ARG0_DIM0]], %[[ARG0_DIM1]]}, index, index) -> tensor<?x?xf32>{%arg1, %arg2}
  %0 = call @importWithResultDims(%arg0, %arg1, %arg2) : (tensor<?x?xi32>, index, index) -> tensor<?x?xf32>
  // CHECK: return %[[RET]]
  return %0 : tensor<?x?xf32>
}

// -----

// Verifies if the user tries specifying result dims and a calculation function
// we properly error.

func.func private @calculateOverconstrainedResultDims(%arg0: index) -> index {
  return %arg0 : index
}

// expected-error @+1 {{cannot have both an explicit result shape calculation function}}
func.func private @importOverconstrainedResultDims(index) -> (tensor<2x?xf32> {iree.abi.dims = [0 : index]}) attributes {
  iree.abi.streamable,
  iree.abi.result_dims = @calculateOverconstrainedResultDims
}

// -----

// Tests using a shape computation function for computing result dimensions.

// CHECK: func.func private @calculateResultDims
func.func private @calculateResultDims(%arg0: tensor<1x?xi32>, %arg1: i32, %arg2: tensor<?xf32>) -> (index, index) {
  // Could do math here, call other imported host functions, etc. Note that
  // doing anything but tensor.dim on the tensor arguments will cause massive
  // performance penalties and should always be avoided.
  //
  // Note that only dynamic dimensions need to be returned.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %arg0_dim1 = tensor.dim %arg0, %c1 : tensor<1x?xi32>
  %arg2_dim0 = tensor.dim %arg2, %c0 : tensor<?xf32>
  return %arg0_dim1, %arg2_dim0 : index, index
}

// CHECK: flow.func private @importCustomResultDims(%arg0: tensor<1x?xi32>, %arg1: i32, %arg2: tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>)
func.func private @importCustomResultDims(tensor<1x?xi32>, i32, tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>) attributes {
  iree.abi.streamable,
  iree.abi.result_dims = @calculateResultDims
}

// CHECK: func.func private @callerCustomResultDims
func.func private @callerCustomResultDims(%arg0: tensor<1x?xi32>, %arg1: i32, %arg2: tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>) {
  // CHECK-DAG: %[[ARG0_DIM1:.+]] = tensor.dim %arg0, %c1
  // CHECK-DAG: %[[ARG2_DIM0:.+]] = tensor.dim %arg2, %c0
  // CHECK: %[[RET_DIMS:.+]]:2 = call @calculateResultDims(%arg0, %arg1, %arg2) : (tensor<1x?xi32>, i32, tensor<?xf32>) -> (index, index)
  // CHECK: %[[RETS:.+]]:2 = flow.call @importCustomResultDims(%arg0, %arg1, %arg2) : (tensor<1x?xi32>{%[[ARG0_DIM1]]}, i32, tensor<?xf32>{%[[ARG2_DIM0]]}) -> (tensor<2x?xf32>{%[[RET_DIMS]]#0}, tensor<?xi8>{%[[RET_DIMS]]#1})
  %0:2 = call @importCustomResultDims(%arg0, %arg1, %arg2) : (tensor<1x?xi32>, i32, tensor<?xf32>) -> (tensor<2x?xf32>, tensor<?xi8>)
  // CHECK: return %[[RETS]]#0, %[[RETS]]#1
  return %0#0, %0#1 : tensor<2x?xf32>, tensor<?xi8>
}

// -----

// Tests that results tied to operands get handled correctly and reuse the
// argument shapes.

// CHECK: flow.func private @importWithTies(%arg0: tensor<?x?xi32>) -> %arg0
func.func private @importWithTies(tensor<?x?xi32>) -> (tensor<?x?xi32> {iree.abi.tied = 0 : index}) attributes {
  iree.abi.streamable
}

// CHECK: func.func private @callerWithTies
func.func private @callerWithTies(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG: %[[ARG0_DIM0:.+]] = tensor.dim %arg0, %c0
  // CHECK-DAG: %[[ARG0_DIM1:.+]] = tensor.dim %arg0, %c1
  // CHECK: %[[RET:.+]] = flow.call @importWithTies(%arg0) : (tensor<?x?xi32>{%[[ARG0_DIM0]], %[[ARG0_DIM1]]}) -> %arg0{%[[ARG0_DIM0]], %[[ARG0_DIM1]]}
  %0 = call @importWithTies(%arg0) : (tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: return %[[RET]]
  return %0 : tensor<?x?xi32>
}

// -----

// Tests that attrs we don't know about are passed through to the new ops.

// CHECK: flow.func private @importPassThroughAttrs(%arg0: tensor<1xi32> {some.arg_attr}) -> tensor<1xi8> {some.result_attr} attributes {some.import_attr}
func.func private @importPassThroughAttrs(tensor<1xi32> {some.arg_attr}) -> (tensor<1xi8> {some.result_attr}) attributes {
  iree.abi.streamable,
  some.import_attr
}

// CHECK: func.func private @callerPassThroughArgs
func.func private @callerPassThroughArgs(%arg0: tensor<1xi32>) -> tensor<1xi8> {
  // CHECK: %[[RET:.+]] = flow.call @importPassThroughAttrs(%arg0) {some.call_attr} : (tensor<1xi32>) -> tensor<1xi8>
  %0 = call @importPassThroughAttrs(%arg0) {some.call_attr} : (tensor<1xi32>) -> tensor<1xi8>
  // CHECK: return %[[RET]]
  return %0 : tensor<1xi8>
}
