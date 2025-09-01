// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

// CHECK-LABEL: @static_tensor_bitcast
util.func public @static_tensor_bitcast(%arg0: tensor<4x4xf32>) -> tensor<4x4xi32> {
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.bitcast %arg0 : tensor<4x4xf32> -> tensor<4x4xi32>
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.bitcast %arg0 : tensor<4x4xf32> to tensor<4x4xi32>
  util.return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @dynamic_tensor_bitcast
util.func public @dynamic_tensor_bitcast(%arg0: tensor<?x?xf32>) -> tensor<?x?xi32> {
  // CHECK: %[[DIM0:.+]] = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  // CHECK: %[[RESULT:.+]] = flow.tensor.bitcast %arg0 : tensor<?x?xf32>{%[[DIM0]], %[[DIM1]]} -> tensor<?x?xi32>{%[[DIM0]], %[[DIM1]]}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.bitcast %arg0 : tensor<?x?xf32> to tensor<?x?xi32>
  util.return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @tensor_ext_bitcast
util.func public @tensor_ext_bitcast(%arg0: tensor<2x?xf32>, %arg1: index, %arg2: index, %arg3:index) -> tensor<3x?x?xi16> {
  // CHECK: %[[BITCAST:.*]] = flow.tensor.bitcast %arg0 : tensor<2x?xf32>{%arg1} -> tensor<3x?x?xi16>{%arg2, %arg3}
  %0 = iree_tensor_ext.bitcast %arg0 : tensor<2x?xf32>{%arg1} -> tensor<3x?x?xi16>{%arg2, %arg3}
  // CHECK-NEXT: return %[[BITCAST]]
  util.return %0 : tensor<3x?x?xi16>
}

// -----

// Verify that bitcasts in dispatches don't get converted.

// CHECK-LABEL: @tensor_ext_bitcast_in_dispatch_workgroups
util.func public @tensor_ext_bitcast_in_dispatch_workgroups(%arg0: tensor<2xf32>, %x: index) -> tensor<4xi16> {
  %0 = flow.dispatch.workgroups[%x](%arg0) : (tensor<2xf32>) -> tensor<4xi16> = (
    %arg: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>,
    %ret: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi16>>
  ) {
    %arg_value = iree_tensor_ext.dispatch.tensor.load %arg, offsets=[0], sizes=[2], strides=[1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
    // CHECK-NOT: flow.tensor.bitcast
    %1 = iree_tensor_ext.bitcast %arg_value : tensor<2xf32> -> tensor<4xi16>
    iree_tensor_ext.dispatch.tensor.store %1, %ret,  offsets=[0], sizes=[4], strides=[1]
      : tensor<4xi16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi16>>
    flow.return
  }
  util.return %0 : tensor<4xi16>
}
