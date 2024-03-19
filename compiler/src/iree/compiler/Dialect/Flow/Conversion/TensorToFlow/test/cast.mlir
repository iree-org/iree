// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

 util.func public @static_tensor_cast_to_dynamic(%arg0: tensor<4x4xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<?x?xf32>{%[[C4]], %[[C4]]}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.cast %arg0 : tensor<4x4xf32> to tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// -----

 util.func public @dynamic_tensor_cast_to_static(%arg0: tensor<?xf32>) -> tensor<4xf32> {
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<?xf32>{%[[C4]]} -> tensor<4xf32>
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.cast %arg0 : tensor<?xf32> to tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// -----

 util.func public @dynamic_tensor_cast_to_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x3xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x?xf32>
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<?x?xf32>{%[[D0]], %[[C3]]} -> tensor<?x3xf32>{%[[D0]]}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.cast %arg0 : tensor<?x?xf32> to tensor<?x3xf32>
  util.return %0 : tensor<?x3xf32>
}

// -----

 util.func public @tensor_cast_within_dispatch_workgroups_not_converted() -> tensor<f32> {
  %x = arith.constant 100 : index
  %0 = flow.dispatch.workgroups[%x]() : () -> (tensor<f32>) = () {
    // CHECK: = tensor.cast %[[source:.+]] : tensor<4x4xf32> to tensor<?x?xf32>
    %1 = "test.source"() : () -> (tensor<4x4xf32>)
    %2 = tensor.cast %1 : tensor<4x4xf32> to tensor<?x?xf32>
    "test.sink"(%2) : (tensor<?x?xf32>) -> ()
    flow.return
  }
  util.return %0 : tensor<f32>
}
