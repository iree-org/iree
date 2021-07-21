// RUN: iree-opt -split-input-file -iree-convert-upstream-to-iree %s | IreeFileCheck %s

func @static_tensor_cast_to_dynamic(%arg0: tensor<4x4xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[C4_0:.*]] = constant 4 : index
  // CHECK-DAG: %[[C4_1:.*]] = constant 4 : index
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<?x?xf32>{%[[C4_0]], %[[C4_1]]}
  // CHECK: return %[[RESULT]]
  %0 = tensor.cast %arg0 : tensor<4x4xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
func @dynamic_tensor_cast_to_static(%arg0: tensor<?xf32>) -> tensor<4xf32> {
  // CHECK: %[[C4_0:.*]] = constant 4 : index
  // CHECK: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<?xf32>{%[[C4_0]]} -> tensor<4xf32>
  // CHECK: return %[[RESULT]]
  %0 = tensor.cast %arg0 : tensor<?xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
func @dynamic_tensor_cast_to_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x3xf32> {
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x?xf32>
  // CHECK-DAG: %[[C3:.*]] = constant 3 : index
  // CHECK: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<?x?xf32>{%[[D0]], %[[C3]]} -> tensor<?x3xf32>{%[[D0]]}
  // CHECK: return %[[RESULT]]
  %0 = tensor.cast %arg0 : tensor<?x?xf32> to tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}
