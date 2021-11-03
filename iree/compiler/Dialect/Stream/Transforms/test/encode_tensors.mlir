// RUN: iree-opt -split-input-file -iree-stream-encode-tensors %s | IreeFileCheck %s

// CHECK-LABEL: @denseTensorSizeOf
func @denseTensorSizeOf(%arg0: index) -> index {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 20 : index
  // CHECK: %[[DYNAMIC_SIZE:.+]] = arith.muli %arg0, %[[STATIC_SIZE]] : index
  %0 = stream.tensor.sizeof tensor<?x5xf32>{%arg0} : index
  // CHECK: return %[[DYNAMIC_SIZE]]
  return %0 : index
}

// -----

// CHECK-LABEL: @denseTensorConstant
func @denseTensorConstant(%arg0: index) -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 1280 : index
  // CHECK: %[[DYNAMIC_SIZE:.+]] = arith.muli %arg0, %[[STATIC_SIZE]] : index
  // CHECK: %[[RET:.+]] = stream.async.constant : !stream.resource<constant>{%[[DYNAMIC_SIZE]]} = dense<0.000000e+00> : tensor<1x5x64xf32>
  %0 = stream.tensor.constant : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorSplat
func @denseTensorSplat(%arg0: f32, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[RET:.+]] = stream.async.splat %arg0 : f32 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : f32 -> tensor<?x1x10xf32>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<*>
}

// -----

// NOTE: clone likes to fold; the fills ensure it doesn't.

// CHECK-LABEL: @denseTensorClone
func @denseTensorClone(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: f32) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[RET:.+]] = stream.async.clone %arg0 : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.clone %arg0 : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2}
  %1 = stream.tensor.fill %arg3, %0[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg1} in %0 as !stream.resource<*>{%arg2}
  return %0, %1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSlice
func @denseTensorSlice(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 4 : index
  // CHECK: %[[END:.+]] = arith.addi %arg4, %[[OFFSET]] : index
  // CHECK: %[[RET:.+]] = stream.async.slice %arg0[%[[OFFSET]] to %[[END]]] : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg4}
  %0 = stream.tensor.slice %arg0[%c0, %c1 for %arg3, %c1] : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x1xf32>{%arg3} in !stream.resource<*>{%arg4}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFill
func @denseTensorFill(%arg0: f32, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[LENGTH:.+]] = arith.constant 20 : index
  // CHECK: %[[RET:.+]] = stream.async.fill %arg0, %arg1[%[[OFFSET]] to %[[LENGTH]] for %[[LENGTH]]] : f32 -> %arg1 as !stream.resource<*>{%arg3}
  %0 = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorUpdate
func @denseTensorUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.update %arg0, %arg2[%[[OFFSET]] to %arg1] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg4}
  %0 = stream.tensor.update %arg0, %arg2[%c0, %c0] : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<?x4xf32>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorLoad
func @denseTensorLoad(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.load %arg0[%[[OFFSET]]] : !stream.resource<staging>{%arg2} -> f32
  %0 = stream.tensor.load %arg0[%c0] : tensor<?xf32>{%arg1} in !stream.resource<staging>{%arg2} -> f32
  // CHECK: return %[[RET]]
  return %0 : f32
}

// -----

// CHECK-LABEL: @denseTensorLoadRank0
func @denseTensorLoadRank0(%arg0: !stream.resource<staging>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.load %arg0[%[[OFFSET]]] : !stream.resource<staging>{%arg1} -> f32
  %0 = stream.tensor.load %arg0 : tensor<f32> in !stream.resource<staging>{%arg1} -> f32
  // CHECK: return %[[RET]]
  return %0 : f32
}

// -----

// CHECK-LABEL: @denseTensorStore
func @denseTensorStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.store %arg3, %arg0[%[[OFFSET]]] : f32 -> %arg0 as !stream.resource<staging>{%arg2}
  %0 = stream.tensor.store %arg3, %arg0[%c0] : f32 -> tensor<?xf32>{%arg1} in %arg0 as !stream.resource<staging>{%arg2}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @denseTensorStoreRank0
func @denseTensorStoreRank0(%arg0: !stream.resource<staging>, %arg1: index, %arg2: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.store %arg2, %arg0[%[[OFFSET]]] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  %0 = stream.tensor.store %arg2, %arg0 : f32 -> tensor<f32> in %arg0 as !stream.resource<staging>{%arg1}
  // CHECK: return %[[RET]]
  return %0 : !stream.resource<staging>
}
