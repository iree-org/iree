// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @tensorImport
func.func @tensorImport(%arg0: !hal.buffer_view, %arg1: index) -> !stream.resource<external> {
  %c20 = arith.constant 20 : index
  // CHECK: = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  return %0 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @tensorExport
func.func @tensorExport(%arg0: !stream.resource<external>, %arg1: index) -> !hal.buffer_view {
  %c200 = arith.constant 200 : index
  // CHECK: = stream.tensor.export %arg0 : tensor<?x1x10xf32>{%arg1} in !stream.resource<external>{%c200} -> !hal.buffer_view
  %0 = stream.tensor.export %arg0 : tensor<?x1x10xf32>{%arg1} in !stream.resource<external>{%c200} -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @tensorSizeOf
func.func @tensorSizeOf(%arg0: index) -> index {
  // CHECK: = stream.tensor.sizeof tensor<?x5xf32>{%arg0} : index
  %0 = stream.tensor.sizeof tensor<?x5xf32>{%arg0} : index
  return %0 : index
}

// -----

// CHECK-LABEL: @tensorEmpty
func.func @tensorEmpty(%arg0: index, %arg1: index) -> !stream.resource<*> {
  // CHECK: = stream.tensor.empty : tensor<?x0xf32>{%arg0} in !stream.resource<*>{%arg1}
  %0 = stream.tensor.empty : tensor<?x0xf32>{%arg0} in !stream.resource<*>{%arg1}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorConstant
func.func @tensorConstant(%arg0: index) -> !stream.resource<constant> {
  // CHECK: = stream.tensor.constant : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  %0 = stream.tensor.constant : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @tensorSplat
func.func @tensorSplat(%arg0: f32, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: = stream.tensor.splat %arg0 : f32 -> tensor<?x1x10xf32>{%arg1} in !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : f32 -> tensor<?x1x10xf32>{%arg1} in !stream.resource<*>{%arg2}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorClone
func.func @tensorClone(%arg0: !stream.resource<*>, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: = stream.tensor.clone %arg0 : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2}
  %0 = stream.tensor.clone %arg0 : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorSlice
func.func @tensorSlice(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: = stream.tensor.slice %arg0[%c0, %c1 for %arg3, %c1] : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x1xf32>{%arg3} in !stream.resource<*>{%arg4}
  %0 = stream.tensor.slice %arg0[%c0, %c1 for %arg3, %c1] : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x1xf32>{%arg3} in !stream.resource<*>{%arg4}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorFill
func.func @tensorFill(%arg0: f32, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  %0 = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorUpdate
func.func @tensorUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: = stream.tensor.update %arg0, %arg2[%c0, %c0] : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<?x4xf32>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  %0 = stream.tensor.update %arg0, %arg2[%c0, %c0] : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<?x4xf32>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorLoad
func.func @tensorLoad(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.tensor.load %arg0[%c0] : tensor<?xf32>{%arg1} in !stream.resource<staging>{%arg2} -> f32
  %0 = stream.tensor.load %arg0[%c0] : tensor<?xf32>{%arg1} in !stream.resource<staging>{%arg2} -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @tensorLoadRank0
func.func @tensorLoadRank0(%arg0: !stream.resource<staging>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.tensor.load %arg0 : tensor<f32> in !stream.resource<staging>{%arg1} -> f32
  %0 = stream.tensor.load %arg0 : tensor<f32> in !stream.resource<staging>{%arg1} -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @tensorStore
func.func @tensorStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.tensor.store %arg3, %arg0[%c0] : f32 -> tensor<?xf32>{%arg1} in %arg0 as !stream.resource<staging>{%arg2}
  %0 = stream.tensor.store %arg3, %arg0[%c0] : f32 -> tensor<?xf32>{%arg1} in %arg0 as !stream.resource<staging>{%arg2}
  return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @tensorStoreRank0
func.func @tensorStoreRank0(%arg0: !stream.resource<staging>, %arg1: index, %arg2: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.tensor.store %arg2, %arg0 : f32 -> tensor<f32> in %arg0 as !stream.resource<staging>{%arg1}
  %0 = stream.tensor.store %arg2, %arg0 : f32 -> tensor<f32> in %arg0 as !stream.resource<staging>{%arg1}
  return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @tensorTrace
//  CHECK-SAME: (%[[TENSOR0:.+]]: !stream.resource<staging>, %[[TENSOR0_SIZE:.+]]: index, %[[TENSOR1:.+]]: !stream.resource<staging>, %[[TENSOR1_SIZE:.+]]: index, %[[TENSOR1_DIM0:.+]]: index, %[[TENSOR1_DIM2:.+]]: index)
func.func @tensorTrace(%tensor0: !stream.resource<staging>, %tensor0_size: index, %tensor1: !stream.resource<staging>, %tensor1_size: index, %tensor1_dim0: index, %tensor1_dim2: index) {
  //      CHECK: stream.tensor.trace "FOOBAR" = [
  // CHECK-NEXT:   %[[TENSOR0]] : tensor<5xf32> in !stream.resource<staging>{%[[TENSOR0_SIZE]]},
  // CHECK-NEXT:   %[[TENSOR1]] : tensor<?x3x?xi32>{%[[TENSOR1_DIM0]], %[[TENSOR1_DIM2]]} in !stream.resource<staging>{%[[TENSOR1_SIZE]]}
  // CHECK-NEXT: ]
  stream.tensor.trace "FOOBAR" = [
    %tensor0 : tensor<5xf32> in !stream.resource<staging>{%tensor0_size},
    %tensor1 : tensor<?x3x?xi32>{%tensor1_dim0, %tensor1_dim2} in !stream.resource<staging>{%tensor1_size}
  ]
  return
}
