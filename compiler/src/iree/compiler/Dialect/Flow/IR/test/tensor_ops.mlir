// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @tensorReshape
func.func @tensorReshape(%arg0 : tensor<4x4xf32>) -> tensor<16xf32> {
  // CHECK-NEXT: %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<16xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: @tensorReshapeScalar
func.func @tensorReshapeScalar(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @tensorReshapeDynamic
func.func @tensorReshapeDynamic(%arg0 : tensor<?x4xf32>) -> tensor<?x2xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  // CHECK: %0 = flow.tensor.reshape %arg0 : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c8}
  %0 = flow.tensor.reshape %arg0 : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c8}
  return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: @tensorReshapeComplex
func.func @tensorReshapeComplex(%arg0 : tensor<4x4xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  // CHECK-NEXT: flow.tensor.reshape %arg0 : tensor<4x4xcomplex<f32>> -> tensor<16xcomplex<f32>>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xcomplex<f32>> -> tensor<16xcomplex<f32>>
  return %0 : tensor<16xcomplex<f32>>
}

// -----

// CHECK-LABEL: @tensorLoad
func.func @tensorLoad(%arg0 : tensor<4x4xf32>, %arg1 : index, %arg2 : index) -> f32 {
  // CHECK-NEXT: %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<4x4xf32>
  %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<4x4xf32>
  return %0 : f32
}

// CHECK-LABEL: @tensorLoadScalar
func.func @tensorLoadScalar(%arg0 : tensor<f32>) -> f32 {
  // CHECK-NEXT: %0 = flow.tensor.load %arg0 : tensor<f32>
  %0 = flow.tensor.load %arg0 : tensor<f32>
  return %0 : f32
}

// CHECK-LABEL: @tensorLoadDynamic
func.func @tensorLoadDynamic(%arg0 : tensor<?x4xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  return %0 : f32
}

// -----

// CHECK-LABEL: @tensorStore
func.func @tensorStore(%arg0 : tensor<4x4xf32>, %arg1 : index, %arg2 : index, %arg3 : f32) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<4x4xf32>
  %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @tensorStoreScalar
func.func @tensorStoreScalar(%arg0 : f32, %arg1 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.store %arg0, %arg1 : tensor<f32>
  %0 = flow.tensor.store %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @tensorStoreDynamic
func.func @tensorStoreDynamic(%arg0 : tensor<?x4xf32>, %arg1 : index, %arg2 : index, %arg3 : f32) -> tensor<?x4xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorAlloca
func.func @tensorAlloca(%arg0: index) -> tensor<?x0x1xf32> {
  // CHECK-NEXT: = flow.tensor.alloca : tensor<?x0x1xf32>{%arg0}
  %0 = flow.tensor.alloca : tensor<?x0x1xf32>{%arg0}
  return %0 : tensor<?x0x1xf32>
}

// -----

// CHECK-LABEL: @tensorEmpty
func.func @tensorEmpty(%arg0: index) -> tensor<?x0x1xf32> {
  // CHECK-NEXT: = flow.tensor.empty : tensor<?x0x1xf32>{%arg0}
  %0 = flow.tensor.empty : tensor<?x0x1xf32>{%arg0}
  return %0 : tensor<?x0x1xf32>
}

// -----

// CHECK-LABEL: @tensorSplat
func.func @tensorSplat(%arg0 : f32) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @tensorSplatScalar
func.func @tensorSplatScalar(%arg0 : f32) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<f32>
  %0 = flow.tensor.splat %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @tensorSplatDynamic
func.func @tensorSplatDynamic(%arg0 : f32) -> tensor<?x4xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.splat %arg0 : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.splat %arg0 : tensor<?x4xf32>{%c4}
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorClone
func.func @tensorClone(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.clone %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.clone %arg0 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @tensorCloneScalar
func.func @tensorCloneScalar(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.clone %arg0 : tensor<f32>
  %0 = flow.tensor.clone %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @tensorCloneDynamic
func.func @tensorCloneDynamic(%arg0 : tensor<?x4xf32>) -> tensor<?x4xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.clone %arg0 : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.clone %arg0 : tensor<?x4xf32>{%c4}
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorSlice
func.func @tensorSlice(%arg0 : tensor<4x4xf32>, %arg1 : index, %arg2 : index) -> tensor<2x2xf32> {
  // CHECK-NEXT: %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<4x4xf32> -> tensor<2x2xf32>
  %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<4x4xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @tensorSliceDynamic
func.func @tensorSliceDynamic(%arg0 : tensor<?x4xf32>, %arg1 : index, %arg2 : index) -> tensor<?x2xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c2}
  %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c2}
  return %0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
func.func @tensorUpdate(%arg0 : tensor<2x2xf32>, %arg1 : tensor<4x4xf32>, %arg2 : index, %arg3 : index) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<2x2xf32> -> %arg1 as tensor<4x4xf32>
  %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<2x2xf32> -> %arg1 as tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @tensorUpdateDynamic
func.func @tensorUpdateDynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x4xf32>, %arg2 : index, %arg3 : index) -> tensor<?x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<?x?xf32>{%c1, %c2} -> %arg1 as tensor<?x4xf32>{%c3}
  %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<?x?xf32>{%c1, %c2} -> %arg1 as tensor<?x4xf32>{%c3}
  return %0 : tensor<?x4xf32>
}
