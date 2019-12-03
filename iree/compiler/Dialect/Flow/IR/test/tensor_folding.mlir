// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests folding and canonicalization of tensor ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @reshapeNoOp
func @reshapeNoOp(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @reshapeNoOpScalar
func @reshapeNoOpScalar(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: return %arg0 : tensor<f32>
  %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @reshapeTransitive
func @reshapeTransitive(%arg0 : tensor<4x4xf32>) -> tensor<8x2xf32> {
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<2x8xf32>
  // CHECK-NEXT: [[T:%.+]] = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<8x2xf32>
  %1 = flow.tensor.reshape %0 : tensor<2x8xf32> -> tensor<8x2xf32>
  // CHECK-NEXT: return [[T]] : tensor<8x2xf32>
  return %1 : tensor<8x2xf32>
}

// -----

// CHECK-LABEL: @loadConst
func @loadConst() -> i32 {
  %0 = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %c0 = constant 0 : i32
  %c1 = constant 1 : i32
  // CHECK-NEXT: [[C2:%.+]] = constant 2 : i32
  %2 = flow.tensor.load %0[%c1, %c0] : tensor<2x2xi32>
  // CHECK-NEXT: return [[C2]]
  return %2 : i32
}

// CHECK-LABEL: @loadConstScalar
func @loadConstScalar() -> i32 {
  %0 = constant dense<4> : tensor<i32>
  // CHECK-NEXT: [[C4:%.+]] = constant 4 : i32
  %1 = flow.tensor.load %0 : tensor<i32>
  // CHECK-NEXT: return [[C4]]
  return %1 : i32
}

// -----

// CHECK-LABEL: @storeConst
func @storeConst() -> tensor<2x2xi32> {
  %0 = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %c0 = constant 0 : i32
  %c1 = constant 1 : i32
  %c4 = constant 4 : i32
  // CHECK-NEXT: [[C:%.+]] = constant dense<[
  // CHECK-SAME:     [0, 1], [4, 3]
  // CHECK-SAME: ]> : tensor<2x2xi32>
  %1 = flow.tensor.store %c4, %0[%c1, %c0] : tensor<2x2xi32>
  // CHECK-NEXT: return [[C]]
  return %1 : tensor<2x2xi32>
}

// CHECK-LABEL: @storeConstScalar
func @storeConstScalar() -> tensor<i32> {
  %0 = constant dense<0> : tensor<i32>
  %1 = constant 4 : i32
  // CHECK-NEXT: [[C:%.+]] = constant dense<4> : tensor<i32>
  %2 = flow.tensor.store %1, %0 : tensor<i32>
  // CHECK-NEXT: return [[C]]
  return %2 : tensor<i32>
}

// -----

// CHECK-LABEL: @splatConst
func @splatConst() -> tensor<4xi32> {
  %0 = constant 4 : i32
  // CHECK-NEXT: [[C:%.+]] = constant dense<4> : tensor<4xi32>
  %1 = flow.tensor.splat %0 : tensor<4xi32>
  // CHECK-NEXT: return [[C]]
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: @splatConstScalar
func @splatConstScalar() -> tensor<i32> {
  %0 = constant 4 : i32
  // CHECK-NEXT: [[C:%.+]] = constant dense<4> : tensor<i32>
  %1 = flow.tensor.splat %0 : tensor<i32>
  // CHECK-NEXT: return [[C]]
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: @cloneConst
func @cloneConst() -> tensor<4xi32> {
  %0 = constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK-NEXT: [[C:%.+]] = constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %1 = flow.tensor.clone %0 : tensor<4xi32>
  // CHECK-NEXT: return [[C]]
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: @cloneDynamic
func @cloneDynamic(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = flow.tensor.clone %arg0 : tensor<4xi32>
  // CHECK-NEXT: return %arg0
  return %0 : tensor<4xi32>
}

// -----

// TODO(benvanik): const folder for slice.

// -----

// TODO(benvanik): const folder for update.

// CHECK-LABEL: @updateReplace
func @updateReplace(%arg0 : tensor<4xi32>, %arg1 : tensor<4xi32>) -> tensor<4xi32> {
  %c0 = constant 0 : i32
  %0 = flow.tensor.update %arg0, %arg1[%c0] : tensor<4xi32> -> tensor<4xi32>
  // CHECK-NEXT: return %arg0
  return %0 : tensor<4xi32>
}
