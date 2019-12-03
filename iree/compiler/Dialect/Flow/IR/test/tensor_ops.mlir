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

// Tests printing and parsing of tensor ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @tensorReshape
func @tensorReshape(%arg0 : tensor<4x4xf32>) -> tensor<16xf32> {
  // CHECK-NEXT: %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<16xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: @tensorReshapeScalar
func @tensorReshapeScalar(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorLoad
func @tensorLoad(%arg0 : tensor<4x4xf32>, %arg1 : i32, %arg2 : i32) -> f32 {
  // CHECK-NEXT: %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<4x4xf32>
  %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<4x4xf32>
  return %0 : f32
}

// CHECK-LABEL: @tensorLoadScalar
func @tensorLoadScalar(%arg0 : tensor<f32>) -> f32 {
  // CHECK-NEXT: %0 = flow.tensor.load %arg0 : tensor<f32>
  %0 = flow.tensor.load %arg0 : tensor<f32>
  return %0 : f32
}

// -----

// CHECK-LABEL: @tensorStore
func @tensorStore(%arg0 : tensor<4x4xf32>, %arg1 : i32, %arg2 : i32, %arg3 : f32) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<4x4xf32>
  %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @tensorStoreScalar
func @tensorStoreScalar(%arg0 : f32, %arg1 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.store %arg0, %arg1 : tensor<f32>
  %0 = flow.tensor.store %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorSplat
func @tensorSplat(%arg0 : f32) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @tensorSplatScalar
func @tensorSplatScalar(%arg0 : f32) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<f32>
  %0 = flow.tensor.splat %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorClone
func @tensorClone(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.clone %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.clone %arg0 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @tensorCloneScalar
func @tensorCloneScalar(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.clone %arg0 : tensor<f32>
  %0 = flow.tensor.clone %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorSlice
func @tensorSlice(%arg0 : tensor<4x4xf32>, %arg1 : i32, %arg2 : i32) -> tensor<2x2xf32> {
  // CHECK-NEXT: %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<4x4xf32> -> tensor<2x2xf32>
  %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<4x4xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
func @tensorUpdate(%arg0 : tensor<2x2xf32>, %arg1 : tensor<4x4xf32>, %arg2 : i32, %arg3 : i32) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<2x2xf32> -> tensor<4x4xf32>
  %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<2x2xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
