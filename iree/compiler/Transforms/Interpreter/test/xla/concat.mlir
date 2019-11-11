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

// RUN: iree-opt --lower-xla-to-iree-interpreter %s --split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @concat.1D
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.1D(%arg0 : tensor<4xi32>, %arg1 : tensor<4xi32>) -> tensor<8xi32> {
  // CHECK-DAG: [[ARG0_MEMREF:%.+]] = iree.tensor_to_memref([[ARG0]]
  // CHECK-DAG: [[ARG1_MEMREF:%.+]] = iree.tensor_to_memref([[ARG1]]
  // CHECK:     [[RES:%.+]] = "iree_hl_interp.concat"([[ARG0_MEMREF]], [[ARG1_MEMREF]]) {dimension = 0 : i32}
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>

  // CHECK: [[RES_TENSOR:%.+]] = iree.memref_to_tensor([[RES]]
  // CHECK: return [[RES_TENSOR]]
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @concat.2D.Dim0
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.2D.Dim0(%arg0 : tensor<4x4xi32>, %arg1 : tensor<4x4xi32>) -> tensor<8x4xi32> {
  // CHECK-DAG: [[ARG0_MEMREF:%.+]]  = iree.tensor_to_memref([[ARG0]]
  // CHECK-DAG: [[ARG1_MEMREF:%.+]]  = iree.tensor_to_memref([[ARG1]]
  // CHECK:     [[RES:%.+]] = "iree_hl_interp.concat"([[ARG0_MEMREF]], [[ARG1_MEMREF]]) {dimension = 0 : i32}
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<8x4xi32>

  // CHECK: [[RES_TENSOR:%.+]] = iree.memref_to_tensor([[RES]]
  // CHECK: return [[RES_TENSOR]]
  return %0 : tensor<8x4xi32>
}

// -----

// CHECK-LABEL: func @concat.2D.Dim1
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.2D.Dim1(%arg0 : tensor<4x4xi32>, %arg1 : tensor<4x4xi32>) -> tensor<4x8xi32> {
  // CHECK-DAG: [[ARG0_MEMREF:%.+]]  = iree.tensor_to_memref([[ARG0]]
  // CHECK-DAG: [[ARG1_MEMREF:%.+]]  = iree.tensor_to_memref([[ARG1]]
  // CHECK:     [[RES:%.+]] = "iree_hl_interp.concat"([[ARG0_MEMREF]], [[ARG1_MEMREF]]) {dimension = 1 : i32}
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x8xi32>

  // CHECK: [[RES_TENSOR:%.+]] = iree.memref_to_tensor([[RES]]
  // CHECK: return [[RES_TENSOR]]
  return %0 : tensor<4x8xi32>
}
