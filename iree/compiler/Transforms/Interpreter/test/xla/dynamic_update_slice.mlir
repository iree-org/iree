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

// RUN: iree-opt --lower-xla-to-iree-interpreter %s --split-input-file | IreeFileCheck %s

// -----

// CHECK-LABEL: func @dynamic_update_slice.1D
// CHECK-SAME: [[OPERAND:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[UPDATE:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[INDICES:%[a-zA-Z0-9]+]]
func @dynamic_update_slice.1D(%operand : tensor<4xi32>, %update : tensor<1xi32>, %indices : tensor<i32>) -> tensor<4xi32> {
  // CHECK-DAG: [[OPERAND_MEMREF:%.+]] = iree.tensor_to_memref([[OPERAND]]
  // CHECK-DAG: [[UPDATE_MEMREF:%.+]] = iree.tensor_to_memref([[UPDATE]]
  // CHECK-DAG: [[INDICES_MEMREF:%.+]] = iree.tensor_to_memref([[INDICES]]
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_SHAPE:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_RESHAPED:%.+]] = "iree_hl_interp.reshape"([[INDICES_MEMREF]], [[INDICES_SHAPE]])
  // CHECK-DAG: [[INDICES_CONCAT:%.+]] = "iree_hl_interp.concat"([[INDICES_RESHAPED]]) {dimension = 0 : i32}
  // CHECK-DAG: [[SRC_INDICES:%.+]] = iree.constant[dense<0> : tensor<1xi64>
  // CHECK-DAG: [[DST:%.+]] = "iree_hl_interp.clone"([[OPERAND_MEMREF]])
  // CHECK-NEXT: "iree_hl_interp.copy"([[UPDATE_MEMREF]], [[SRC_INDICES]], [[DST]], [[INDICES_CONCAT]], [[LENGTHS]])
  %0 = "xla_hlo.dynamic-update-slice"(%operand, %update, %indices) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>

  // CHECK-NEXT: [[RES:%.+]] = iree.memref_to_tensor([[DST]]
  // CHECK-NEXT: return [[RES]]
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @dynamic_update_slice.2D
// CHECK-SAME: [[OPERAND:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[UPDATE:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[INDICES_0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[INDICES_1:%[a-zA-Z0-9]+]]
func @dynamic_update_slice.2D(%operand : tensor<2x4xi32>, %update : tensor<1x1xi32>, %indices_0 : tensor<i32>, %indices_1 : tensor<i32>) -> tensor<2x4xi32> {
  // CHECK-DAG: [[OPERAND_MEMREF:%.+]] = iree.tensor_to_memref([[OPERAND]]
  // CHECK-DAG: [[UPDATE_MEMREF:%.+]] = iree.tensor_to_memref([[UPDATE]]
  // CHECK-DAG: [[INDICES_0_MEMREF:%.+]] = iree.tensor_to_memref([[INDICES_0]]
  // CHECK-DAG: [[INDICES_1_MEMREF:%.+]] = iree.tensor_to_memref([[INDICES_1]]
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant[dense<1> : tensor<2xi64>
  // CHECK-DAG: [[INDICES_0_SHAPE:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_0_RESHAPED:%.+]] = "iree_hl_interp.reshape"([[INDICES_0_MEMREF]], [[INDICES_0_SHAPE]])
  // CHECK-DAG: [[INDICES_1_SHAPE:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_1_RESHAPED:%.+]] = "iree_hl_interp.reshape"([[INDICES_1_MEMREF]], [[INDICES_1_SHAPE]])
  // CHECK-DAG: [[INDICES_CONCAT:%.+]] = "iree_hl_interp.concat"([[INDICES_0_RESHAPED]], [[INDICES_1_RESHAPED]]) {dimension = 0 : i32}
  // CHECK-DAG: [[SRC_INDICES:%.+]] = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-NEXT: [[DST:%.+]] = "iree_hl_interp.clone"([[OPERAND_MEMREF]])
  // CHECK-NEXT: "iree_hl_interp.copy"([[UPDATE_MEMREF]], [[SRC_INDICES]], [[DST]], [[INDICES_CONCAT]], [[LENGTHS]])
  %0 = "xla_hlo.dynamic-update-slice"(%operand, %update, %indices_0, %indices_1) : (tensor<2x4xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<2x4xi32>

  // CHECK-NEXT: [[RES:%.+]] = iree.memref_to_tensor([[DST]]
  // CHECK-NEXT: return [[RES]]
  return %0 : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: func @dynamic_update_slice.1D.notlast
// CHECK-SAME: [[OPERAND:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[UPDATE:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[INDICES:%[a-zA-Z0-9]+]]
func @dynamic_update_slice.1D.notlast(%operand : tensor<4xi32>, %update : tensor<1xi32>, %indices : tensor<i32>) -> tensor<4xi32> {
  // CHECK-DAG: [[OPERAND_MEMREF:%.+]] = iree.tensor_to_memref([[OPERAND]]
  // CHECK-DAG: [[UPDATE_MEMREF:%.+]] = iree.tensor_to_memref([[UPDATE]]
  // CHECK-DAG: [[INDICES_MEMREF:%.+]] = iree.tensor_to_memref([[INDICES]]
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_SHAPE:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_RESHAPED:%.+]] = "iree_hl_interp.reshape"([[INDICES_MEMREF]], [[INDICES_SHAPE]])
  // CHECK-DAG: [[INDICES_CONCAT:%.+]] = "iree_hl_interp.concat"([[INDICES_RESHAPED]]) {dimension = 0 : i32}
  // CHECK-DAG: [[SRC_INDICES:%.+]] = iree.constant[dense<0> : tensor<1xi64>
  // CHECK-DAG: [[DST:%.+]] = "iree_hl_interp.clone"([[OPERAND_MEMREF]])
  // CHECK-NEXT: "iree_hl_interp.copy"([[UPDATE_MEMREF]], [[SRC_INDICES]], [[DST]], [[INDICES_CONCAT]], [[LENGTHS]])
  %0 = "xla_hlo.dynamic-update-slice"(%operand, %update, %indices) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>

  // CHECK-NEXT: [[DST_TENSOR:%.+]] = iree.memref_to_tensor([[DST]]
  // CHECK-NEXT: [[RES:%.+]] = xla_hlo.add [[OPERAND]], [[DST_TENSOR]]
  %1 = xla_hlo.add %operand, %0 : tensor<4xi32>

  // CHECK-DAG: return [[RES]]
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @dynamic_update_slice.2D.notlast
// CHECK-SAME: [[OPERAND:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[UPDATE:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[INDICES_0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[INDICES_1:%[a-zA-Z0-9]+]]
func @dynamic_update_slice.2D.notlast(%operand : tensor<2x4xi32>, %update : tensor<1x1xi32>, %indices_0 : tensor<i32>, %indices_1 : tensor<i32>) -> tensor<2x4xi32> {
  // CHECK-DAG: [[OPERAND_MEMREF:%.+]] = iree.tensor_to_memref([[OPERAND]]
  // CHECK-DAG: [[UPDATE_MEMREF:%.+]] = iree.tensor_to_memref([[UPDATE]]
  // CHECK-DAG: [[INDICES_0_MEMREF:%.+]] = iree.tensor_to_memref([[INDICES_0]]
  // CHECK-DAG: [[INDICES_1_MEMREF:%.+]] = iree.tensor_to_memref([[INDICES_1]]
  // CHECK-DAG: [[LENGTHS:%.+]] = iree.constant[dense<1> : tensor<2xi64>
  // CHECK-DAG: [[INDICES_0_SHAPE:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_0_RESHAPED:%.+]] = "iree_hl_interp.reshape"([[INDICES_0_MEMREF]], [[INDICES_0_SHAPE]])
  // CHECK-DAG: [[INDICES_1_SHAPE:%.+]] = iree.constant[dense<1> : tensor<1xi64>
  // CHECK-DAG: [[INDICES_1_RESHAPED:%.+]] = "iree_hl_interp.reshape"([[INDICES_1_MEMREF]], [[INDICES_1_SHAPE]])
  // CHECK-DAG: [[INDICES_CONCAT:%.+]] = "iree_hl_interp.concat"([[INDICES_0_RESHAPED]], [[INDICES_1_RESHAPED]]) {dimension = 0 : i32}
  // CHECK-DAG: [[SRC_INDICES:%.+]] = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-DAG: [[DST:%.+]] = "iree_hl_interp.clone"([[OPERAND_MEMREF]])
  // CHECK-NEXT: "iree_hl_interp.copy"([[UPDATE_MEMREF]], [[SRC_INDICES]], [[DST]], [[INDICES_CONCAT]], [[LENGTHS]])
  %0 = "xla_hlo.dynamic-update-slice"(%operand, %update, %indices_0, %indices_1) : (tensor<2x4xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<2x4xi32>

  // CHECK-NEXT: [[DST_TENSOR:%.+]] = iree.memref_to_tensor([[DST]]
  // CHECK-NEXT: [[RES:%.+]] = xla_hlo.add [[OPERAND]], [[DST_TENSOR]]
  %1 = xla_hlo.add %operand, %0 : tensor<2x4xi32>

  // CHECK-NEXT: return [[RES]]
  return %1 : tensor<2x4xi32>
}
