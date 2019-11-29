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

// RUN: iree-opt --lower-xla-to-iree-interpreter %s | IreeFileCheck %s

// CHECK-LABEL: @slice
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @slice(%arg : tensor<3x4xf32>) -> tensor<1x4xf32> {
  // CHECK-DAG:  [[SRC:%.+]]   = iree.tensor_to_memref([[ARG]]
  // CHECK-DAG:  [[SRC_INDICES:%.+]] = iree.constant[dense<[1, 0]>
  // CHECK-DAG:  [[DST:%.+]]     = "iree_hl_interp.alloc_heap"() : () -> memref<1x4xf32>
  // CHECK-DAG:  [[DST_INDICES:%.+]] = iree.constant[dense<0>
  // CHECK-DAG:  [[LENGTHS:%.+]] = iree.constant[dense<[1, 4]>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: [[RESULT_TENSOR:%.+]] = iree.memref_to_tensor([[DST]]
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: return [[RESULT_TENSOR]]
  return %result : tensor<1x4xf32>
}

// CHECK-LABEL: @slice_noncontiguous
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @slice_noncontiguous(%arg : tensor<3x4xf32>) -> tensor<2x2xf32> {
  // CHECK-DAG:  [[SRC:%.+]]   = iree.tensor_to_memref([[ARG]]
  // CHECK-DAG:  [[SRC_INDICES:%.+]] = iree.constant[dense<1>
  // CHECK-DAG:  [[DST:%.+]]     = "iree_hl_interp.alloc_heap"() : () -> memref<2x2xf32>
  // CHECK-DAG:  [[DST_INDICES:%.+]] = iree.constant[dense<0>
  // CHECK-DAG:  [[LENGTHS:%.+]] = iree.constant[dense<2>
  // CHECK-NEXT: "iree_hl_interp.copy"([[SRC]], [[SRC_INDICES]], [[DST]], [[DST_INDICES]], [[LENGTHS]])
  // CHECK-NEXT: [[RESULT_TENSOR:%.+]] = iree.memref_to_tensor([[DST]]
  %result = "xla_hlo.slice"(%arg) {start_indices = dense<1> : tensor<2xi64>, limit_indices = dense<3> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: return [[RESULT_TENSOR]]
  return %result : tensor<2x2xf32>
}
