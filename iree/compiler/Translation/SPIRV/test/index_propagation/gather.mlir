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

// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false %s | IreeFileCheck %s

// CHECK: [[MAP0:\#.*]] = ([[DIM00:d.*]], [[DIM01:d.*]], [[DIM02:d.*]]) -> (s0, 0, [[DIM00]])
// CHECK: [[MAP1:\#.*]] = ([[DIM10:d.*]], [[DIM11:d.*]], [[DIM12:d.*]]) -> (0)
// CHECK: [[MAP2:\#.*]] = ([[DIM20:d.*]], [[DIM21:d.*]], [[DIM22:d.*]]) -> (0, [[DIM20]])

module {
  // CHECK: func @foo
  // CHECK-SAME: [[ARG0:%.*]]: memref<5x1x10xf32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  // CHECK-SAME: [[ARG1:%.*]]: memref<i64> {iree.index_computation_info = {{\[\[}}[[MAP1]]{{\]\]}}, iree.symbol_number_info = {{\[\[}}[[MAP1]], 0 : i32{{\]\]}}}
  func @foo(%arg0: memref<5x1x10xf32>, %arg1: memref<i64>, %arg2: memref<1x10xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[10, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<5x1x10xf32>) : tensor<5x1x10xf32>
    %1 = iree.load_input(%arg1 : memref<i64>) : tensor<i64>
    %2 = "xla_hlo.gather"(%0, %1) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<[1, 1, 10]> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x10xf32>, tensor<i64>) -> tensor<1x10xf32>
    iree.store_output(%2 : tensor<1x10xf32>, %arg2 : memref<1x10xf32>)
    iree.return
  }
}
