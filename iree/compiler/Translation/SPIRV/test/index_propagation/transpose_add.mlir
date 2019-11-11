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

// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false %s | FileCheck %s

// CHECK-DAG: [[MAP0:\#.*]] = ([[DIM0:d.*]], [[DIM1:d.*]]) -> ([[DIM1]], [[DIM0]])
// CHECK-DAG: [[MAP1:\#.*]] = ([[DIM0:d.*]], [[DIM1:d.*]]) -> ([[DIM0]], [[DIM1]])
module {
 // CHECK: func {{@.*}}({{%.*}}: memref<12x12xf32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]}}, {{\[}}[[MAP1]]{{\]\]}}}
 func @transpose_add(%arg0: memref<12x12xf32>, %arg1: memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: {{%.*}} = iree.load_input({{%.*}} : memref<12x12xf32>) {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]]{{\]}}, {{\[}}[[MAP1]], [[MAP1]]{{\]\]}}}
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    // CHECK: {{%.*}} = "xla_hlo.transpose"({{%.*}}) {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP1]]{{\]\]}}, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    %1 = "xla_hlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    // CHECK: {{%.*}} = xla_hlo.add {{%.*}} {{%.*}} {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]], [[MAP0]]{{\]\]}}}
    %2 = xla_hlo.add %0, %1 : tensor<12x12xf32>
    iree.store_output(%2 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}
