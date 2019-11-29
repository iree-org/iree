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

// CHECK: [[MAP0:\#.*]] = ([[DIM0:d.*]], [[DIM1:d.*]]) -> ([[DIM1]], [[DIM0]])

module {
   // CHECK: func {{@.*}}({{%.*}}: memref<12x42xi32> {iree.index_computation_info = {{\[\[}}[[MAP0]]{{\]\]}}}
  func @simple_load_store(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: {{%.*}} = iree.load_input({{%.*}} : memref<12x42xi32>) {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]]{{\]\]}}}
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // CHECK: {{%.*}} = "xla_hlo.copy"({{%.*}}) {iree.index_computation_info = {{\[\[}}[[MAP0]], [[MAP0]]{{\]\]}}}
    %1 = "xla_hlo.copy"(%0) : (tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}
