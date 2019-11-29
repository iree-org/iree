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

// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv %s | FileCheck %s --enable-var-scope

module {
  // CHECK: spv.globalVariable [[ARG0:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1:@.*]] bind(0, 1)
  // CHECK: spv.globalVariable [[ARG2:@.*]] bind(0, 2)
  func @foo(%arg0: memref<5x1x10xf32>, %arg1: memref<i64>, %arg2: memref<1x10xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[10, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ADDRESS_ARG1:%.*]] = spv._address_of [[ARG1]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0
    // CHECK: [[LOAD_ADDRESS_ARG1:%.*]] = spv.AccessChain [[ADDRESS_ARG1]]{{\[}}[[ZERO1]]{{\]}}
    // CHECK: [[INDEX1:%.*]] = spv.Load {{".*"}} [[LOAD_ADDRESS_ARG1]]
    // CHECK: [[ADDRESS_ARG0:%.*]] = spv._address_of [[ARG0]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0
    // CHECK: {{%.*}} = spv.AccessChain [[ADDRESS_ARG0]]{{\[}}[[ZERO2]], [[INDEX1]]
    %0 = iree.load_input(%arg0 : memref<5x1x10xf32>) : tensor<5x1x10xf32>
    %1 = iree.load_input(%arg1 : memref<i64>) : tensor<i64>
    %2 = "xla_hlo.gather"(%0, %1) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<[1, 1, 10]> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x10xf32>, tensor<i64>) -> tensor<1x10xf32>
    iree.store_output(%2 : tensor<1x10xf32>, %arg2 : memref<1x10xf32>)
    iree.return
  }
}
