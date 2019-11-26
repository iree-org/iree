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

// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | FileCheck %s --enable-var-scope

module {
  // CHECK: spv.globalVariable [[VAR0:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[VAR1:@.*]] bind(0, 1)
  func @extract_element(%arg0: memref<i1>, %arg1: memref<i1>)
    attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<1> : tensor<3xi32>, iree.num_dims = 3 : i32, iree.ordinal = 0 : i32} {
    %0 = "iree.load_input"(%arg0) : (memref<i1>) -> tensor<i1>
    // CHECK: [[ARG0:%.*]] = spv._address_of [[VAR0]]
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0]]{{\[}}[[ZERO]]{{\]}}
    %1 = "std.extract_element"(%0) : (tensor<i1>) -> i1
    // CHECK: [[ARG1:%.*]] = spv._address_of [[VAR1]]
    // CHECK: {{%.*}} = spv.AccessChain [[ARG1]]{{\[}}[[ZERO]]{{\]}}
    "iree.store_output"(%1, %arg1) : (i1, memref<i1>) -> ()
    "iree.return"() : () -> ()
  }
}
