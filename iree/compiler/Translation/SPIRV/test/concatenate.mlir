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
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @concatenate(%arg0: memref<1x64xf32>, %arg1 : memref<1x10xf32>, %arg2 : memref<1x74xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[1, 74]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[ZERO2]], [[GLOBALIDY]]{{\]}}
    // CHECK: [[INPUTVAL0:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO3:%.*]] = spv.constant 0 : i32
    // CHECK: [[NEGATIVE_SIXTY_FOUR:%.*]] = spv.constant -64 : i32
    // CHECK: [[VAR1:%.*]] = spv.IAdd [[GLOBALIDY]], [[NEGATIVE_SIXTY_FOUR]] : i32
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO3]], [[ZERO2]], [[VAR1]]{{\]}}
    // CHECK: [[INPUTVAL1:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]] : f32
    // CHECK: [[TRUE:%.*]] = spv.constant true
    // CHECK: [[SIXTY_FOUR:%.*]] = spv.constant 64 : i32
    // CHECK: [[CHECK:%.*]] = spv.SGreaterThanEqual [[GLOBALIDY]], [[SIXTY_FOUR]] : i32
    // CHECK: [[COND:%.*]] = spv.LogicalAnd [[TRUE]], [[CHECK]] : i1
    // CHECK: [[RESULT:%.*]] = spv.Select [[COND]], [[INPUTVAL1]], [[INPUTVAL0]] : i1, f32
    %0 = iree.load_input(%arg0 : memref<1x64xf32>) : tensor<1x64xf32>
    %1 = iree.load_input(%arg1 : memref<1x10xf32>) : tensor<1x10xf32>
    %2 = "xla_hlo.concatenate"(%0, %1) {dimension = 1 : i64} : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
    iree.store_output(%2 : tensor<1x74xf32>, %arg2 : memref<1x74xf32>)
    iree.return
  }
}
