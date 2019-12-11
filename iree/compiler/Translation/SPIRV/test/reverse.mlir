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

// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK-DAG: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: func @reverse_2d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<12 x f32 [4]> [48]> [0]>, StorageBuffer>
  func @reverse_2d(%arg0: memref<12x12xf32>, %arg1 : memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 12]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: spv.selection
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[NEGATIVE_ONE:%.*]] = spv.constant -1 : i32
    // CHECK: [[NEGATIVE_IDY:%.*]] = spv.IMul [[GLOBALIDY]], [[NEGATIVE_ONE]] : i32
    // CHECK: [[ELEVEN:%.*]] = spv.constant 11 : i32
    // CHECK: [[REVERSE_IDY:%.*]] = spv.IAdd [[NEGATIVE_IDY]], [[ELEVEN]] : i32
    // CHECK: [[NEGATIVE_IDX:%.*]] = spv.IMul [[GLOBALIDX]], [[NEGATIVE_ONE]] : i32
    // CHECK: [[REVERSE_IDX:%.*]] = spv.IAdd [[NEGATIVE_IDX]], [[ELEVEN]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[ZERO]], [[REVERSE_IDY]], [[REVERSE_IDX]]{{\]}}
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    iree.store_output(%1 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK-DAG: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: func @reverse_3d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<3 x !spv.array<3 x !spv.array<3 x f32 [4]> [12]> [36]> [0]>, StorageBuffer>
  func @reverse_3d(%arg0: memref<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[3, 3, 3]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[GLOBALIDZ:%.*]] = spv.CompositeExtract [[GLOBALID]][2 : i32]
    // CHECK: spv.selection
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[NEGATIVE_ONE:%.*]] = spv.constant -1 : i32
    // CHECK: [[NEGATIVE_IDY:%.*]] = spv.IMul [[GLOBALIDY]], [[NEGATIVE_ONE]] : i32
    // CHECK: [[TWO:%.*]] = spv.constant 2 : i32
    // CHECK: [[REVERSE_IDY:%.*]] = spv.IAdd [[NEGATIVE_IDY]], [[TWO]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[ZERO]], [[GLOBALIDZ]], [[REVERSE_IDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<3x3x3xf32>) : tensor<3x3x3xf32>
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    iree.store_output(%1 : tensor<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
    iree.return
  }
}
