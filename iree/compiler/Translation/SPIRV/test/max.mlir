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

// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | FileCheck %s

module {
  func @maxf(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.FMax [[VAL1:%.*]], [[VAL2:%.*]] : f32
    %2 = xla_hlo.max %0, %1 : tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @maxi(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2 : memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.SMax [[VAL1:%.*]], [[VAL2:%.*]] : i32
    %2 = xla_hlo.max %0, %1 : tensor<12x42xi32>
    iree.store_output(%2 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    iree.return
  }
}

