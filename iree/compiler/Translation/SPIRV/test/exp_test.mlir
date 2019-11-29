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
  func @scalar_rgn_dispatch_0(%arg0: memref<f32>)
    attributes  {iree.executable.export, iree.executable.workload = dense<1> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %cst = constant dense<1.000000e+00> : tensor<f32>
    //CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %0 = "xla_hlo.exp"(%cst) : (tensor<f32>) -> tensor<f32>
    iree.store_output(%0 : tensor<f32>, %arg0 : memref<f32>)
    iree.return
  }
}

// -----

module {
  func @exp(%arg0: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %2 = "xla_hlo.exp"(%0) : (tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}
