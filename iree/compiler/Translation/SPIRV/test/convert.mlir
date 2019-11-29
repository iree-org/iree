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
  func @convert_f2f_nop(%arg0: memref<12xf32>, %arg1 : memref<12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12]> : tensor<1xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12xf32>) : tensor<12xf32>
    // CHECK-NOT: spv.FConvert
    %1 = "xla_hlo.convert"(%0) : (tensor<12xf32>) -> tensor<12xf32>
    iree.store_output(%1 : tensor<12xf32>, %arg1 : memref<12xf32>)
    iree.return
  }
}

// -----

module {
  func @convert_f2f(%arg0: memref<12xf32>, %arg1 : memref<12xf16>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12]> : tensor<1xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12xf32>) : tensor<12xf32>
    // CHECK: spv.FConvert {{%.*}} f32 to f16
    %1 = "xla_hlo.convert"(%0) : (tensor<12xf32>) -> tensor<12xf16>
    iree.store_output(%1 : tensor<12xf16>, %arg1 : memref<12xf16>)
    iree.return
  }
}

// -----

module {
  func @convert_i2i_nop(%arg0: memref<12xi32>, %arg1 : memref<12xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12]> : tensor<1xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12xi32>) : tensor<12xi32>
    // CHECK-NOT: spv.SConvert
    %1 = "xla_hlo.convert"(%0) : (tensor<12xi32>) -> tensor<12xi32>
    iree.store_output(%1 : tensor<12xi32>, %arg1 : memref<12xi32>)
    iree.return
  }
}

// -----

module {
  func @convert_i2i(%arg0: memref<12xi32>, %arg1 : memref<12xi16>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12]> : tensor<1xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12xi32>) : tensor<12xi32>
    // CHECK: spv.SConvert {{%.*}} i32 to i16
    %1 = "xla_hlo.convert"(%0) : (tensor<12xi32>) -> tensor<12xi16>
    iree.store_output(%1 : tensor<12xi16>, %arg1 : memref<12xi16>)
    iree.return
  }
}

// -----

module {
  func @convert_i2f(%arg0: memref<12xi32>, %arg1 : memref<12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12]> : tensor<1xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12xi32>) : tensor<12xi32>
    // CHECK: spv.ConvertSToF
    %1 = "xla_hlo.convert"(%0) : (tensor<12xi32>) -> tensor<12xf32>
    iree.store_output(%1 : tensor<12xf32>, %arg1 : memref<12xf32>)
    iree.return
  }
}

// -----

module {
  func @convert_f2i_nop(%arg0: memref<12xf32>, %arg1 : memref<12xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12]> : tensor<1xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12xf32>) : tensor<12xf32>
    // CHECK: spv.ConvertFToS
    %1 = "xla_hlo.convert"(%0) : (tensor<12xf32>) -> tensor<12xi32>
    iree.store_output(%1 : tensor<12xi32>, %arg1 : memref<12xi32>)
    iree.return
  }
}
