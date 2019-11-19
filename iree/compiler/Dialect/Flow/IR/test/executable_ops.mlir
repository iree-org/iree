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

// Tests printing and parsing of executable/structural ops.

// RUN: iree-opt -split-input-file %s | iree-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: @dispatch_ex
flow.executable @dispatch_ex {
  // CHECK: module {
  module {
    // CHECK: @dispatch0
    func @dispatch0() {
      return
    }
  }
  // CHECK: flow.dispatch.entry @dispatch0
  flow.dispatch.entry @dispatch0
  // CHECK: flow.dispatch.entry @dispatch0 as("dispatch0_alias")
  flow.dispatch.entry @dispatch0 as("dispatch0_alias")
}

// -----

// CHECK-LABEL: @reduction_ex
flow.executable @reduction_ex {
  // CHECK: module {
  module {
    // CHECK: @entry
    func @entry(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK: @apply
    func @apply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
      return %0 : tensor<f32>
    }
  }
  // CHECK: flow.reduction.entry @entry
  // CHECK-SAME: apply(@apply)
  // CHECK-SAME: as("entry_alias")
  // CHECK-SAME: attributes {dimension = 1 : i32}
  flow.reduction.entry @entry apply(@apply) as("entry_alias") attributes {dimension = 1 : i32}
}
