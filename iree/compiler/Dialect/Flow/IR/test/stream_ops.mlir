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

// Tests printing and parsing of stream ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | FileCheck %s --dump-input=fail

flow.executable @dispatch_0 {
  flow.dispatch.entry @rgn_dispatch_0
  module {
    func @rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.mul %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: func @fragment
func @fragment(%arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %cst = constant dense<[4, 1, 1]> : vector<3xi32>
  // CHECK: %0:2 = flow.ex.stream.fragment(%arg1 = %cst : vector<3xi32>, %arg2 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = flow.ex.stream.fragment(%arg1 = %cst : vector<3xi32>, %arg2 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // CHECK-NEXT: flow.dispatch
    %1 = flow.dispatch @dispatch_0::@rgn_dispatch_0[%arg1 : vector<3xi32>](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: flow.return
    flow.return %1, %1 : tensor<4xf32>, tensor<4xf32>
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: return
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
