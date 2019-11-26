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

// Tests printing and parsing of dispatch ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | FileCheck %s --dump-input=fail

flow.executable @ex0 {
  module {
    func @dispatch_fn(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
      return %arg0 : tensor<4xf32>
    }
  }
  flow.dispatch.entry @dispatch_fn
}

// CHECK-LABEL: @dispatch
func @dispatch(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant dense<1> : vector<3xi32>
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn[%cst : vector<3xi32>](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch_fn[%cst : vector<3xi32>](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
