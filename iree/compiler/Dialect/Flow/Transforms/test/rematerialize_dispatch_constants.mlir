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

// RUN: iree-opt -split-input-file -iree-flow-rematerialize-dispatch-constants %s | FileCheck %s --enable-var-scope --dump-input=always

// CHECK-LABEL: func @rematerializeSmall
func @rematerializeSmall(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = constant dense<[4, 4, 1]> : vector<3xi32>
  %small = constant dense<1.23> : tensor<4x4xf32>
  // CHECK: %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4x4xf32>, %arg2 = %small : tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-NEXT: %cst_0 = constant dense<1.230000e+00> : tensor<4x4xf32>
    // CHECK-NEXT: %1 = xla_hlo.add %arg1, %cst_0 : tensor<4x4xf32>
    %3 = xla_hlo.add %arg1, %arg2 : tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @noRematerializeLarge
func @noRematerializeLarge(%arg0 : tensor<4096x4xf32>) -> tensor<4096x4xf32> {
  %cst = constant dense<[4, 4, 1]> : vector<3xi32>
  // CHECK: %cst_0 = constant dense<1.230000e+00> : tensor<4096x4xf32>
  %large = constant dense<1.23> : tensor<4096x4xf32>
  // CHECK-NEXT: %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4096x4xf32>, %arg2 = %cst_0 : tensor<4096x4xf32>) -> tensor<4096x4xf32> {
  %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4096x4xf32>, %arg2 = %large : tensor<4096x4xf32>) -> tensor<4096x4xf32> {
    // CHECK-NEXT: %1 = xla_hlo.add %arg1, %arg2 : tensor<4096x4xf32>
    %3 = xla_hlo.add %arg1, %arg2 : tensor<4096x4xf32>
    flow.return %3 : tensor<4096x4xf32>
  }
  return %0 : tensor<4096x4xf32>
}

// -----

// CHECK-LABEL: func @noRematerializeIntoDot
func @noRematerializeIntoDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = constant dense<[4, 4, 1]> : vector<3xi32>
  // CHECK: %cst_0 = constant dense<1.230000e+00> : tensor<4x4xf32>
  %small = constant dense<1.23> : tensor<4x4xf32>
  // CHECK-NEXT: %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4x4xf32>, %arg2 = %cst_0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = flow.dispatch.region[%cst : vector<3xi32>](%arg1 = %arg0 : tensor<4x4xf32>, %arg2 = %small : tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-NEXT: %1 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    flow.return %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}
