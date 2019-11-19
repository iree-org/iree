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

// RUN: iree-opt -split-input-file -iree-flow-dispatchability-analysis -iree-flow-identify-dispatch-regions %s | FileCheck %s --dump-input=always

// CHECK-LABEL: @empty
func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @simpleMath
func @simpleMath(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: constant dense<[4, 1, 1]>
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%cst : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) : tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @stdElementwiseOps
func @stdElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: constant dense<[4, 1, 1]>
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%cst : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) : tensor<4xf32> {
  // CHECK-NEXT:   %1 = addf %arg1, %arg1 : tensor<4xf32>
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = subf %1, %arg1 : tensor<4xf32>
  %1 = subf %0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %3 = mulf %2, %arg1 : tensor<4xf32>
  %2 = mulf %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @hloElementwiseOps
func @hloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: constant dense<[4, 1, 1]>
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%cst : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) : tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = xla_hlo.sub %1, %arg1 : tensor<4xf32>
  %1 = xla_hlo.sub %0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %3 = xla_hlo.mul %2, %arg1 : tensor<4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @interleavedDot
func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %cst = constant dense<[4, 4, 1]>
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%cst : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4x4xf32>) : tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = xla_hlo.add %arg1, %arg1 : tensor<4x4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4x4xf32>
  // CHECK-NEXT: flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %cst_0 = constant dense<[4, 4, 1]> : tensor<3xi32>
  // CHECK-NEXT: %1 = flow.dispatch.region
  // CHECK-SAME: [%cst_0 : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %0 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) : tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "xla_hlo.dot"(%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %cst_1 = constant dense<[4, 4, 1]> : tensor<3xi32>
  // CHECK-NEXT: %2 = flow.dispatch.region
  // CHECK-SAME: [%cst_1 : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %1 : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) : tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = xla_hlo.mul %arg1, %arg2 : tensor<4x4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %2 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @caller
func @caller(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: constant dense<[4, 1, 1]>
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%cst : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) : tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = call @callee(%1) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %3 = xla_hlo.mul %2, %arg1 : tensor<4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
// CHECK-LABEL: func @callee
func @callee(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: constant dense<[4, 1, 1]>
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%cst : tensor<3xi32>]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) : tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.mul %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.mul %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
