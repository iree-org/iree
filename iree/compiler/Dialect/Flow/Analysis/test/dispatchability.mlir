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

// RUN: iree-opt -split-input-file -test-iree-flow-dispatchability %s | IreeFileCheck %s

// CHECK-LABEL: @empty
// CHECK-SAME: dispatchable = true
func @empty() {
  return
}

// -----

// CHECK-LABEL: @simpleMath
// CHECK-SAME: dispatchable = true
func @simpleMath(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @stdElementwiseOps
// CHECK-SAME: dispatchable = true
func @stdElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  %1 = subf %0, %arg0 : tensor<4xf32>
  %2 = mulf %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @hloElementwiseOps
// CHECK-SAME: dispatchable = true
func @hloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = xla_hlo.sub %0, %arg0 : tensor<4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @interleavedDot
// CHECK-SAME: dispatchable = false
func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4x4xf32>
  %1 = "xla_hlo.dot"(%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @caller
// CHECK-SAME: dispatchable = true
func @caller(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
// CHECK-LABEL: func @callee
// CHECK-SAME: dispatchable = true
func @callee(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.mul %arg0, %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @dotCaller
// CHECK-SAME: dispatchable = false
func @dotCaller(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4x4xf32>
  %1 = call @dotCallee(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
// CHECK-LABEL: func @dotCallee
// CHECK-SAME: dispatchable = false
func @dotCallee(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "xla_hlo.dot"(%arg0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
