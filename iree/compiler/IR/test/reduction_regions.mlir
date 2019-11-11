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

// RUN: iree-opt %s -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: @singleReduction
func @singleReduction(%arg0 : tensor<5x1xf32>) {
  // CHECK: %0 = "some.shape"(%arg0) : (tensor<5x1xf32>) -> tensor<1xi32>
  %workload = "some.shape"(%arg0) : (tensor<5x1xf32>) -> tensor<1xi32>
  // CHECK: %1 = "some.constant"() : () -> tensor<f32>
  %initialValueF = "some.constant"() : () -> tensor<f32>
  // CHECK: %2 = iree.reduction_region[%0 : tensor<1xi32>](%arg0) : (tensor<5x1xf32>) -> (tensor<1xf32>)
  // CHECK-NEXT:     invocation((%arg1, %arg2) = %1 : tensor<f32>) {
  // CHECK-NEXT:   %3 = "my.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:   iree.return %3 : tensor<f32>
  // CHECK-NEXT: } {dimensions = dense<[1, 2]> : tensor<2xi64>}
  %ret = iree.reduction_region[%workload : tensor<1xi32>](%arg0) : (tensor<5x1xf32>) -> (tensor<1xf32>)
      invocation((%i0, %i1) = %initialValueF : tensor<f32>) {
    %resultF = "my.add"(%i0, %i1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    iree.return %resultF : tensor<f32>
  } {dimensions = dense<[1, 2]> : tensor<2xi64>}
  return
}

// -----

// CHECK-LABEL: @fusedReduction
func @fusedReduction(%arg0 : tensor<5x1xf32>, %arg1 : tensor<5x1xi32>) {
  // CHECK: %0 = "some.shape"(%arg0) : (tensor<5x1xf32>) -> tensor<1xi32>
  %workload = "some.shape"(%arg0) : (tensor<5x1xf32>) -> tensor<1xi32>
  // CHECK: %1 = "some.constant"() : () -> tensor<f32>
  // CHECK: %2 = "some.constant"() : () -> tensor<i32>
  %initialValueF = "some.constant"() : () -> tensor<f32>
  %initialValueI = "some.constant"() : () -> tensor<i32>
  // CHECK: %3:2 = iree.reduction_region[%0 : tensor<1xi32>](%arg0, %arg1) : (tensor<5x1xf32>, tensor<5x1xi32>) -> (tensor<1xf32>, tensor<1xi32>)
  // CHECK-NEXT:     invocation((%arg2, %arg3) = %1 : tensor<f32>, (%arg4, %arg5) = %2 : tensor<i32>) {
  // CHECK-NEXT:   %4 = "my.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:   %5 = "my.add"(%arg4, %arg5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT:   iree.return %4, %5 : tensor<f32>, tensor<i32>
  // CHECK-NEXT: } {dimensions = dense<[1, 2]> : tensor<2xi64>}
  %ret:2 = iree.reduction_region[%workload : tensor<1xi32>](%arg0, %arg1) : (tensor<5x1xf32>, tensor<5x1xi32>) -> (tensor<1xf32>, tensor<1xi32>)
      invocation((%i0, %i1) = %initialValueF : tensor<f32>,
                 (%i2, %i3) = %initialValueI : tensor<i32>) {
    %resultF = "my.add"(%i0, %i1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %resultI = "my.add"(%i2, %i3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    iree.return %resultF, %resultI : tensor<f32>, tensor<i32>
  } {dimensions = dense<[1, 2]> : tensor<2xi64>}
  return
}
