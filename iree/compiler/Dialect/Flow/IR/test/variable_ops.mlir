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

// Tests printing and parsing of variable ops.

// RUN: iree-opt -split-input-file %s | iree-opt | FileCheck %s --dump-input=fail

// CHECK: flow.variable @v_immutable : tensor<i32>
flow.variable @v_immutable : tensor<i32>
// CHECK: flow.variable @v_mutable mutable : tensor<i32>
flow.variable @v_mutable mutable : tensor<i32>

// -----

// CHECK: flow.variable @v_initialized_const dense<4> : tensor<4xi32>
flow.variable @v_initialized_const dense<4> : tensor<4xi32>

// -----

// CHECK: flow.variable @v_initialized init(@initializer) : tensor<4xi32>
flow.variable @v_initialized init(@initializer) : tensor<4xi32>
func @initializer() -> tensor<4xi32>

// -----

flow.variable @v_loaded : tensor<4xi32>
// CHECK-LABEL: @loaded
func @loaded() {
  // CHECK-NEXT: %0 = flow.variable.load @v_loaded : tensor<4xi32>
  %0 = flow.variable.load @v_loaded : tensor<4xi32>
  return
}

// -----

flow.variable @v_stored mutable : tensor<4xi32>
// CHECK-LABEL: @stored
func @stored() {
  // CHECK-NEXT: = constant
  %cst = constant dense<5> : tensor<4xi32>
  // CHECK-NEXT: flow.variable.store @v_stored, %cst : tensor<4xi32>
  flow.variable.store @v_stored, %cst : tensor<4xi32>
  return
}
