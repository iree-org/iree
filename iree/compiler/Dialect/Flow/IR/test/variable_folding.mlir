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

// Tests folding and canonicalization of variable ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt | FileCheck %s --dump-input=fail

// CHECK: flow.variable @v_initialized dense<4> : tensor<4xi32>
flow.variable @v_initialized init(@initializer) : tensor<4xi32>
func @initializer() -> tensor<4xi32> {
  %0 = constant dense<4> : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

flow.variable @v_unused : tensor<4xi32>
// CHECK-LABEL: @unused_load
func @unused_load() {
  // CHECK-NEXT: return
  %0 = flow.variable.load @v_unused : tensor<4xi32>
  return
}

// -----

flow.variable @v_nop mutable : tensor<4xi32>
// CHECK-LABEL: @nop_load_store
func @nop_load_store() {
  // CHECK-NEXT: return
  %0 = flow.variable.load @v_nop : tensor<4xi32>
  flow.variable.store @v_nop, %0 : tensor<4xi32>
  return
}

