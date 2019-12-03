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

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK: hal.variable @v_initialized 4 : i32
hal.variable @v_initialized init(@initializer) : i32
func @initializer() -> i32 {
  %0 = constant 4 : i32
  return %0 : i32
}

// -----

hal.variable @v_unused : !ireex.ref<!hal.buffer>
// CHECK-LABEL: @unused_load
func @unused_load() {
  // CHECK-NEXT: return
  %0 = hal.variable.load @v_unused : !ireex.ref<!hal.buffer>
  return
}

// -----

hal.variable @v_nop mutable : !ireex.ref<!hal.buffer>
// CHECK-LABEL: @nop_load_store
func @nop_load_store() {
  // CHECK-NEXT: return
  %0 = hal.variable.load @v_nop : !ireex.ref<!hal.buffer>
  hal.variable.store %0, @v_nop : !ireex.ref<!hal.buffer>
  return
}

