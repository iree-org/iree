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

// RUN: iree-opt -split-input-file -iree-convert-flow-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: hal.variable @var_i32 mutable : !ireex.ref<!hal.buffer>
flow.variable @var_i32 mutable : tensor<i32>
func @fn() {
  // CHECK: [[V:%.+]] = hal.variable.load @var_i32 : !ireex.ref<!hal.buffer>
  %0 = flow.variable.load @var_i32 : tensor<i32>
  // CHECK-NEXT: hal.variable.store [[V]], @var_i32 : !ireex.ref<!hal.buffer>
  flow.variable.store %0, @var_i32 : tensor<i32>
  return
}

// -----

// CHECK-LABEL: hal.variable @var_i1 mutable : !ireex.ref<!hal.buffer>
flow.variable @var_i1 mutable : tensor<i1>
func @fn() {
  // CHECK: [[V:%.+]] = hal.variable.load @var_i1 : !ireex.ref<!hal.buffer>
  %0 = flow.variable.load @var_i1 : tensor<i1>
  // CHECK-NEXT: hal.variable.store [[V]], @var_i1 : !ireex.ref<!hal.buffer>
  flow.variable.store %0, @var_i1 : tensor<i1>
  return
}
