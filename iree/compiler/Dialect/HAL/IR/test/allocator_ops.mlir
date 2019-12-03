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

// Tests printing and parsing of hal.allocator ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @allocator_compute_size
func @allocator_compute_size() -> i32 {
  %0 = "test_hal.allocator"() : () -> !ireex.ref<!hal.allocator>
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: [[SZ:%.+]] = hal.allocator.compute_size %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4
  %sz = hal.allocator.compute_size %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4
  // CHECK-NEXT: return [[SZ]]
  return %sz : i32
}

// -----

// CHECK-LABEL: @allocator_allocate
func @allocator_allocate() -> !ireex.ref<!hal.buffer> {
  // CHECK-DAG: [[C123:%.+]] = constant 123
  %0 = constant 123 : i32
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %1 = "test_hal.allocator"() : () -> !ireex.ref<!hal.allocator>
  // CHECK: [[CB:%.+]] = hal.allocator.allocate [[AL]], "HostVisible|HostCoherent", "Transfer", [[C123]] : !ireex.ref<!hal.buffer>
  %buffer = hal.allocator.allocate %1, "HostVisible|HostCoherent", "Transfer", %0 : !ireex.ref<!hal.buffer>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !ireex.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @allocator_allocate_const
func @allocator_allocate_const() -> !ireex.ref<!hal.buffer> {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %allocator = "test_hal.allocator"() : () -> !ireex.ref<!hal.allocator>
  // CHECK: [[CB:%.+]] = hal.allocator.allocate.const [[AL]], "HostVisible|HostCoherent", "Transfer" : !ireex.ref<!hal.buffer> = dense<123> : tensor<4x4xi32>
  %buffer = hal.allocator.allocate.const %allocator, "HostVisible|HostCoherent", "Transfer" : !ireex.ref<!hal.buffer> = dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !ireex.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @allocator_allocate_shaped
func @allocator_allocate_shaped() -> !ireex.ref<!hal.buffer> {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %0 = "test_hal.allocator"() : () -> !ireex.ref<!hal.allocator>
  // CHECK-DAG: {{.+}} = "test_hal.shape"
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: [[CB:%.+]] = hal.allocator.allocate.shaped [[AL]], "HostVisible|HostCoherent", "Transfer", shape=[{{.+}}#0, {{.+}}#1], element_size=4 : !ireex.ref<!hal.buffer>
  %buffer = hal.allocator.allocate.shaped %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4 : !ireex.ref<!hal.buffer>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !ireex.ref<!hal.buffer>
}
