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
  %1 = "test_hal.shape"() : () -> vector<2xi32>
  // CHECK: [[SZ:%.+]] = hal.allocator.compute_size %0, "HostVisible|HostCoherent", "Transfer", %1, 4 : vector<2xi32>
  %sz = hal.allocator.compute_size %0, "HostLocal", "Transfer", %1, 4 : vector<2xi32>
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
  %buffer = hal.allocator.allocate %1, "HostLocal", "Transfer", %0 : !ireex.ref<!hal.buffer>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !ireex.ref<!hal.buffer>
}
