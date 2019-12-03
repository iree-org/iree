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

// Tests folding and canonicalization of HAL allocator ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @simplify_allocate_shaped
func @simplify_allocate_shapedy() -> !ireex.ref<!hal.buffer> {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %0 = "test_hal.allocator"() : () -> !ireex.ref<!hal.allocator>
  // CHECK-DAG: [[SH:%.+]]:2 = "test_hal.shape"
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK-NEXT: %buffer = hal.allocator.allocate_shaped [[AL]], "HostVisible|HostCoherent", "Transfer", shape=[
// CHECK-SAME:     [[SH]]#0, [[SH]]#1
// CHECK-SAME: ], element_size=4 : !ireex.ref<!hal.buffer>
  %sz = hal.allocator.compute_size %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4
  %buffer = hal.allocator.allocate %0, "HostVisible|HostCoherent", "Transfer", %sz : !ireex.ref<!hal.buffer>
  // CHECK-NEXT: return %buffer
  return %buffer : !ireex.ref<!hal.buffer>
}
