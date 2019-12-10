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

// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @allocatorComputeSize
func @allocatorComputeSize(%arg0 : !ireex.ref<!hal.allocator>) -> i32 {
  %c1024_i32 = constant 1024 : i32
  // CHECK: %0 = vm.call.variadic @_hal.allocator.compute_size(%arg0, %c6, %c15, [%c1024, %c1024], %c4) : (!ireex.ref<!hal.allocator>, i32, i32, i32..., i32) -> i32
  %0 = hal.allocator.compute_size %arg0, "HostLocal", "All", shape=[%c1024_i32, %c1024_i32], element_size=4
  return %0 : i32
}

// -----

// CHECK-LABEL: @allocatorAllocate
func @allocatorAllocate(%arg0 : !ireex.ref<!hal.allocator>) -> !ireex.ref<!hal.buffer> {
  %c1024_i32 = constant 1024 : i32
  // CHECK: %ref = vm.call @_hal.allocator.allocate(%arg0, %c6, %c15, %c1024) : (!ireex.ref<!hal.allocator>, i32, i32, i32) -> !ireex.ref<!hal.buffer>
  %0 = hal.allocator.allocate %arg0, "HostLocal", "All", %c1024_i32 : !ireex.ref<!hal.buffer>
  return %0 : !ireex.ref<!hal.buffer>
}

// -----

// CHECK: vm.rodata @allocatorAllocateConst_const_0 dense<123> : tensor<4x4xi32>
// CHECK-LABEL: func @allocatorAllocateConst
func @allocatorAllocateConst() -> !ireex.ref<!hal.buffer> {
  %allocator = "test_hal.allocator"() : () -> !ireex.ref<!hal.allocator>
  // CHECK:  %allocatorAllocateConst_const_0 = vm.const.ref.rodata @allocatorAllocateConst_const_0 : !ireex.byte_buffer_ref
  // CHECK: %ref = vm.call.variadic @_hal.allocator.allocate.const(%0, %c6, %c2, [%c4, %c4_0], %c4_1, %allocatorAllocateConst_const_0) : (!ireex.ref<!hal.allocator>, i32, i32, i32..., i32, !ireex.byte_buffer_ref) -> !ireex.ref<!hal.buffer>
  %buffer = hal.allocator.allocate.const %allocator, "HostVisible|HostCoherent", "Transfer" : !ireex.ref<!hal.buffer> = dense<123> : tensor<4x4xi32>
  return %buffer : !ireex.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @allocatorAllocateShaped
func @allocatorAllocateShaped(%arg0 : !ireex.ref<!hal.allocator>) -> !ireex.ref<!hal.buffer> {
  %c1024_i32 = constant 1024 : i32
  // CHECK: %ref = vm.call.variadic @_hal.allocator.allocate.shaped(%arg0, %c6, %c15, [%c1024, %c1024], %c4) : (!ireex.ref<!hal.allocator>, i32, i32, i32..., i32) -> !ireex.ref<!hal.buffer>
  %0 = hal.allocator.allocate.shaped %arg0, "HostLocal", "All", shape=[%c1024_i32, %c1024_i32], element_size=4 : !ireex.ref<!hal.buffer>
  return %0 : !ireex.ref<!hal.buffer>
}
