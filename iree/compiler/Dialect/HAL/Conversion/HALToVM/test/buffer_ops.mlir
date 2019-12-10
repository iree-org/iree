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

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan() -> !ireex.ref<!hal.buffer> {
  %0 = "test_hal.buffer"() : () -> !ireex.ref<!hal.buffer>
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: %ref = vm.call @_hal.buffer.subspan(%0, %1, %2) : (!ireex.ref<!hal.buffer>, i32, i32) -> !ireex.ref<!hal.buffer>
  %buffer = hal.buffer.subspan %0, %1, %2 : !ireex.ref<!hal.buffer>
  return %buffer : !ireex.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @buffer_fill
func @buffer_fill(%arg0 : !ireex.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.pattern"() : () -> i32
  // CHECK: vm.call @_hal.buffer.fill(%arg0, %0, %1, %2) : (!ireex.ref<!hal.buffer>, i32, i32, i32) -> ()
  hal.buffer.fill %arg0, %0, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_read_data
func @buffer_read_data(%arg0 : !ireex.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.mutable_data"() : () -> !ireex.mutable_byte_buffer_ref
  %2 = "test_hal.device_size"() : () -> i32
  %3 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @_hal.buffer.read_data(%arg0, %0, %1, %2, %3) : (!ireex.ref<!hal.buffer>, i32, !ireex.mutable_byte_buffer_ref, i32, i32) -> ()
  hal.buffer.read_data %arg0, %0, %1, %2, %3 : !ireex.mutable_byte_buffer_ref
  return
}

// -----

// CHECK-LABEL: @buffer_write_data
func @buffer_write_data(%arg0 : !ireex.ref<!hal.buffer>) {
  %0 = "test_hal.mutable_data"() : () -> !ireex.mutable_byte_buffer_ref
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  %3 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @_hal.buffer.write_data(%0, %1, %arg0, %2, %3) : (!ireex.mutable_byte_buffer_ref, i32, !ireex.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.write_data %0, %1, %arg0, %2, %3 : !ireex.mutable_byte_buffer_ref
  return
}

// -----

// CHECK-LABEL: @buffer_copy_data
func @buffer_copy_data(%arg0 : !ireex.ref<!hal.buffer>, %arg1 : !ireex.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @_hal.buffer.copy_data(%arg0, %0, %arg1, %1, %2) : (!ireex.ref<!hal.buffer>, i32, !ireex.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_load
func @buffer_load(%arg0 : !ireex.ref<!hal.buffer>) -> (i8, i16, i32) {
  %0 = "test_hal.device_size"() : () -> i32
  // CHECK: %1 = vm.call @_hal.buffer.load(%arg0, %0, %c1) : (!ireex.ref<!hal.buffer>, i32, i32) -> i32
  %1 = hal.buffer.load %arg0[%0] : i8
  // CHECK: %2 = vm.call @_hal.buffer.load(%arg0, %0, %c2) : (!ireex.ref<!hal.buffer>, i32, i32) -> i32
  %2 = hal.buffer.load %arg0[%0] : i16
  // CHECK: %3 = vm.call @_hal.buffer.load(%arg0, %0, %c4) : (!ireex.ref<!hal.buffer>, i32, i32) -> i32
  %3 = hal.buffer.load %arg0[%0] : i32
  return %1, %2, %3 : i8, i16, i32
}

// -----

// CHECK-LABEL: @buffer_store
func @buffer_store(%arg0 : !ireex.ref<!hal.buffer>, %arg1 : i8, %arg2 : i16, %arg3 : i32) {
  %0 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @_hal.buffer.store(%arg1, %arg0, %0, %c1) : (i32, !ireex.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg1, %arg0[%0] : i8
  // CHECK: vm.call @_hal.buffer.store(%arg2, %arg0, %0, %c2) : (i32, !ireex.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg2, %arg0[%0] : i16
  // CHECK: vm.call @_hal.buffer.store(%arg3, %arg0, %0, %c4) : (i32, !ireex.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg3, %arg0[%0] : i32
  return
}

// -----

// CHECK-LABEL: @buffer_view_compute_offset
func @buffer_view_compute_offset(%arg0 : !ireex.ref<!hal.buffer>) -> i32 {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  // CHECK: %2 = vm.call.variadic @_hal.buffer_view.compute_offset(%arg0, [%0#0, %0#1], [%1#0, %1#1], %c4) : (!ireex.ref<!hal.buffer>, i32..., i32..., i32) -> i32
  %off = hal.buffer_view.compute_offset %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], element_size=4
  return %off : i32
}

// -----

// CHECK-LABEL: @buffer_view_compute_length
func @buffer_view_compute_length(%arg0 : !ireex.ref<!hal.buffer>) -> i32 {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: %1 = vm.call.variadic @_hal.buffer_view.compute_length(%arg0, [%0#0, %0#1], %c4) : (!ireex.ref<!hal.buffer>, i32..., i32) -> i32
  %len = hal.buffer_view.compute_length %arg0, shape=[%0#0, %0#1], element_size=4
  return %len : i32
}

// -----

// CHECK-LABEL: @buffer_view_compute_range
func @buffer_view_compute_range(%arg0 : !ireex.ref<!hal.buffer>) -> (i32, i32) {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  %2:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %3:2 = vm.call.variadic @_hal.buffer_view.compute_range(%arg0, [%0#0, %0#1], [%1#0, %1#1], %2#0, %2#1) : (!ireex.ref<!hal.buffer>, i32..., i32..., i32, i32) -> (i32, i32)
  %off, %len = hal.buffer_view.compute_range %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1], element_size=4
  return %off, %len : i32, i32
}

// -----

// CHECK-LABEL: @buffer_view_slice
func @buffer_view_slice(%arg0 : !ireex.ref<!hal.buffer>) -> !ireex.ref<!hal.buffer> {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  %2:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %ref = vm.call.variadic @_hal.buffer_view.slice(%arg0, [%0#0, %0#1], [%1#0, %1#1], [%2#0, %2#1], %c4) : (!ireex.ref<!hal.buffer>, i32..., i32..., i32..., i32) -> !ireex.ref<!hal.buffer>
  %slice = hal.buffer_view.slice %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1], element_size=4
  return %slice : !ireex.ref<!hal.buffer>
}
