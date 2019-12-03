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

// Tests printing and parsing of hal.buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan() -> !ireex.ref<!hal.buffer> {
  %0 = "test_hal.buffer"() : () -> !ireex.ref<!hal.buffer>
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: %buffer = hal.buffer.subspan %0, %1, %2 : !ireex.ref<!hal.buffer>
  %buffer = hal.buffer.subspan %0, %1, %2 : !ireex.ref<!hal.buffer>
  return %buffer : !ireex.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @buffer_fill
func @buffer_fill(%arg0 : !ireex.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.pattern"() : () -> i32
  // CHECK: hal.buffer.fill %arg0, %0, %1, %2
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
  // CHECK: hal.buffer.read_data %arg0, %0, %1, %2, %3 : !ireex.mutable_byte_buffer_ref
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
  // CHECK: hal.buffer.write_data %0, %1, %arg0, %2, %3 : !ireex.mutable_byte_buffer_ref
  hal.buffer.write_data %0, %1, %arg0, %2, %3 : !ireex.mutable_byte_buffer_ref
  return
}

// -----

// CHECK-LABEL: @buffer_copy_data
func @buffer_copy_data(%arg0 : !ireex.ref<!hal.buffer>, %arg1 : !ireex.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_view_compute_offset
func @buffer_view_compute_offset(%arg0 : !ireex.ref<!hal.buffer>) -> i32 {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  // CHECK: %off = hal.buffer_view.compute_offset %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1]
  %off = hal.buffer_view.compute_offset %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1]
  return %off : i32
}

// -----

// CHECK-LABEL: @buffer_view_compute_range
func @buffer_view_compute_range(%arg0 : !ireex.ref<!hal.buffer>) -> (i32, i32) {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  %2:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %off, %len = hal.buffer_view.compute_range %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1]
  %off, %len = hal.buffer_view.compute_range %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1]
  return %off, %len : i32, i32
}

// -----

// CHECK-LABEL: @buffer_view_slice
func @buffer_view_slice(%arg0 : !ireex.ref<!hal.buffer>) -> !ireex.ref<!hal.buffer> {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  %2:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %slice = hal.buffer_view.slice %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1]
  %slice = hal.buffer_view.slice %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1]
  return %slice : !ireex.ref<!hal.buffer>
}
