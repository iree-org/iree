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

// Tests printing and parsing of hal.command_buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: @make_memory_barrier
func @make_memory_barrier() -> tuple<i32, i32> {
  // CHECK: %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  return %memory_barrier : tuple<i32, i32>
}

// -----

// CHECK-LABEL: @make_buffer_barrier
func @make_buffer_barrier(%arg0 : !ireex.ref<!hal.buffer>) -> tuple<i32, i32, !ireex.ref<!hal.buffer>, i32, i32> {
  %0 = "test_hal.offset"() : () -> i32
  %1 = "test_hal.length"() : () -> i32
  // CHECK: %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %arg0, %0, %1 : tuple<i32, i32, !ireex.ref<!hal.buffer>, i32, i32>
  %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %arg0, %0, %1 : tuple<i32, i32, !ireex.ref<!hal.buffer>, i32, i32>
  return %buffer_barrier : tuple<i32, i32, !ireex.ref<!hal.buffer>, i32, i32>
}

// -----

// CHECK-LABEL: @make_buffer_binding
func @make_buffer_binding(%arg0 : !ireex.ref<!hal.buffer>) -> tuple<i32, !ireex.ref<!hal.buffer>, i32, i32> {
  %0 = "test_hal.offset"() : () -> i32
  %1 = "test_hal.length"() : () -> i32
  // CHECK: %buffer_binding = hal.make_buffer_binding "Read|Write", %arg0, %0, %1 : tuple<i32, !ireex.ref<!hal.buffer>, i32, i32>
  %buffer_binding = hal.make_buffer_binding "Read|Write", %arg0, %0, %1 : tuple<i32, !ireex.ref<!hal.buffer>, i32, i32>
  return %buffer_binding : tuple<i32, !ireex.ref<!hal.buffer>, i32, i32>
}

// -----

// CHECK-LABEL: @command_buffer_create
func @command_buffer_create(%arg0 : !ireex.ref<!hal.device>) {
  // CHECK: %cmd = hal.command_buffer.create %arg0, "OneShot", "Transfer|Dispatch" : !ireex.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create %arg0, "OneShot", "Transfer|Dispatch" : !ireex.ref<!hal.command_buffer>
  return
}

// -----

// CHECK-LABEL: @command_buffer_begin_end
func @command_buffer_begin_end(%arg0 : !ireex.ref<!hal.command_buffer>) {
  // CHECK: hal.command_buffer.begin %arg0
  hal.command_buffer.begin %arg0
  // CHECK: hal.command_buffer.end %arg0
  hal.command_buffer.end %arg0
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
func @command_buffer_execution_barrier(%arg0 : !ireex.ref<!hal.command_buffer>) {
  %0 = "test_hal.barrier_list"() : () -> tuple<>
  // CHECK: hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess", memory_barriers=%0 : tuple<>, buffer_barriers=%0 : tuple<>
  hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess", memory_barriers=%0 : tuple<>, buffer_barriers=%0 : tuple<>
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer
func @command_buffer_fill_buffer(%arg0 : !ireex.ref<!hal.command_buffer>) {
  %0 = "test_hal.buffer"() : () -> !ireex.ref<!hal.buffer>
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  %3 = "test_hal.pattern"() : () -> i32
  // CHECK: hal.command_buffer.fill_buffer %arg0, %0, %1, %2, %3
  hal.command_buffer.fill_buffer %arg0, %0, %1, %2, %3
  return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
func @command_buffer_copy_buffer(%arg0 : !ireex.ref<!hal.command_buffer>) {
  %0 = "test_hal.buffer"() : () -> !ireex.ref<!hal.buffer>
  %1 = "test_hal.source_offset"() : () -> i32
  %2 = "test_hal.target_offset"() : () -> i32
  %3 = "test_hal.length"() : () -> i32
  // CHECK: hal.command_buffer.copy_buffer %arg0, %0, %1, %0, %2, %3
  hal.command_buffer.copy_buffer %arg0, %0, %1, %0, %2, %3
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch
func @command_buffer_dispatch(%arg0 : !ireex.ref<!hal.command_buffer>) {
  %0 = "test_hal.executable"() : () -> !ireex.ref<!hal.executable>
  %1 = "test_hal.workgroups"() : () -> vector<3xi32>
  %2 = "test_hal.binding"() : () -> tuple<i32, !ireex.ref<!hal.buffer>, i32, i32>
  // CHECK: hal.command_buffer.dispatch %arg0, %0, @entry[%1](%2, %2, %2)
  hal.command_buffer.dispatch %arg0, %0, @entry[%1](%2, %2, %2)
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
func @command_buffer_dispatch_indirect(%arg0 : !ireex.ref<!hal.command_buffer>) {
  %0 = "test_hal.executable"() : () -> !ireex.ref<!hal.executable>
  %1 = "test_hal.workgroups"() : () -> tuple<i32, !ireex.ref<!hal.buffer>, i32, i32>
  %2 = "test_hal.binding"() : () -> tuple<i32, !ireex.ref<!hal.buffer>, i32, i32>
  // CHECK: hal.command_buffer.dispatch.indirect %arg0, %0, @entry[%1](%2, %2, %2)
  hal.command_buffer.dispatch.indirect %arg0, %0, @entry[%1](%2, %2, %2)
  return
}
