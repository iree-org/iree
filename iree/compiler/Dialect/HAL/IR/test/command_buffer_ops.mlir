// Tests printing and parsing of hal.command_buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @make_memory_barrier
func @make_memory_barrier() -> tuple<i32, i32> {
  // CHECK: %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  return %memory_barrier : tuple<i32, i32>
}

// -----

// CHECK-LABEL: @make_buffer_barrier
func @make_buffer_barrier(%arg0 : !iree.ref<!hal.buffer>) -> tuple<i32, i32, !iree.ref<!hal.buffer>, i32, i32> {
  %0 = "test_hal.offset"() : () -> i32
  %1 = "test_hal.length"() : () -> i32
  // CHECK: %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %arg0, %0, %1 : tuple<i32, i32, !iree.ref<!hal.buffer>, i32, i32>
  %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %arg0, %0, %1 : tuple<i32, i32, !iree.ref<!hal.buffer>, i32, i32>
  return %buffer_barrier : tuple<i32, i32, !iree.ref<!hal.buffer>, i32, i32>
}

// -----

// CHECK-LABEL: @command_buffer_create
func @command_buffer_create(%arg0 : !iree.ref<!hal.device>) {
  // CHECK: %cmd = hal.command_buffer.create %arg0, "OneShot", "Transfer|Dispatch" : !iree.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create %arg0, "OneShot", "Transfer|Dispatch" : !iree.ref<!hal.command_buffer>
  return
}

// -----

// CHECK-LABEL: @command_buffer_begin_end
func @command_buffer_begin_end(%arg0 : !iree.ref<!hal.command_buffer>) {
  // CHECK: hal.command_buffer.begin %arg0
  hal.command_buffer.begin %arg0
  // CHECK: hal.command_buffer.end %arg0
  hal.command_buffer.end %arg0
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
func @command_buffer_execution_barrier(%arg0 : !iree.ref<!hal.command_buffer>) {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %0, %1, %2 : tuple<i32, i32, !iree.ref<!hal.buffer>, i32, i32>
  // CHECK: hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess"
  hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess"
  // CHECK-NEXT: hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess", memory_barriers=[%memory_barrier, %memory_barrier]
  hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess",
      memory_barriers=[%memory_barrier, %memory_barrier]
  // CHECK-NEXT: hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess", buffer_barriers=[%buffer_barrier, %buffer_barrier]
  hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess",
      buffer_barriers=[%buffer_barrier, %buffer_barrier]
  // CHECK-NEXT: hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess", memory_barriers=[%memory_barrier, %memory_barrier], buffer_barriers=[%buffer_barrier, %buffer_barrier]
  hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess",
      memory_barriers=[%memory_barrier, %memory_barrier],
      buffer_barriers=[%buffer_barrier, %buffer_barrier]
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer
func @command_buffer_fill_buffer(%arg0 : !iree.ref<!hal.command_buffer>) {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  %3 = "test_hal.pattern"() : () -> i32
  // CHECK: hal.command_buffer.fill_buffer %arg0, %0, %1, %2, %3
  hal.command_buffer.fill_buffer %arg0, %0, %1, %2, %3
  return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
func @command_buffer_copy_buffer(%arg0 : !iree.ref<!hal.command_buffer>) {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %1 = "test_hal.source_offset"() : () -> i32
  %2 = "test_hal.target_offset"() : () -> i32
  %3 = "test_hal.length"() : () -> i32
  // CHECK: hal.command_buffer.copy_buffer %arg0, %0, %1, %0, %2, %3
  hal.command_buffer.copy_buffer %arg0, %0, %1, %0, %2, %3
  return
}

// -----

// CHECK-LABEL: @command_buffer_bind_descriptor_set
func @command_buffer_bind_descriptor_set(%arg0 : !iree.ref<!hal.command_buffer>) {
  %0 = "test_hal.executable"() : () -> !iree.ref<!hal.executable>
  %1 = "test_hal.descriptor_set"() : () -> !iree.ref<!hal.descriptor_set>
  %2 = "test_hal.offset"() : () -> i32
  // CHECK: hal.command_buffer.bind_descriptor_set %arg0, %0, set=0, %1
  hal.command_buffer.bind_descriptor_set %arg0, %0, set=0, %1
  // CHECK-NEXT: hal.command_buffer.bind_descriptor_set %arg0, %0, set=0, %1, offsets=[%2]
  hal.command_buffer.bind_descriptor_set %arg0, %0, set=0, %1, offsets=[%2]
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch
func @command_buffer_dispatch(%arg0 : !iree.ref<!hal.command_buffer>) {
  %0 = "test_hal.executable"() : () -> !iree.ref<!hal.executable>
  %1 = "test_hal.workgroup_x"() : () -> i32
  %2 = "test_hal.workgroup_y"() : () -> i32
  %3 = "test_hal.workgroup_z"() : () -> i32
  // CHECK: hal.command_buffer.dispatch %arg0, %0, entry_point=0, workgroup_xyz=[%1, %2, %3]
  hal.command_buffer.dispatch %arg0, %0, entry_point=0, workgroup_xyz=[%1, %2, %3]
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
func @command_buffer_dispatch_indirect(%arg0 : !iree.ref<!hal.command_buffer>) {
  %0 = "test_hal.executable"() : () -> !iree.ref<!hal.executable>
  %1 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %2 = "test_hal.offset"() : () -> i32
  // CHECK: hal.command_buffer.dispatch.indirect %arg0, %0, entry_point=0, workgroups=%1[%2]
  hal.command_buffer.dispatch.indirect %arg0, %0, entry_point=0, workgroups=%1[%2]
  return
}
