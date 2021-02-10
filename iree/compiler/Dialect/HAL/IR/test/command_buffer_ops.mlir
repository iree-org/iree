// Tests printing and parsing of hal.command_buffer ops.

// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @make_memory_barrier
func @make_memory_barrier() -> tuple<i32, i32> {
  // CHECK: %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  return %memory_barrier : tuple<i32, i32>
}

// -----

// CHECK-LABEL: @make_buffer_barrier
func @make_buffer_barrier(%arg0 : !hal.buffer) -> tuple<i32, i32, !hal.buffer, index, index> {
  %0 = "test_hal.offset"() : () -> index
  %1 = "test_hal.length"() : () -> index
  // CHECK: %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %arg0, %0, %1 : tuple<i32, i32, !hal.buffer, index, index>
  %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %arg0, %0, %1 : tuple<i32, i32, !hal.buffer, index, index>
  return %buffer_barrier : tuple<i32, i32, !hal.buffer, index, index>
}

// -----

// CHECK-LABEL: @command_buffer_create
func @command_buffer_create(%arg0 : !hal.device) {
  // CHECK: %cmd = hal.command_buffer.create %arg0, OneShot, "Transfer|Dispatch" : !hal.command_buffer
  %cmd = hal.command_buffer.create %arg0, OneShot, "Transfer|Dispatch" : !hal.command_buffer
  return
}

// -----

// CHECK-LABEL: @command_buffer_begin_end
func @command_buffer_begin_end(%arg0 : !hal.command_buffer) {
  // CHECK: hal.command_buffer.begin %arg0
  hal.command_buffer.begin %arg0
  // CHECK: hal.command_buffer.end %arg0
  hal.command_buffer.end %arg0
  return
}

// -----

// CHECK-LABEL: @command_buffer_device
func @command_buffer_device(%arg0 : !hal.command_buffer) {
  // CHECK: %0 = hal.command_buffer.device %arg0 : !hal.device
  %0 = hal.command_buffer.device %arg0 : !hal.device
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
func @command_buffer_execution_barrier(%arg0 : !hal.command_buffer) {
  // CHECK: hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess", "None"
  hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess", "None"
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer
func @command_buffer_fill_buffer(%arg0 : !hal.command_buffer) {
  %0 = "test_hal.buffer"() : () -> !hal.buffer
  %1 = "test_hal.offset"() : () -> index
  %2 = "test_hal.length"() : () -> index
  %3 = "test_hal.pattern"() : () -> i32
  // CHECK: hal.command_buffer.fill_buffer %arg0, %0, %1, %2, %3
  hal.command_buffer.fill_buffer %arg0, %0, %1, %2, %3
  return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
func @command_buffer_copy_buffer(%arg0 : !hal.command_buffer) {
  %0 = "test_hal.buffer"() : () -> !hal.buffer
  %1 = "test_hal.source_offset"() : () -> index
  %2 = "test_hal.target_offset"() : () -> index
  %3 = "test_hal.length"() : () -> index
  // CHECK: hal.command_buffer.copy_buffer %arg0, %0, %1, %0, %2, %3
  hal.command_buffer.copy_buffer %arg0, %0, %1, %0, %2, %3
  return
}

// -----

// CHECK-LABEL: @command_buffer_bind_descriptor_set
func @command_buffer_bind_descriptor_set(%arg0 : !hal.command_buffer) {
  %0 = "test_hal.executable_layout"() : () -> !hal.executable_layout
  %1 = "test_hal.descriptor_set"() : () -> !hal.descriptor_set
  %2 = "test_hal.offset"() : () -> index
  // CHECK: hal.command_buffer.bind_descriptor_set %arg0, %0, set = 0, %1
  hal.command_buffer.bind_descriptor_set %arg0, %0, set = 0, %1
  // CHECK-NEXT: hal.command_buffer.bind_descriptor_set %arg0, %0, set = 0, %1, offsets = [%2]
  hal.command_buffer.bind_descriptor_set %arg0, %0, set = 0, %1, offsets = [%2]
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch
func @command_buffer_dispatch(%arg0 : !hal.command_buffer) {
  hal.executable @ex {
    hal.executable.target @backend, filter="backend" {
      hal.executable.entry_point @entry0 attributes {
        interface = @interface, ordinal = 0 : i32, signature = (tensor<f32>) -> tensor<f32>
      }
    }
  }
  %0 = "test_hal.workgroup_x"() : () -> index
  %1 = "test_hal.workgroup_y"() : () -> index
  %2 = "test_hal.workgroup_z"() : () -> index
  // CHECK: hal.command_buffer.dispatch.symbol %arg0, @ex::@backend::@entry0, workgroup_xyz = [%0, %1, %2]
  hal.command_buffer.dispatch.symbol %arg0, @ex::@backend::@entry0, workgroup_xyz = [%0, %1, %2]
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
func @command_buffer_dispatch_indirect(%arg0 : !hal.command_buffer) {
  hal.executable @ex {
    hal.executable.target @backend, filter="backend" {
      hal.executable.entry_point @entry0 attributes {
        interface = @interface, ordinal = 0 : i32, signature = (tensor<f32>) -> tensor<f32>
      }
    }
  }
  %0 = "test_hal.buffer"() : () -> !hal.buffer
  %1 = "test_hal.offset"() : () -> index
  // CHECK: hal.command_buffer.dispatch.indirect.symbol %arg0, @ex::@backend::@entry0, workgroups = %0[%1]
  hal.command_buffer.dispatch.indirect.symbol %arg0, @ex::@backend::@entry0, workgroups = %0[%1]
  return
}
