// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @command_buffer_create
func @command_buffer_create(%arg0: !hal.device) {
  // CHECK: %ref = vm.call @hal.command_buffer.create(%arg0, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create device(%arg0 : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  return
}

// -----

// CHECK-LABEL: @command_buffer_begin_end
func @command_buffer_begin_end(%arg0: !hal.command_buffer) {
  // CHECK: vm.call @hal.command_buffer.begin(%arg0) : (!vm.ref<!hal.command_buffer>) -> ()
  hal.command_buffer.begin<%arg0 : !hal.command_buffer>
  // CHECK: vm.call @hal.command_buffer.end(%arg0) : (!vm.ref<!hal.command_buffer>) -> ()
  hal.command_buffer.end<%arg0 : !hal.command_buffer>
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
func @command_buffer_execution_barrier(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer
) {
  // CHECK: vm.call @hal.command_buffer.execution_barrier(%arg0, %c1, %c2, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32)
  hal.command_buffer.execution_barrier<%arg0 : !hal.command_buffer>
      source("CommandIssue")
      target("CommandProcess")
      flags("None")
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer
func @command_buffer_fill_buffer(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer
) {
  %c100 = constant 100 : index
  %c200 = constant 200 : index
  %c300 = constant 300 : i32
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %c100, %c200, %c300) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.buffer>, i32, i32, i32) -> ()
  hal.command_buffer.fill_buffer<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.buffer)[%c100, %c200]
      pattern(%c300 : i32)
  return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
func @command_buffer_copy_buffer(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer
) {
  %c100 = constant 100 : index
  %c200 = constant 200 : index
  %c300 = constant 300 : index
  // CHECK: vm.call @hal.command_buffer.copy_buffer(%arg0, %arg1, %c100, %arg1, %c200, %c300) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.buffer>, i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.command_buffer.copy_buffer<%arg0 : !hal.command_buffer>
      source(%arg1 : !hal.buffer)[%c100]
      target(%arg1 : !hal.buffer)[%c200]
      length(%c300)
  return
}

// -----

// CHECK-LABEL: @command_buffer_bind_descriptor_set
func @command_buffer_bind_descriptor_set(
  %arg0: !hal.command_buffer,
  %arg1: !hal.executable_layout,
  %arg2: !hal.descriptor_set
) {
  %c0 = constant 0 : index
  %c100 = constant 100 : index
  // CHECK: vm.call.variadic @hal.command_buffer.bind_descriptor_set(%arg0, %arg1, %zero, %arg2, []) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, !vm.ref<!hal.descriptor_set>, i32 ...)
  hal.command_buffer.bind_descriptor_set<%arg0 : !hal.command_buffer>
      layout(%arg1 : !hal.executable_layout)[%c0]
      set(%arg2 : !hal.descriptor_set)
  // CHECK: vm.call.variadic @hal.command_buffer.bind_descriptor_set(%arg0, %arg1, %zero, %arg2, [%c100]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable_layout>, i32, !vm.ref<!hal.descriptor_set>, i32 ...)
  hal.command_buffer.bind_descriptor_set<%arg0 : !hal.command_buffer>
      layout(%arg1 : !hal.executable_layout)[%c0]
      set(%arg2 : !hal.descriptor_set)
      offsets([%c100])
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch
func @command_buffer_dispatch(
  %arg0: !hal.command_buffer,
  %arg1: !hal.executable
) {
  %c100 = constant 100 : index
  %c200 = constant 200 : index
  %c300 = constant 300 : index
  // CHECK: vm.call @hal.command_buffer.dispatch(%arg0, %arg1, %zero, %c100, %c200, %c300) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
  hal.command_buffer.dispatch<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.executable)[0]
      workgroups([%c100, %c200, %c300])
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
func @command_buffer_dispatch_indirect(
  %arg0: !hal.command_buffer,
  %arg1: !hal.executable,
  %arg2: !hal.buffer
) {
  %c100 = constant 100 : index
  // CHECK: vm.call @hal.command_buffer.dispatch.indirect(%arg0, %arg1, %zero, %arg2, %c100) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, !vm.ref<!hal.buffer>, i32) -> ()
  hal.command_buffer.dispatch.indirect<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.executable)[0]
      workgroups(%arg2 : !hal.buffer)[%c100]
  return
}
