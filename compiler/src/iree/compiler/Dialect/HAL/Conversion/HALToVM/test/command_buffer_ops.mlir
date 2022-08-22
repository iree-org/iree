// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --canonicalize %s | FileCheck %s

// CHECK-LABEL: @command_buffer_create
func.func @command_buffer_create(%arg0: !hal.device) {
  // CHECK: %ref = vm.call @hal.command_buffer.create(%arg0, %c1, %c3) : (!vm.ref<!hal.device>, i32, i32) -> !vm.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create device(%arg0 : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  return
}

// -----

// CHECK-LABEL: @command_buffer_finalize
func.func @command_buffer_finalize(%arg0: !hal.command_buffer) {
  // CHECK: vm.call @hal.command_buffer.finalize(%arg0) : (!vm.ref<!hal.command_buffer>) -> ()
  hal.command_buffer.finalize<%arg0 : !hal.command_buffer>
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
func.func @command_buffer_execution_barrier(
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

// CHECK-LABEL: @command_buffer_fill_buffer_i8
func.func @command_buffer_fill_buffer_i8(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer,
  %arg2: i8
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 1
  // CHECK-DAG: %[[EXTEND:.+]] = vm.ext.i8.i32.u %arg2 : i32 -> i32
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %c100, %c200, %[[EXTEND]], %[[PATTERN_LENGTH]]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.buffer>, i64, i64, i32, i32) -> ()
  hal.command_buffer.fill_buffer<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.buffer)[%c100, %c200]
      pattern(%arg2 : i8)
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer_i16
func.func @command_buffer_fill_buffer_i16(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer,
  %arg2: i16
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 2
  // CHECK-DAG: %[[EXTEND:.+]] = vm.ext.i16.i32.u %arg2 : i32 -> i32
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %c100, %c200, %[[EXTEND]], %[[PATTERN_LENGTH]]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.buffer>, i64, i64, i32, i32) -> ()
  hal.command_buffer.fill_buffer<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.buffer)[%c100, %c200]
      pattern(%arg2 : i16)
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer_i32
func.func @command_buffer_fill_buffer_i32(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer,
  %arg2: i32
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 4
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %c100, %c200, %arg2, %[[PATTERN_LENGTH]]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.buffer>, i64, i64, i32, i32) -> ()
  hal.command_buffer.fill_buffer<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.buffer)[%c100, %c200]
      pattern(%arg2 : i32)
  return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
func.func @command_buffer_copy_buffer(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK: vm.call @hal.command_buffer.copy_buffer(%arg0, %arg1, %c100, %arg1, %c200, %c300) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.buffer>, i64, !vm.ref<!hal.buffer>, i64, i64) -> ()
  hal.command_buffer.copy_buffer<%arg0 : !hal.command_buffer>
      source(%arg1 : !hal.buffer)[%c100]
      target(%arg1 : !hal.buffer)[%c200]
      length(%c300)
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch
func.func @command_buffer_dispatch(
  %arg0: !hal.command_buffer,
  %arg1: !hal.executable
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK: vm.call @hal.command_buffer.dispatch(%arg0, %arg1, %zero, %c100, %c200, %c300) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32) -> ()
  hal.command_buffer.dispatch<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.executable)[0]
      workgroups([%c100, %c200, %c300])
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
func.func @command_buffer_dispatch_indirect(
  %arg0: !hal.command_buffer,
  %arg1: !hal.executable,
  %arg2: !hal.buffer
) {
  %c100 = arith.constant 100 : index
  // CHECK: vm.call @hal.command_buffer.dispatch.indirect(%arg0, %arg1, %zero, %arg2, %c100) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, !vm.ref<!hal.buffer>, i64) -> ()
  hal.command_buffer.dispatch.indirect<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.executable)[0]
      workgroups(%arg2 : !hal.buffer)[%c100]
  return
}
