// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --canonicalize --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @command_buffer_create
util.func public @command_buffer_create(%arg0: !hal.device) {
  // CHECK: %ref = vm.call @hal.command_buffer.create(%arg0, %c1, %c3, %zero) : (!vm.ref<!hal.device>, i32, i32, i32) -> !vm.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create device(%arg0 : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_create_bindings
util.func public @command_buffer_create_bindings(%arg0: !hal.device, %arg1: index) {
  // CHECK: %ref = vm.call @hal.command_buffer.create(%arg0, %c1, %c3, %arg1) : (!vm.ref<!hal.device>, i32, i32, i32) -> !vm.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create device(%arg0 : !hal.device) mode("OneShot") categories("Transfer|Dispatch") bindings(%arg1) : !hal.command_buffer
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_finalize
util.func public @command_buffer_finalize(%arg0: !hal.command_buffer) {
  // CHECK: vm.call @hal.command_buffer.finalize(%arg0) : (!vm.ref<!hal.command_buffer>) -> ()
  hal.command_buffer.finalize<%arg0 : !hal.command_buffer>
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
util.func public @command_buffer_execution_barrier(
  %arg0: !hal.command_buffer,
  %arg1: !hal.buffer
) {
  // CHECK: vm.call @hal.command_buffer.execution_barrier(%arg0, %c1, %c2, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32)
  hal.command_buffer.execution_barrier<%arg0 : !hal.command_buffer>
      source("CommandIssue")
      target("CommandProcess")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer_i8
util.func public @command_buffer_fill_buffer_i8(
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
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer_i16
util.func public @command_buffer_fill_buffer_i16(
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
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer_i32
util.func public @command_buffer_fill_buffer_i32(
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
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
util.func public @command_buffer_copy_buffer(
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
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_collective_all_reduce_sum
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[CHANNEL:.+]]: !vm.ref<!hal.channel>,
//  CHECK-SAME:  %[[PARAM:.+]]: i32,
//  CHECK-SAME:  %[[SEND_BUFFER:.+]]: !vm.ref<!hal.buffer>, %[[RECV_BUFFER:.+]]: !vm.ref<!hal.buffer>,
//  CHECK-SAME:  %[[COUNT:.+]]: i32)
util.func public @command_buffer_collective_all_reduce_sum(
    %cmd: !hal.command_buffer,
    %channel: !hal.channel,
    %param: i32,
    %send_buffer: !hal.buffer, %recv_buffer: !hal.buffer,
    %count: index) {
  // CHECK-DAG: %[[OP_BITS:.+]] = vm.const.i32 590081
  // CHECK-DAG: %[[PARAM:.+]] = vm.const.i32.zero
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK-DAG: %[[COUNT_I64:.+]] = vm.ext.i32.i64.s %[[COUNT]]
  // CHECK: vm.call @hal.command_buffer.collective
  // CHECK-SAME: (%[[CMD]], %[[CHANNEL]], %[[OP_BITS]], %[[PARAM]]
  // CHECK-SAME:  %[[SEND_BUFFER]], %c10, %c128,
  // CHECK-SAME:  %[[RECV_BUFFER]], %c20, %c256,
  // CHECK-SAME:  %[[COUNT_I64]])
  hal.command_buffer.collective<%cmd : !hal.command_buffer>
      channel(%channel : !hal.channel)
      op(<all_reduce with sum : f32>)
      send(%send_buffer : !hal.buffer)[%c10, %c128]
      recv(%recv_buffer : !hal.buffer)[%c20, %c256]
      count(%count)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_collective_send
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[CHANNEL:.+]]: !vm.ref<!hal.channel>,
//  CHECK-SAME:  %[[PARAM:.+]]: i32,
//  CHECK-SAME:  %[[SEND_BUFFER:.+]]: !vm.ref<!hal.buffer>,
//  CHECK-SAME:  %[[COUNT:.+]]: i32)
util.func public @command_buffer_collective_send(
    %cmd: !hal.command_buffer,
    %channel: !hal.channel,
    %param: i32,
    %send_buffer: !hal.buffer,
    %count: index) {
  // CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK-DAG: %[[OP_BITS:.+]] = vm.const.i32 262150
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[COUNT_I64:.+]] = vm.ext.i32.i64.s %[[COUNT]]
  // CHECK: vm.call @hal.command_buffer.collective
  // CHECK-SAME: (%[[CMD]], %[[CHANNEL]], %[[OP_BITS]], %[[PARAM]],
  // CHECK-SAME:  %[[SEND_BUFFER]], %c10, %c128,
  // CHECK-SAME:  %[[NULL_BUFFER]], %zero, %zero,
  // CHECK-SAME:  %[[COUNT_I64]])
  hal.command_buffer.collective<%cmd : !hal.command_buffer>
      channel(%channel : !hal.channel)
      op(<send : si32>)
      param(%param : i32)
      send(%send_buffer : !hal.buffer)[%c10, %c128]
      count(%count)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_push_descriptor_set
//  CHECK-SAME: %[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME: %[[LAYOUT:.+]]: !vm.ref<!hal.pipeline_layout>,
//  CHECK-SAME: %[[BUFFER:.+]]: !vm.ref<!hal.buffer>,
//  CHECK-SAME: %[[SLOT:.+]]: i32
util.func public @command_buffer_push_descriptor_set(
    %cmd: !hal.command_buffer,
    %layout: !hal.pipeline_layout,
    %buffer: !hal.buffer,
    %slot: index
  ) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4096 = arith.constant 4096 : index
  %c8000 = arith.constant 8000 : index
  // CHECK: %[[C0:.+]] = vm.const.i32.zero
  // CHECK: %[[C1:.+]] = vm.const.i32 1
  // CHECK: %[[NULL:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK: vm.call.variadic @hal.command_buffer.push_descriptor_set
  // CHECK-SAME: (%[[CMD]], %[[LAYOUT]], %c1, [
  // CHECK-SAME:   (%[[C0]], %[[C0]], %[[BUFFER]], %c4096, %c8000),
  // CHECK-SAME:   (%[[C1]], %[[SLOT]], %[[NULL]], %c4, %c4096)
  // CHECK-SAME: ]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.pipeline_layout>, i32, tuple<i32, i32, !vm.ref<!hal.buffer>, i64, i64> ...)
  hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer>
      layout(%layout : !hal.pipeline_layout)[%c1]
      bindings([
        %c0 = (%buffer : !hal.buffer)[%c4096, %c8000],
        %c1 = (%slot : index)[%c4, %c4096]
      ])
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch
util.func public @command_buffer_dispatch(
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
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
util.func public @command_buffer_dispatch_indirect(
  %arg0: !hal.command_buffer,
  %arg1: !hal.executable,
  %arg2: !hal.buffer
) {
  %c100 = arith.constant 100 : index
  // CHECK: vm.call @hal.command_buffer.dispatch.indirect(%arg0, %arg1, %zero, %arg2, %c100) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, !vm.ref<!hal.buffer>, i64) -> ()
  hal.command_buffer.dispatch.indirect<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.executable)[0]
      workgroups(%arg2 : !hal.buffer)[%c100]
  util.return
}
