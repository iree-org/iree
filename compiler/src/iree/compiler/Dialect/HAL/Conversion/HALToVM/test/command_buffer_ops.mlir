// RUN: iree-opt --split-input-file --iree-vm-conversion --canonicalize --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @command_buffer_create
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64)
util.func public @command_buffer_create(%device: !hal.device, %affinity: i64) {
  // CHECK: = vm.call @hal.command_buffer.create(%[[DEVICE]], %c1, %c3, %[[AFFINITY]], %zero) : (!vm.ref<!hal.device>, i32, i32, i64, i32) -> !vm.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") affinity(%affinity) : !hal.command_buffer
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_create_bindings
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64, %[[CAPACITY:.+]]: i32)
util.func public @command_buffer_create_bindings(%device: !hal.device, %affinity: i64, %capacity: index) {
  // CHECK: = vm.call @hal.command_buffer.create(%[[DEVICE]], %c1, %c3, %[[AFFINITY]], %[[CAPACITY]]) : (!vm.ref<!hal.device>, i32, i32, i64, i32) -> !vm.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") affinity(%affinity) bindings(%capacity) : !hal.command_buffer
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
  // CHECK-DAG: %[[UNUSED_SLOT:.+]] = vm.const.i32.zero
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 1
  // CHECK-DAG: %[[EXTEND:.+]] = vm.ext.i8.i32.u %arg2 : i32 -> i32
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %c100, %c200, %[[UNUSED_SLOT]], %[[EXTEND]], %[[PATTERN_LENGTH]]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.buffer>, i64, i64, i32, i32, i32) -> ()
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
  // CHECK-DAG: %[[UNUSED_SLOT:.+]] = vm.const.i32.zero
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 2
  // CHECK-DAG: %[[EXTEND:.+]] = vm.ext.i16.i32.u %arg2 : i32 -> i32
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %c100, %c200, %[[UNUSED_SLOT]], %[[EXTEND]], %[[PATTERN_LENGTH]])
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
  // CHECK-DAG: %[[UNUSED_SLOT:.+]] = vm.const.i32.zero
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 4
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %c100, %c200, %[[UNUSED_SLOT]], %arg2, %[[PATTERN_LENGTH]])
  hal.command_buffer.fill_buffer<%arg0 : !hal.command_buffer>
      target(%arg1 : !hal.buffer)[%c100, %c200]
      pattern(%arg2 : i32)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer_i32_indirect
util.func public @command_buffer_fill_buffer_i32_indirect(
  %arg0: !hal.command_buffer,
  %arg1: index,
  %arg2: i32
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 4
  // CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %[[NULL_BUFFER]], %c100, %c200, %arg1, %arg2, %[[PATTERN_LENGTH]])
  hal.command_buffer.fill_buffer<%arg0 : !hal.command_buffer>
      target(%arg1 : index)[%c100, %c200]
      pattern(%arg2 : i32)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_update_buffer
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[HOST_BUFFER:[a-z0-9]+]]: !vm.buffer, %[[HOST_BUFFER_SIZE:[a-z0-9]+]]: i32, %[[SRC_OFFSET:[a-z0-9]+]]: i32,
//  CHECK-SAME:  %[[DEVICE_BUFFER:[a-z0-9]+]]: !vm.ref<!hal.buffer>, %[[DST_OFFSET:[a-z0-9]+]]: i32,
//  CHECK-SAME:  %[[LENGTH:[a-z0-9]+]]: i32)
util.func public @command_buffer_update_buffer(
    %cmd: !hal.command_buffer,
    %host_buffer: !util.buffer, %host_buffer_size: index, %src_offset: index,
    %device_buffer: !hal.buffer, %dst_offset: index,
    %length: index
  ) {
  // CHECK-DAG: %[[UNUSED_SLOT:.+]] = vm.const.i32.zero
  //  CHECK-DAG: %[[SRC_OFFSET_I64:.+]] = vm.ext.i32.i64.s %[[SRC_OFFSET]]
  //  CHECK-DAG: %[[DST_OFFSET_I64:.+]] = vm.ext.i32.i64.s %[[DST_OFFSET]]
  //  CHECK-DAG: %[[LENGTH_I64:.+]] = vm.ext.i32.i64.s %[[LENGTH]]
  //      CHECK: vm.call @hal.command_buffer.update_buffer
  // CHECK-SAME: (%[[CMD]],
  // CHECK-SAME:  %[[HOST_BUFFER]], %[[SRC_OFFSET_I64]],
  // CHECK-SAME:  %[[DEVICE_BUFFER]], %[[DST_OFFSET_I64]],
  // CHECK-SAME:  %[[LENGTH_I64]], %[[UNUSED_SLOT]])
  hal.command_buffer.update_buffer<%cmd : !hal.command_buffer>
      source(%host_buffer : !util.buffer{%host_buffer_size})[%src_offset]
      target(%device_buffer : !hal.buffer)[%dst_offset]
      length(%length)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_update_buffer_indirect
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[HOST_BUFFER:[a-z0-9]+]]: !vm.buffer, %[[HOST_BUFFER_SIZE:[a-z0-9]+]]: i32, %[[SRC_OFFSET:[a-z0-9]+]]: i32,
//  CHECK-SAME:  %[[DEVICE_BUFFER_SLOT:[a-z0-9]+]]: i32, %[[DST_OFFSET:[a-z0-9]+]]: i32,
//  CHECK-SAME:  %[[LENGTH:[a-z0-9]+]]: i32)
util.func public @command_buffer_update_buffer_indirect(
    %cmd: !hal.command_buffer,
    %host_buffer: !util.buffer, %host_buffer_size: index, %src_offset: index,
    %device_buffer: index, %dst_offset: index,
    %length: index
  ) {
  //  CHECK-DAG: %[[SRC_OFFSET_I64:.+]] = vm.ext.i32.i64.s %[[SRC_OFFSET]]
  //  CHECK-DAG: %[[DST_OFFSET_I64:.+]] = vm.ext.i32.i64.s %[[DST_OFFSET]]
  //  CHECK-DAG: %[[LENGTH_I64:.+]] = vm.ext.i32.i64.s %[[LENGTH]]
  //  CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  //      CHECK: vm.call @hal.command_buffer.update_buffer
  // CHECK-SAME: (%[[CMD]],
  // CHECK-SAME:  %[[HOST_BUFFER]], %[[SRC_OFFSET_I64]],
  // CHECK-SAME:  %[[NULL_BUFFER]], %[[DST_OFFSET_I64]],
  // CHECK-SAME:  %[[LENGTH_I64]], %[[DEVICE_BUFFER_SLOT]])
  hal.command_buffer.update_buffer<%cmd : !hal.command_buffer>
      source(%host_buffer : !util.buffer{%host_buffer_size})[%src_offset]
      target(%device_buffer : index)[%dst_offset]
      length(%length)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
// CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>, %[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @command_buffer_copy_buffer(
  %cmd: !hal.command_buffer,
  %buffer: !hal.buffer
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[UNUSED_SLOT:.+]] = vm.const.i32.zero
  // CHECK: vm.call @hal.command_buffer.copy_buffer(%[[CMD]], %[[UNUSED_SLOT]], %[[UNUSED_SLOT]], %[[BUFFER]], %c100, %[[BUFFER]], %c200, %c300)
  hal.command_buffer.copy_buffer<%cmd : !hal.command_buffer>
      source(%buffer : !hal.buffer)[%c100]
      target(%buffer : !hal.buffer)[%c200]
      length(%c300)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer_indirect
// CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>, %[[BUFFER_SLOT:.+]]: i32)
util.func public @command_buffer_copy_buffer_indirect(
  %cmd: !hal.command_buffer,
  %buffer_slot: index
) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK: vm.call @hal.command_buffer.copy_buffer(%[[CMD]], %[[BUFFER_SLOT]], %[[BUFFER_SLOT]], %[[NULL_BUFFER]], %c100, %[[NULL_BUFFER]], %c200, %c300)
  hal.command_buffer.copy_buffer<%cmd : !hal.command_buffer>
      source(%buffer_slot : index)[%c100]
      target(%buffer_slot : index)[%c200]
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
  // CHECK-DAG: %[[ZERO_I32:.+]] = vm.const.i32.zero
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK-DAG: %[[COUNT_I64:.+]] = vm.ext.i32.i64.s %[[COUNT]]
  // CHECK: vm.call @hal.command_buffer.collective
  // CHECK-SAME: (%[[CMD]], %[[CHANNEL]], %[[OP_BITS]], %[[ZERO_I32]]
  // CHECK-SAME:  %[[ZERO_I32]], %[[ZERO_I32]],
  // CHECK-SAME:  %[[SEND_BUFFER]], %[[RECV_BUFFER]],
  // CHECK-SAME:  %c10, %c128, %c20, %c256,
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
  // CHECK-DAG: %[[OP_BITS:.+]] = vm.const.i32 262150
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[COUNT_I64:.+]] = vm.ext.i32.i64.s %[[COUNT]]
  // CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK-DAG: %[[UNUSED_SLOT:.+]] = vm.const.i32.zero
  // CHECK-DAG: %[[ZERO_I64:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.command_buffer.collective
  // CHECK-SAME: (%[[CMD]], %[[CHANNEL]], %[[OP_BITS]], %[[PARAM]],
  // CHECK-SAME:  %[[UNUSED_SLOT]], %[[UNUSED_SLOT]],
  // CHECK-SAME:  %[[SEND_BUFFER]], %[[NULL_BUFFER]],
  // CHECK-SAME:  %c10, %c128, %[[ZERO_I64]], %[[ZERO_I64]],
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
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[LAYOUT:.+]]: !vm.ref<!hal.pipeline_layout>,
//  CHECK-SAME:  %[[BUFFER:.+]]: !vm.ref<!hal.buffer>,
//  CHECK-SAME:  %[[SLOT:.+]]: i32)
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
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[EXECUTABLE:.+]]: !vm.ref<!hal.executable>)
util.func public @command_buffer_dispatch(
  %cmd: !hal.command_buffer,
  %executable: !hal.executable
) {
  // CHECK-DAG: %[[ORDINAL:.+]] = vm.const.i32 123
  %ordinal = arith.constant 123 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.command_buffer.dispatch(%[[CMD]], %[[EXECUTABLE]], %[[ORDINAL]], %c100, %c200, %c300, %[[FLAGS]])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
      target(%executable : !hal.executable)[%ordinal]
      workgroups([%c100, %c200, %c300])
      flags(None)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[EXECUTABLE:.+]]: !vm.ref<!hal.executable>,
//  CHECK-SAME:  %[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
util.func public @command_buffer_dispatch_indirect(
  %cmd: !hal.command_buffer,
  %executable: !hal.executable,
  %buffer: !hal.buffer
) {
  // CHECK-DAG: %[[ORDINAL:.+]] = vm.const.i32 123
  %ordinal = arith.constant 123 : index
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[UNUSED_SLOT:.+]] = vm.const.i32.zero
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.command_buffer.dispatch.indirect(%[[CMD]], %[[EXECUTABLE]], %[[ORDINAL]], %[[UNUSED_SLOT]], %[[BUFFER]], %c100, %[[FLAGS]])
  hal.command_buffer.dispatch.indirect<%cmd : !hal.command_buffer>
      target(%executable : !hal.executable)[%ordinal]
      workgroups(%buffer : !hal.buffer)[%c100]
      flags(None)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect_indirect
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[EXECUTABLE:.+]]: !vm.ref<!hal.executable>,
//  CHECK-SAME:  %[[BUFFER_SLOT:.+]]: i32)
util.func public @command_buffer_dispatch_indirect_indirect(
  %cmd: !hal.command_buffer,
  %executable: !hal.executable,
  %buffer_slot: index
) {
  // CHECK-DAG: %[[ORDINAL:.+]] = vm.const.i32 123
  %ordinal = arith.constant 123 : index
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.command_buffer.dispatch.indirect(%[[CMD]], %[[EXECUTABLE]], %[[ORDINAL]], %[[BUFFER_SLOT]], %[[NULL_BUFFER]], %c100, %[[FLAGS]])
  hal.command_buffer.dispatch.indirect<%cmd : !hal.command_buffer>
      target(%executable : !hal.executable)[%ordinal]
      workgroups(%buffer_slot : index)[%c100]
      flags(None)
  util.return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch2
//  CHECK-SAME: (%[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[EXECUTABLE:.+]]: !vm.ref<!hal.executable>,
//  CHECK-SAME:  %[[BUFFER:.+]]: !vm.ref<!hal.buffer>,
//  CHECK-SAME:  %[[SLOT:.+]]: i32)
util.func public @command_buffer_dispatch2(
  %cmd: !hal.command_buffer,
  %executable: !hal.executable,
  %buffer: !hal.buffer,
  %slot: index
) {
  // CHECK-DAG: %[[ORDINAL:.+]] = vm.const.i32 123
  // CHECK-DAG: %[[C0:.+]] = vm.const.i32.zero
  %ordinal = arith.constant 123 : index
  // CHECK-DAG: %[[X:.+]] = vm.const.i32 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = vm.const.i32 200
  %y = arith.constant 200 : index
  // CHECK-DAG: %[[Z:.+]] = vm.const.i32 300
  %z = arith.constant 300 : index
  // CHECK-DAG: %[[CONSTANT0:.+]] = vm.const.i32 31
  %constant0 = arith.constant 31 : i32
  // CHECK-DAG: %[[CONSTANT1:.+]] = vm.const.i32 32
  %constant1 = arith.constant 32 : i32
  %c4 = arith.constant 4 : index
  %c4096 = arith.constant 4096 : index
  %c8000 = arith.constant 8000 : index
  // CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call.variadic @hal.command_buffer.dispatch2
  // CHECK-SAME: %[[CMD]],
  // CHECK-SAME: %[[EXECUTABLE]], %[[ORDINAL]],
  // CHECK-SAME: %[[X]], %[[Y]], %[[Z]],
  // CHECK-SAME: %[[FLAGS]],
  // CHECK-SAME: [%[[CONSTANT0]], %[[CONSTANT1]]],
  // CHECK-SAME: [(%[[C0]], %[[C0]], %[[BUFFER]], %c4096, %c8000),
  // CHECK-SAME:  (%[[C0]], %[[SLOT]], %[[NULL_BUFFER]], %c4, %c4096)]
  hal.command_buffer.dispatch2<%cmd : !hal.command_buffer>
      target(%executable : !hal.executable)[%ordinal]
      workgroups([%x, %y, %z])
      constants([%constant0, %constant1])
      bindings([
        (%buffer : !hal.buffer)[%c4096, %c8000],
        (%slot : index)[%c4, %c4096]
      ])
      flags(None)
  util.return
}

// -----

// CHECK-LABEL: vm.func private @command_buffer_dispatch2
//  CHECK-SAME: (%[[CMD:[a-z0-9]+]]: !vm.ref<!hal.command_buffer>,
//  CHECK-SAME:  %[[EXECUTABLE:[a-z0-9]+]]: !vm.ref<!hal.executable>,
//  CHECK-SAME:  %[[WORKGROUPS_SLOT:[a-z0-9]+]]: i32,
//  CHECK-SAME:  %[[BUFFER:[a-z0-9]+]]: !vm.ref<!hal.buffer>,
//  CHECK-SAME:  %[[SLOT:[a-z0-9]+]]: i32)
util.func public @command_buffer_dispatch2(
  %cmd: !hal.command_buffer,
  %executable: !hal.executable,
  %workgroups_slot: index,
  %buffer: !hal.buffer,
  %slot: index
) {
  // CHECK-DAG: %[[ORDINAL:.+]] = vm.const.i32 123
  // CHECK-DAG: %[[C0:.+]] = vm.const.i32.zero
  %ordinal = arith.constant 123 : index
  // CHECK-DAG: %[[WORKGROUPS_OFFSET:.+]] = vm.const.i64 100
  %workgroups_offset = arith.constant 100 : index
  // CHECK-DAG: %[[CONSTANT0:.+]] = vm.const.i32 31
  %constant0 = arith.constant 31 : i32
  // CHECK-DAG: %[[CONSTANT1:.+]] = vm.const.i32 32
  %constant1 = arith.constant 32 : i32
  %c4 = arith.constant 4 : index
  %c4096 = arith.constant 4096 : index
  %c8000 = arith.constant 8000 : index
  // CHECK-DAG: %[[NULL_BUFFER:.+]] = vm.const.ref.zero : !vm.ref<!hal.buffer>
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call.variadic @hal.command_buffer.dispatch2.indirect
  // CHECK-SAME: %[[CMD]],
  // CHECK-SAME: %[[EXECUTABLE]], %[[ORDINAL]],
  // CHECK-SAME: %[[WORKGROUPS_SLOT]], %[[NULL_BUFFER]], %[[WORKGROUPS_OFFSET]],
  // CHECK-SAME: %[[FLAGS]],
  // CHECK-SAME: [%[[CONSTANT0]], %[[CONSTANT1]]],
  // CHECK-SAME: [(%[[C0]], %[[C0]], %[[BUFFER]], %c4096, %c8000),
  // CHECK-SAME:  (%[[C0]], %[[SLOT]], %[[NULL_BUFFER]], %c4, %c4096)]
  hal.command_buffer.dispatch2.indirect<%cmd : !hal.command_buffer>
      target(%executable : !hal.executable)[%ordinal]
      workgroups(%workgroups_slot : index)[%workgroups_offset]
      constants([%constant0, %constant1])
      bindings([
        (%buffer : !hal.buffer)[%c4096, %c8000],
        (%slot : index)[%c4, %c4096]
      ])
      flags(None)
  util.return
}
