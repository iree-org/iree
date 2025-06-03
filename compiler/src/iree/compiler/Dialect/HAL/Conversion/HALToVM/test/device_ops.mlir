// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=32},canonicalize)' %s | FileCheck %s

// CHECK-LABEL: @device_allocator
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @device_allocator(%device: !hal.device) -> !hal.allocator {
  // CHECK: %ref = vm.call @hal.device.allocator(%[[DEVICE]]) {nosideeffects} : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  util.return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @device_query_i64
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @device_query_i64(%device: !hal.device) -> (i1, i64) {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i64(%[[DEVICE]], %[[NS]], %[[KEY]]) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i64
  // CHECK: vm.return %[[RET]]#0, %[[RET]]#1
  util.return %ok, %value : i1, i64
}

// -----

// CHECK-LABEL: @device_query_i64_default
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @device_query_i64_default(%device: !hal.device) -> i64 {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i64(%[[DEVICE]], %[[NS]], %[[KEY]]) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i64 = 123 : i64
  // CHECK: %[[OUT:.+]] = vm.select.i64 %[[RET]]#0, %[[RET]]#1, %c123 : i64
  // CHECK: vm.return %[[OUT]]
  util.return %value : i64
}

// -----

// CHECK-LABEL: @device_query_i32
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @device_query_i32(%device: !hal.device) -> (i1, i32) {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i64(%[[DEVICE]], %[[NS]], %[[KEY]]) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)
  // CHECK: %[[RET_I32:.+]] = vm.trunc.i64.i32 %[[RET]]#1 : i64 -> i32
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32
  // CHECK: vm.return %[[RET]]#0, %[[RET_I32]]
  util.return %ok, %value : i1, i32
}

// -----

// CHECK-LABEL: @device_query_i32_default
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @device_query_i32_default(%device: !hal.device) -> i32 {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i64(%[[DEVICE]], %[[NS]], %[[KEY]]) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)
  // CHECK: %[[RET_I32:.+]] = vm.trunc.i64.i32 %[[RET]]#1 : i64 -> i32
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32 = 123 : i32
  // CHECK: %[[OUT:.+]] = vm.select.i32 %[[RET]]#0, %[[RET_I32]], %c123 : i32
  // CHECK: vm.return %[[OUT]]
  util.return %value : i32
}

// -----

// CHECK-LABEL: @device_query_i1
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @device_query_i1(%device: !hal.device) -> (i1, i1) {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i64(%[[DEVICE]], %[[NS]], %[[KEY]]) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)
  // CHECK: %[[RET_I1:.+]] = vm.cmp.nz.i64 %[[RET]]#1 : i64
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i1
  // CHECK: vm.return %[[RET]]#0, %[[RET_I1]]
  util.return %ok, %value : i1, i1
}

// -----

// CHECK-LABEL: @device_query_i1_default
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
util.func public @device_query_i1_default(%device: !hal.device) -> i1 {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i64(%[[DEVICE]], %[[NS]], %[[KEY]]) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)
  // CHECK: %[[RET_I1:.+]] = vm.cmp.nz.i64 %[[RET]]#1 : i64
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i1 = 1 : i1
  // CHECK: %[[OUT:.+]] = vm.select.i32 %[[RET]]#0, %[[RET_I1]], %c1
  // CHECK: vm.return %[[OUT]]
  util.return %value : i1
}

// -----

// CHECK-LABEL: @device_queue_alloca
util.func public @device_queue_alloca(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME: %[[SIZE_I32:.+]]: i32)
    %size: index) -> !hal.buffer {
  %c100_i64 = arith.constant 100 : i64
  // CHECK: %{{.+}} = vm.const.i64.zero
  // CHECK: %{{.+}} = vm.const.i64 100
  // CHECK: %{{.+}} = vm.const.i32 48
  %memory_type = hal.memory_type<"DeviceLocal"> : i32
  // CHECK: %{{.+}} = vm.const.i32 3
  %buffer_usage = hal.buffer_usage<"Transfer"> : i32
  // CHECK: %{{.+}} = vm.ext.i32.i64.s %[[SIZE_I32]]
  // CHECK: = vm.call @hal.device.queue.alloca(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %c100, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})
  %buffer = hal.device.queue.alloca<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      pool(%c100_i64)
      type(%memory_type) usage(%buffer_usage)
      flags("None")
      : !hal.buffer{%size}
  util.return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @device_queue_dealloca
util.func public @device_queue_dealloca(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME: %[[BUFFER:.+]]: !vm.ref<!hal.buffer>)
    %buffer: !hal.buffer) {
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.device.queue.dealloca(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[BUFFER]],
  // CHECK-SAME: %[[FLAGS]])
  hal.device.queue.dealloca<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      buffer(%buffer : !hal.buffer)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_fill_i8
util.func public @device_queue_fill_i8(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[PATTERN_I8_I32:.+]]: i32,
    %pattern_i8: i8,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = vm.const.i64 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 300
  %length = arith.constant 300 : index
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 1
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK-DAG: %[[PATTERN_I8_I64:.+]] = vm.ext.i32.i64.s %[[PATTERN_I8_I32]]
  // CHECK: vm.call @hal.device.queue.fill(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[TARGET_BUFFER]], %[[TARGET_OFFSET]],
  // CHECK-SAME: %[[LENGTH]],
  // CHECK-SAME: %[[PATTERN_I8_I64]], %[[PATTERN_LENGTH]],
  // CHECK-SAME: %[[FLAGS]])
  hal.device.queue.fill<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      target(%target_buffer : !hal.buffer)[%target_offset]
      length(%length)
      pattern(%pattern_i8 : i8)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_fill_i32
util.func public @device_queue_fill_i32(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[PATTERN_I32:.+]]: i32,
    %pattern_i32: i32,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = vm.const.i64 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 300
  %length = arith.constant 300 : index
  // CHECK-DAG: %[[PATTERN_LENGTH:.+]] = vm.const.i32 4
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK-DAG: %[[PATTERN_I32_I64:.+]] = vm.ext.i32.i64.s %[[PATTERN_I32]]
  // CHECK: vm.call @hal.device.queue.fill(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[TARGET_BUFFER]], %[[TARGET_OFFSET]],
  // CHECK-SAME: %[[LENGTH]],
  // CHECK-SAME: %[[PATTERN_I32_I64]], %[[PATTERN_LENGTH]],
  // CHECK-SAME: %[[FLAGS]])
  hal.device.queue.fill<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      target(%target_buffer : !hal.buffer)[%target_offset]
      length(%length)
      pattern(%pattern_i32 : i32)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_update
util.func public @device_queue_update(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_BUFFER:.+]]: !vm.buffer,
    %source_buffer: !util.buffer,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = vm.const.i64 100
  %source_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = vm.const.i64 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 300
  %length = arith.constant 300 : index
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.device.queue.update(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[SOURCE_BUFFER]], %[[SOURCE_OFFSET]],
  // CHECK-SAME: %[[TARGET_BUFFER]], %[[TARGET_OFFSET]],
  // CHECK-SAME: %[[LENGTH]], %[[FLAGS]])
  hal.device.queue.update<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      source(%source_buffer : !util.buffer)[%source_offset]
      target(%target_buffer : !hal.buffer)[%target_offset]
      length(%length)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_copy
util.func public @device_queue_copy(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_BUFFER:.+]]: !vm.ref<!hal.buffer>,
    %source_buffer: !hal.buffer,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = vm.const.i64 100
  %source_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = vm.const.i64 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 300
  %length = arith.constant 300 : index
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.device.queue.copy(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[SOURCE_BUFFER]], %[[SOURCE_OFFSET]],
  // CHECK-SAME: %[[TARGET_BUFFER]], %[[TARGET_OFFSET]],
  // CHECK-SAME: %[[LENGTH]], %[[FLAGS]])
  hal.device.queue.copy<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      source(%source_buffer : !hal.buffer)[%source_offset]
      target(%target_buffer : !hal.buffer)[%target_offset]
      length(%length)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_read
util.func public @device_queue_read(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_FILE:.+]]: !vm.ref<!hal.file>,
    %source_file: !hal.file,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !vm.ref<!hal.buffer>)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = vm.const.i64 100
  %source_offset = arith.constant 100 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = vm.const.i64 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 300
  %length = arith.constant 300 : index
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.device.queue.read(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[SOURCE_FILE]], %[[SOURCE_OFFSET]],
  // CHECK-SAME: %[[TARGET_BUFFER]], %[[TARGET_OFFSET]],
  // CHECK-SAME: %[[LENGTH]], %[[FLAGS]])
  hal.device.queue.read<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      source(%source_file : !hal.file)[%source_offset]
      target(%target_buffer : !hal.buffer)[%target_offset]
      length(%length)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_barrier
util.func public @device_queue_barrier(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>)
    %wait_fence: !hal.fence, %signal_fence: !hal.fence) {
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.device.queue.barrier(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[FLAGS]])
  hal.device.queue.barrier<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_execute
util.func public @device_queue_execute(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME: %[[CMD:.+]]: !vm.ref<!hal.command_buffer>)
    %cmd: !hal.command_buffer) {
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call @hal.device.queue.execute(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[CMD]], %[[FLAGS]])
  hal.device.queue.execute<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      commands(%cmd)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_execute_indirect
util.func public @device_queue_execute_indirect(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !vm.ref<!hal.fence>, %[[SIGNAL_FENCE:.+]]: !vm.ref<!hal.fence>,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[CMD:.+]]: !vm.ref<!hal.command_buffer>,
    %cmd: !hal.command_buffer,
    // CHECK-SAME:  %[[BUFFER0:.+]]: !vm.ref<!hal.buffer>, %[[BUFFER1:.+]]: !vm.ref<!hal.buffer>
    %buffer0: !hal.buffer, %buffer1: !hal.buffer) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c1000 = arith.constant 1000 : index
  %c2000 = arith.constant 2000 : index
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: vm.call.variadic @hal.device.queue.execute.indirect(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]],
  // CHECK-SAME: %[[WAIT_FENCE]], %[[SIGNAL_FENCE]],
  // CHECK-SAME: %[[CMD]], %[[FLAGS]],
  // CHECK-SAME: [(%[[BUFFER0]], %c100, %c1000), (%[[BUFFER1]], %c200, %c2000)])
  hal.device.queue.execute.indirect<%device : !hal.device>
      affinity(%affinity)
      wait(%wait_fence) signal(%signal_fence)
      commands(%cmd)
      bindings([
        (%buffer0 : !hal.buffer)[%c100, %c1000],
        (%buffer1 : !hal.buffer)[%c200, %c2000]
      ])
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_flush
util.func public @device_queue_flush(
    // CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>, %[[AFFINITY:.+]]: i64)
    %device: !hal.device, %affinity: i64) {
  // CHECK: vm.call @hal.device.queue.flush(%[[DEVICE]], %[[AFFINITY]])
  hal.device.queue.flush<%device : !hal.device>
      affinity(%affinity)
  util.return
}
