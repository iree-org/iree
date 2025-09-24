// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @device_allocator
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
util.func public @device_allocator(%device: !hal.device) -> !hal.allocator {
  // CHECK: %allocator = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  util.return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @device_query
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
util.func public @device_query(%device : !hal.device) -> (i1, i32) {
  // CHECK: = hal.device.query<%[[DEVICE]] : !hal.device> key("sys" :: "foo") : i1, i32
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32
  util.return %ok, %value : i1, i32
}

// -----

// CHECK-LABEL: @device_queue_alloca
util.func public @device_queue_alloca(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SIZE:.+]]: index)
    %size: index) -> !hal.buffer {
  %c100_i64 = arith.constant 100 : i64
  // CHECK: %[[MEMORY_TYPE:.+]] = hal.memory_type<{{.+}}> : i32
  %memory_type = hal.memory_type<"DeviceLocal"> : i32
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"{{.+}}Transfer"> : i32
  %buffer_usage = hal.buffer_usage<"Transfer"> : i32
  // CHECK: = hal.device.queue.alloca<%[[DEVICE]] : !hal.device>
  %buffer = hal.device.queue.alloca<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: pool(%c100_i64)
      pool(%c100_i64)
      // CHECK-SAME: type(%[[MEMORY_TYPE]]) usage(%[[BUFFER_USAGE]])
      type(%memory_type) usage(%buffer_usage)
      // CHECK-SAME: flags("None")
      flags("None")
      // CHECK-SAME: : !hal.buffer{%[[SIZE]]}
      : !hal.buffer{%size}
  util.return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @device_queue_dealloca
util.func public @device_queue_dealloca(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[BUFFER:.+]]: !hal.buffer)
    %buffer: !hal.buffer) {
  // CHECK: hal.device.queue.dealloca<%[[DEVICE]] : !hal.device>
  hal.device.queue.dealloca<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: buffer(%[[BUFFER]] : !hal.buffer)
      buffer(%buffer : !hal.buffer)
      // CHECK-SAME: flags(PreferOrigin)
      flags(PreferOrigin)
  util.return
}

// -----

// CHECK-LABEL: @device_queue_fill
util.func public @device_queue_fill(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[PATTERN_I8:.+]]: i8,
    %pattern_i8: i8,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !hal.buffer)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 300
  %length = arith.constant 300 : index
  // CHECK: hal.device.queue.fill<%[[DEVICE]] : !hal.device>
  hal.device.queue.fill<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: target(%[[TARGET_BUFFER]] : !hal.buffer)[%[[TARGET_OFFSET]]]
      target(%target_buffer : !hal.buffer)[%target_offset]
      // CHECK-SAME: length(%[[LENGTH]])
      length(%length)
      // CHECK-SAME: pattern(%[[PATTERN_I8]] : i8)
      pattern(%pattern_i8 : i8)
      // CHECK-SAME: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_update
util.func public @device_queue_update(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_BUFFER:.+]]: !util.buffer,
    %source_buffer: !util.buffer,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !hal.buffer)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100
  %source_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 300
  %length = arith.constant 300 : index
  // CHECK: hal.device.queue.update<%[[DEVICE]] : !hal.device>
  hal.device.queue.update<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: source(%[[SOURCE_BUFFER]] : !util.buffer)[%[[SOURCE_OFFSET]]]
      source(%source_buffer : !util.buffer)[%source_offset]
      // CHECK-SAME: target(%[[TARGET_BUFFER]] : !hal.buffer)[%[[TARGET_OFFSET]]]
      target(%target_buffer : !hal.buffer)[%target_offset]
      // CHECK-SAME: length(%[[LENGTH]])
      length(%length)
      // CHECK-SAME: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_copy
util.func public @device_queue_copy(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_BUFFER:.+]]: !hal.buffer,
    %source_buffer: !hal.buffer,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !hal.buffer)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100
  %source_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 300
  %length = arith.constant 300 : index
  // CHECK: hal.device.queue.copy<%[[DEVICE]] : !hal.device>
  hal.device.queue.copy<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: source(%[[SOURCE_BUFFER]] : !hal.buffer)[%[[SOURCE_OFFSET]]]
      source(%source_buffer : !hal.buffer)[%source_offset]
      // CHECK-SAME: target(%[[TARGET_BUFFER]] : !hal.buffer)[%[[TARGET_OFFSET]]]
      target(%target_buffer : !hal.buffer)[%target_offset]
      // CHECK-SAME: length(%[[LENGTH]])
      length(%length)
      // CHECK-SAME: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_read
util.func public @device_queue_read(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_FILE:.+]]: !hal.file,
    %source_file: !hal.file,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !hal.buffer)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100
  %source_offset = arith.constant 100 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 300
  %length = arith.constant 300 : index
  // CHECK: hal.device.queue.read<%[[DEVICE]] : !hal.device>
  hal.device.queue.read<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: source(%[[SOURCE_FILE]] : !hal.file)[%[[SOURCE_OFFSET]]]
      source(%source_file : !hal.file)[%source_offset]
      // CHECK-SAME: target(%[[TARGET_BUFFER]] : !hal.buffer)[%[[TARGET_OFFSET]]]
      target(%target_buffer : !hal.buffer)[%target_offset]
      // CHECK-SAME: length(%[[LENGTH]])
      length(%length)
      // CHECK-SAME: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_write
util.func public @device_queue_write(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_BUFFER:.+]]: !hal.buffer,
    %source_buffer: !hal.buffer,
    // CHECK-SAME:  %[[TARGET_FILE:.+]]: !hal.file)
    %target_file: !hal.file) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100
  %source_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 200
  %target_offset = arith.constant 200 : i64
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 300
  %length = arith.constant 300 : index
  // CHECK: hal.device.queue.write<%[[DEVICE]] : !hal.device>
  hal.device.queue.write<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: source(%[[SOURCE_BUFFER]] : !hal.buffer)[%[[SOURCE_OFFSET]]]
      source(%source_buffer : !hal.buffer)[%source_offset]
      // CHECK-SAME: target(%[[TARGET_FILE]] : !hal.file)[%[[TARGET_OFFSET]]]
      target(%target_file : !hal.file)[%target_offset]
      // CHECK-SAME: length(%[[LENGTH]])
      length(%length)
      // CHECK-SAME: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_barrier
util.func public @device_queue_barrier(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence)
    %wait_fence: !hal.fence, %signal_fence: !hal.fence) {
  // CHECK: hal.device.queue.barrier<%[[DEVICE]] : !hal.device>
  hal.device.queue.barrier<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_execute
util.func public @device_queue_execute(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[CMD:.+]]: !hal.command_buffer)
    %cmd: !hal.command_buffer) {
  // CHECK: hal.device.queue.execute<%[[DEVICE]] : !hal.device>
  hal.device.queue.execute<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: commands(%[[CMD]])
      commands(%cmd)
      // CHECK-SAME: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_execute_indirect
util.func public @device_queue_execute_indirect(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[CMD:.+]]: !hal.command_buffer,
    %cmd: !hal.command_buffer,
    // CHECK-SAME:  %[[BUFFER0:.+]]: !hal.buffer, %[[BUFFER1:.+]]: !hal.buffer
    %buffer0: !hal.buffer, %buffer1: !hal.buffer) {
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c1000 = arith.constant 1000 : index
  %c2000 = arith.constant 2000 : index
  // CHECK: hal.device.queue.execute.indirect<%[[DEVICE]] : !hal.device>
  hal.device.queue.execute.indirect<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: commands(%[[CMD]])
      commands(%cmd)
      // CHECK-SAME: bindings([
      bindings([
        // CHECK-NEXT: (%[[BUFFER0]] : !hal.buffer)[%c100, %c1000]
        (%buffer0 : !hal.buffer)[%c100, %c1000],
        // CHECK-NEXT: (%[[BUFFER1]] : !hal.buffer)[%c200, %c2000]
        (%buffer1 : !hal.buffer)[%c200, %c2000]
      ])
      // CHECK-NEXT: flags("None")
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @device_queue_flush
util.func public @device_queue_flush(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64)
    %device: !hal.device, %affinity: i64) {
  // CHECK: hal.device.queue.flush<%[[DEVICE]] : !hal.device>
  hal.device.queue.flush<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
  util.return
}

// -----

// CHECK-LABEL: @device_memoize
util.func public @device_memoize(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64)
    %device: !hal.device, %affinity: i64) -> !hal.command_buffer {
  // CHECK: hal.device.memoize<%[[DEVICE]] : !hal.device> affinity(%[[AFFINITY]])
  %result = hal.device.memoize<%device : !hal.device> affinity(%affinity) -> !hal.command_buffer {
    // CHECK-NEXT: hal.command_buffer.create
    %cmd = hal.command_buffer.create device(%device : !hal.device)
                                       mode(OneShot)
                                 categories("Transfer|Dispatch")
                                   affinity(%affinity) : !hal.command_buffer
    // CHECK-NEXT: hal.return
    hal.return %cmd : !hal.command_buffer
  }
  util.return %result : !hal.command_buffer
}
