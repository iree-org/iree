// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @skip_command_buffer_device
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64)
util.func public @skip_command_buffer_device(%device: !hal.device, %affinity: i64) -> !hal.executable {
  %cmd = hal.command_buffer.create device(%device : !hal.device)
                                     mode(OneShot)
                               categories("Transfer|Dispatch")
                                 affinity(%affinity) : !hal.command_buffer

  // CHECK-NOT: hal.command_buffer.device
  //      CHECK: = hal.executable.lookup device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:     executable(@executable_name) : !hal.executable
  %device2 = hal.command_buffer.device<%cmd : !hal.command_buffer> : !hal.device
  %exe = hal.executable.lookup device(%device2 : !hal.device)
                           executable(@executable_name) : !hal.executable

  util.return %exe : !hal.executable
}

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_fill_buffer
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[BASE_BUFFER:.+]]: !hal.buffer
util.func public @fold_buffer_subspan_into_fill_buffer(
    %cmd: !hal.command_buffer,
    %buffer: !hal.buffer
  ) {
  %c0 = arith.constant 0 : index
  %c8192 = arith.constant 8192 : index
  %c100000 = arith.constant 100000 : index
  %c262144 = arith.constant 262144 : index
  %c1234_i32 = arith.constant 1234 : i32
  %target_subspan = hal.buffer.subspan<%buffer : !hal.buffer>[%c8192, %c262144] : !hal.buffer
  // CHECK: hal.command_buffer.fill_buffer
  hal.command_buffer.fill_buffer<%cmd : !hal.command_buffer>
      // CHECK-SAME: target(%[[BASE_BUFFER]] : !hal.buffer)[%c108192, %c8192]
      target(%target_subspan : !hal.buffer)[%c100000, %c8192]
      pattern(%c1234_i32 : i32)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @fold_buffer_subspans_into_update_buffer
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[SOURCE_BUFFER:.+]]: !util.buffer, %[[SOURCE_BUFFER_SIZE:.+]]: index,
//  CHECK-SAME: %[[TARGET_BUFFER:.+]]: !hal.buffer
util.func public @fold_buffer_subspans_into_update_buffer(
    %cmd: !hal.command_buffer,
    %source_buffer: !util.buffer, %source_buffer_size: index,
    %target_buffer: !hal.buffer
  ) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c8192 = arith.constant 8192 : index
  %c100000 = arith.constant 100000 : index
  %c262144 = arith.constant 262144 : index
  %source_subspan = util.buffer.subspan %source_buffer[%c4096] : !util.buffer{%source_buffer_size} -> !util.buffer{%c262144}
  %target_subspan = hal.buffer.subspan<%target_buffer : !hal.buffer>[%c8192, %c262144] : !hal.buffer
  // CHECK: hal.command_buffer.update_buffer
  hal.command_buffer.update_buffer<%cmd : !hal.command_buffer>
      // CHECK-SAME: source(%[[SOURCE_BUFFER]] : !util.buffer{%[[SOURCE_BUFFER_SIZE]]})[%c4096]
      source(%source_subspan : !util.buffer{%c262144})[%c0]
      // CHECK-SAME: target(%[[TARGET_BUFFER]] : !hal.buffer)[%c108192]
      target(%target_subspan : !hal.buffer)[%c100000]
      // CHECK-SAME: length(%c8192)
      length(%c8192)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_copy_buffer
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[BASE_BUFFER:.+]]: !hal.buffer
util.func public @fold_buffer_subspan_into_copy_buffer(
    %cmd: !hal.command_buffer,
    %buffer: !hal.buffer
  ) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c8192 = arith.constant 8192 : index
  %c100000 = arith.constant 100000 : index
  %c262144 = arith.constant 262144 : index
  %source_subspan = hal.buffer.subspan<%buffer : !hal.buffer>[%c4096, %c262144] : !hal.buffer
  %target_subspan = hal.buffer.subspan<%buffer : !hal.buffer>[%c8192, %c262144] : !hal.buffer
  // CHECK: hal.command_buffer.copy_buffer
  hal.command_buffer.copy_buffer<%cmd : !hal.command_buffer>
      // CHECK-SAME: source(%[[BASE_BUFFER]] : !hal.buffer)[%c4096]
      source(%source_subspan : !hal.buffer)[%c0]
      // CHECK-SAME: target(%[[BASE_BUFFER]] : !hal.buffer)[%c108192]
      target(%target_subspan : !hal.buffer)[%c100000]
      length(%c8192)
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_dispatch
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[EXECUTABLE:.+]]: !hal.executable,
//  CHECK-SAME: %[[BASE_BUFFER:.+]]: !hal.buffer
util.func public @fold_buffer_subspan_into_dispatch(
    %cmd: !hal.command_buffer,
    %executable: !hal.executable,
    %buffer: !hal.buffer
  ) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c4096 = arith.constant 4096 : index
  %c8000 = arith.constant 8000 : index
  %c262140 = arith.constant 262140 : index
  %c262144 = arith.constant 262144 : index
  %subspan = hal.buffer.subspan<%buffer : !hal.buffer>[%c4096, %c262144] : !hal.buffer
  //      CHECK: hal.command_buffer.dispatch
  // CHECK-SAME:   bindings([
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
      target(%executable: !hal.executable)[%c0]
      workgroups([%c1, %c1, %c1])
      bindings([
        // 0 + 4096:
        // CHECK-NEXT: (%[[BASE_BUFFER]] : !hal.buffer)[%c4096, %c8000]
        (%subspan : !hal.buffer)[%c0, %c8000],
        // 4096 + 4:
        // CHECK-NEXT: (%[[BASE_BUFFER]] : !hal.buffer)[%c4100, %c262140]
        (%subspan : !hal.buffer)[%c4, %c262140],
        // No change:
        // CHECK-NEXT: (%[[BASE_BUFFER]] : !hal.buffer)[%c4096, %c262144]
        (%buffer : !hal.buffer)[%c4096, %c262144]
      ])
      flags("None")
  util.return
}

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_dispatch_indirect
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[EXECUTABLE:.+]]: !hal.executable,
//  CHECK-SAME: %[[BASE_BUFFER:.+]]: !hal.buffer
util.func public @fold_buffer_subspan_into_dispatch_indirect(
    %cmd: !hal.command_buffer,
    %executable: !hal.executable,
    %buffer: !hal.buffer
  ) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4096 = arith.constant 4096 : index
  %c262144 = arith.constant 262144 : index
  %subspan = hal.buffer.subspan<%buffer : !hal.buffer>[%c4096, %c262144] : !hal.buffer
  // CHECK: hal.command_buffer.dispatch.indirect
  hal.command_buffer.dispatch.indirect<%cmd : !hal.command_buffer>
      target(%executable: !hal.executable)[%c0]
      // 4096 + 4:
      // CHECK-SAME: workgroups(%[[BASE_BUFFER]] : !hal.buffer)[%c4100]
      workgroups(%subspan : !hal.buffer)[%c4]
      bindings([
        (%buffer : !hal.buffer)[%c0, %c1]
      ])
      flags("None")
  util.return
}
