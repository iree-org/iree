// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @skip_command_buffer_device
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func.func @skip_command_buffer_device(%device: !hal.device) -> !hal.executable {
  %cmd = hal.command_buffer.create device(%device : !hal.device)
                                     mode(OneShot)
                               categories("Transfer|Dispatch") : !hal.command_buffer

  // CHECK-NOT: hal.command_buffer.device
  //      CHECK: = hal.executable.lookup device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:     executable(@executable_name) : !hal.executable
  %device2 = hal.command_buffer.device<%cmd : !hal.command_buffer> : !hal.device
  %exe = hal.executable.lookup device(%device2 : !hal.device)
                           executable(@executable_name) : !hal.executable

  return %exe : !hal.executable
}

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_fill_buffer
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[BASE_BUFFER:.+]]: !hal.buffer
func.func @fold_buffer_subspan_into_fill_buffer(
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
  return
}

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_copy_buffer
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[BASE_BUFFER:.+]]: !hal.buffer
func.func @fold_buffer_subspan_into_copy_buffer(
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
  return
}

// -----

// CHECK-LABEL: @fold_buffer_subspan_into_push_descriptor_set
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[LAYOUT:.+]]: !hal.pipeline_layout,
//  CHECK-SAME: %[[BASE_BUFFER:.+]]: !hal.buffer
func.func @fold_buffer_subspan_into_push_descriptor_set(
    %cmd: !hal.command_buffer,
    %layout: !hal.pipeline_layout,
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
  //      CHECK: hal.command_buffer.push_descriptor_set
  // CHECK-SAME:   bindings([
  hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer>
      layout(%layout : !hal.pipeline_layout)[%c0]
      bindings([
        // 0 + 4096:
        // CHECK-NEXT: %c0 = (%[[BASE_BUFFER]] : !hal.buffer)[%c4096, %c8000]
        %c0 = (%subspan : !hal.buffer)[%c0, %c8000],
        // 4096 + 4:
        // CHECK-NEXT: %c1 = (%[[BASE_BUFFER]] : !hal.buffer)[%c4100, %c262140]
        %c1 = (%subspan : !hal.buffer)[%c4, %c262140],
        // No change:
        // CHECK-NEXT: %c2 = (%[[BASE_BUFFER]] : !hal.buffer)[%c4096, %c262144]
        %c2 = (%buffer : !hal.buffer)[%c4096, %c262144]
      ])
  return
}
