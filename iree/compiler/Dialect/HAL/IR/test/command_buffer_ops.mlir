// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @command_buffer_create
//  CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func @command_buffer_create(%device: !hal.device) {
  //      CHECK: %cmd = hal.command_buffer.create
  // CHECK-SAME:   device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:   mode(OneShot)
  // CHECK-SAME:   categories("Transfer|Dispatch") : !hal.command_buffer
  %cmd = hal.command_buffer.create device(%device : !hal.device)
                                     mode(OneShot)
                               categories("Transfer|Dispatch") : !hal.command_buffer
  return
}

// -----

// CHECK-LABEL: @command_buffer_begin_end
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer)
func @command_buffer_begin_end(%cmd: !hal.command_buffer) {
  // CHECK: hal.command_buffer.begin<%[[CMD]] : !hal.command_buffer>
  hal.command_buffer.begin<%cmd : !hal.command_buffer>
  // CHECK: hal.command_buffer.end<%[[CMD]] : !hal.command_buffer>
  hal.command_buffer.end<%cmd : !hal.command_buffer>
  return
}

// -----

// CHECK-LABEL: @command_buffer_device
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer)
func @command_buffer_device(%cmd: !hal.command_buffer) {
  // CHECK: %0 = hal.command_buffer.device<%[[CMD]] : !hal.command_buffer> : !hal.device
  %0 = hal.command_buffer.device<%cmd : !hal.command_buffer> : !hal.device
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer)
func @command_buffer_execution_barrier(%cmd: !hal.command_buffer) {
  //      CHECK: hal.command_buffer.execution_barrier<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   source(CommandIssue)
  // CHECK-SAME:   target(CommandProcess)
  // CHECK-SAME:   flags("None")
  hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer>
      source(CommandIssue)
      target(CommandProcess)
      flags(None)
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME: %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index,
//  CHECK-SAME: %[[PATTERN:.+]]: i32)
func @command_buffer_fill_buffer(
    %cmd: !hal.command_buffer,
    %buffer: !hal.buffer,
    %offset: index,
    %length: index,
    %pattern: i32
  ) {
  //      CHECK: hal.command_buffer.fill_buffer<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   target(%[[BUFFER]] : !hal.buffer)[%[[OFFSET]], %[[LENGTH]]]
  // CHECK-SAME:   pattern(%[[PATTERN]] : i32)
  hal.command_buffer.fill_buffer<%cmd : !hal.command_buffer>
      target(%buffer : !hal.buffer)[%offset, %length]
      pattern(%pattern : i32)
  return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME: %[[SRC_OFFSET:.+]]: index, %[[DST_OFFSET:.+]]: index,
//  CHECK-SAME: %[[LENGTH:.+]]: index)
func @command_buffer_copy_buffer(
    %cmd: !hal.command_buffer,
    %buffer: !hal.buffer,
    %src_offset: index,
    %dst_offset: index,
    %length: index
  ) {
  //      CHECK: hal.command_buffer.copy_buffer<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   source(%[[BUFFER]] : !hal.buffer)[%[[SRC_OFFSET]]]
  // CHECK-SAME:   target(%[[BUFFER]] : !hal.buffer)[%[[DST_OFFSET]]]
  // CHECK-SAME:   length(%[[LENGTH]])
  hal.command_buffer.copy_buffer<%cmd : !hal.command_buffer>
      source(%buffer : !hal.buffer)[%src_offset]
      target(%buffer : !hal.buffer)[%dst_offset]
      length(%length)
  return
}

// -----

// CHECK-LABEL: @command_buffer_bind_descriptor_set
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[LAYOUT:.+]]: !hal.executable_layout,
//  CHECK-SAME: %[[SET:.+]]: !hal.descriptor_set,
//  CHECK-SAME: %[[OFFSET:.+]]: index)
func @command_buffer_bind_descriptor_set(
    %cmd: !hal.command_buffer,
    %layout: !hal.executable_layout,
    %set: !hal.descriptor_set,
    %offset: index
  ) {
  // CHECK: %[[SET_IDX:.+]] = constant 0
  %c0 = constant 0 : index
  //      CHECK: hal.command_buffer.bind_descriptor_set<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   layout(%[[LAYOUT]] : !hal.executable_layout)[%[[SET_IDX]]]
  // CHECK-SAME:   set(%[[SET]] : !hal.descriptor_set)
  hal.command_buffer.bind_descriptor_set<%cmd : !hal.command_buffer>
      layout(%layout : !hal.executable_layout)[%c0]
      set(%set : !hal.descriptor_set)
  //      CHECK: hal.command_buffer.bind_descriptor_set<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   layout(%[[LAYOUT]] : !hal.executable_layout)[%[[SET_IDX]]]
  // CHECK-SAME:   set(%[[SET]] : !hal.descriptor_set)
  // CHECK-SAME:   offsets([%[[OFFSET]]])
  hal.command_buffer.bind_descriptor_set<%cmd : !hal.command_buffer>
      layout(%layout : !hal.executable_layout)[%c0]
      set(%set : !hal.descriptor_set)
      offsets([%offset])
  return
}

// -----

hal.executable @ex {
  hal.executable.variant @backend, target="backend" {
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : index
    }
  }
}

// CHECK-LABEL: @command_buffer_dispatch
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index)
func @command_buffer_dispatch(
    %cmd: !hal.command_buffer,
    %x: index,
    %y: index,
    %z: index
  ) {
  //      CHECK: hal.command_buffer.dispatch.symbol<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   target(@ex::@backend::@entry0)
  // CHECK-SAME:   workgroups([%[[X]], %[[Y]], %[[Z]]])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer>
      target(@ex::@backend::@entry0)
      workgroups([%x, %y, %z])
  return
}

// -----

hal.executable @ex {
  hal.executable.variant @backend, target="backend" {
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : index
    }
  }
}

// CHECK-LABEL: @command_buffer_dispatch_indirect
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME: %[[OFFSET:.+]]: index)
func @command_buffer_dispatch_indirect(
    %cmd: !hal.command_buffer,
    %buffer: !hal.buffer,
    %offset: index) {
  //      CHECK: hal.command_buffer.dispatch.indirect.symbol<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   target(@ex::@backend::@entry0)
  // CHECK-SAME:   workgroups(%[[BUFFER]] : !hal.buffer)[%[[OFFSET]]]
  hal.command_buffer.dispatch.indirect.symbol<%cmd : !hal.command_buffer>
      target(@ex::@backend::@entry0)
      workgroups(%buffer : !hal.buffer)[%offset]
  return
}
