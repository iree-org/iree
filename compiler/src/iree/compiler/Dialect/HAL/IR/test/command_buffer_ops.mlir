// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @command_buffer_create
//  CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func.func @command_buffer_create(%device: !hal.device) {
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

// CHECK-LABEL: @command_buffer_finalize
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer)
func.func @command_buffer_finalize(%cmd: !hal.command_buffer) {
  // CHECK: hal.command_buffer.finalize<%[[CMD]] : !hal.command_buffer>
  hal.command_buffer.finalize<%cmd : !hal.command_buffer>
  return
}

// -----

// CHECK-LABEL: @command_buffer_device
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer)
func.func @command_buffer_device(%cmd: !hal.command_buffer) {
  // CHECK: %0 = hal.command_buffer.device<%[[CMD]] : !hal.command_buffer> : !hal.device
  %0 = hal.command_buffer.device<%cmd : !hal.command_buffer> : !hal.device
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer)
func.func @command_buffer_execution_barrier(%cmd: !hal.command_buffer) {
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
func.func @command_buffer_fill_buffer(
    %cmd: !hal.command_buffer,
    %buffer: !hal.buffer,
    %offset: index,
    %length: index,
    %pattern: i32) {
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
func.func @command_buffer_copy_buffer(
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

// CHECK-LABEL: @command_buffer_collective
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME:  %[[CHANNEL:.+]]: !hal.channel,
//  CHECK-SAME:  %[[PARAM:.+]]: i32,
//  CHECK-SAME:  %[[SEND_BUFFER:.+]]: !hal.buffer, %[[RECV_BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME:  %[[COUNT:.+]]: index)
func.func @command_buffer_collective(
    %cmd: !hal.command_buffer,
    %channel: !hal.channel,
    %param: i32,
    %send_buffer: !hal.buffer, %recv_buffer: !hal.buffer,
    %count: index) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  //      CHECK: hal.command_buffer.collective<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   channel(%[[CHANNEL]] : !hal.channel)
  // CHECK-SAME:   op(<all_reduce with sum : f32>)
  // CHECK-SAME:   send(%[[SEND_BUFFER]] : !hal.buffer)[%c10, %c128]
  // CHECK-SAME:   recv(%[[RECV_BUFFER]] : !hal.buffer)[%c20, %c256]
  // CHECK-SAME:   count(%[[COUNT]])
  hal.command_buffer.collective<%cmd : !hal.command_buffer>
      channel(%channel : !hal.channel)
      op(<all_reduce with sum : f32>)
      send(%send_buffer : !hal.buffer)[%c10, %c128]
      recv(%recv_buffer : !hal.buffer)[%c20, %c256]
      count(%count)

  //      CHECK: hal.command_buffer.collective<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   channel(%[[CHANNEL]] : !hal.channel)
  // CHECK-SAME:   op(<send : f32>)
  // CHECK-SAME:   param(%[[PARAM]] : i32)
  // CHECK-SAME:   send(%[[SEND_BUFFER]] : !hal.buffer)[%c10, %c128]
  // CHECK-SAME:   count(%[[COUNT]])
  hal.command_buffer.collective<%cmd : !hal.command_buffer>
      channel(%channel : !hal.channel)
      op(<send : f32>)
      param(%param : i32)
      send(%send_buffer : !hal.buffer)[%c10, %c128]
      count(%count)

  //      CHECK: hal.command_buffer.collective<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   channel(%[[CHANNEL]] : !hal.channel)
  // CHECK-SAME:   op(<recv : f32>)
  // CHECK-SAME:   param(%[[PARAM]] : i32)
  // CHECK-SAME:   recv(%[[RECV_BUFFER]] : !hal.buffer)[%c20, %c128]
  // CHECK-SAME:   count(%[[COUNT]])
  hal.command_buffer.collective<%cmd : !hal.command_buffer>
      channel(%channel : !hal.channel)
      op(<recv : f32>)
      param(%param : i32)
      recv(%recv_buffer : !hal.buffer)[%c20, %c128]
      count(%count)

  return
}

// -----

// CHECK-LABEL: @command_buffer_push_descriptor_set
//  CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[LAYOUT:.+]]: !hal.pipeline_layout,
//  CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME: %[[SLOT:.+]]: index
func.func @command_buffer_push_descriptor_set(
    %cmd: !hal.command_buffer,
    %layout: !hal.pipeline_layout,
    %buffer: !hal.buffer,
    %slot: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4096 = arith.constant 4096 : index
  %c8000 = arith.constant 8000 : index
  // CHECK: hal.command_buffer.push_descriptor_set<%[[CMD]] : !hal.command_buffer>
  hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer>
      // CHECK-SAME: layout(%[[LAYOUT]] : !hal.pipeline_layout)[%c1]
      layout(%layout : !hal.pipeline_layout)[%c1]
      // CHECK-SAME: bindings([
      bindings([
        // CHECK-NEXT: %c0 = (%[[BUFFER]] : !hal.buffer)[%c4096, %c8000]
        %c0 = (%buffer : !hal.buffer)[%c4096, %c8000],
        // CHECK-NEXT: %c1 = (%[[SLOT]] : index)[%c4, %c4096]
        %c1 = (%slot : index)[%c4, %c4096]
      ])
  return
}

// -----

hal.executable @ex {
  hal.executable.variant @backend target(<"backend", "format">) {
    hal.executable.export @entry0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>,
        #hal.descriptor_set.binding<1, storage_buffer>
      ]>
    ]>)
  }
}

// CHECK-LABEL: @command_buffer_dispatch
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME: %[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index)
func.func @command_buffer_dispatch(
    %cmd: !hal.command_buffer,
    %x: index,
    %y: index,
    %z: index) {
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
  hal.executable.variant @backend target(<"backend", "format">) {
    hal.executable.export @entry0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>,
        #hal.descriptor_set.binding<1, storage_buffer>
      ]>
    ]>)
  }
}

// CHECK-LABEL: @command_buffer_dispatch_indirect
//  CHECK-SAME: (%[[CMD:.+]]: !hal.command_buffer,
//  CHECK-SAME:  %[[BUFFER:.+]]: !hal.buffer,
//  CHECK-SAME:  %[[OFFSET:.+]]: index)
func.func @command_buffer_dispatch_indirect(
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
