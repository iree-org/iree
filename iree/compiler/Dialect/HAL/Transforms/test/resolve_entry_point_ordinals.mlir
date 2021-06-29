// RUN: iree-opt -split-input-file -iree-hal-resolve-entry-point-ordinals %s | IreeFileCheck %s

hal.executable @exe {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @target, filter="target" {
    hal.executable.entry_point @entry attributes {
      interface = @interface,
      ordinal = 0 : index,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
  }
}

// CHECK-LABEL: @dispatch_with_nested_references
// CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
func @dispatch_with_nested_references(%cmd : !hal.command_buffer) {
  %c10 = constant 10 : index
  %c11 = constant 11 : index
  %c12 = constant 12 : index
  //      CHECK: %[[DEVICE:.+]] = hal.command_buffer.device<%[[CMD]]
  //      CHECK: %[[EXE:.+]] = hal.executable.lookup
  // CHECK-SAME:   device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:   executable(@exe) : !hal.executable
  //      CHECK: hal.command_buffer.dispatch<%[[CMD]]
  // CHECK-SAME:   target(%[[EXE]] : !hal.executable)[0]
  // CHECK-SAME:   workgroups([%c10, %c11, %c12])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer>
      target(@exe::@target::@entry)
      workgroups([%c10, %c11, %c12])
  return
}

// -----

// CHECK-LABEL: @dispatch_already_using_ordinals
func @dispatch_already_using_ordinals(
  // CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
  %cmd: !hal.command_buffer,
  // CHECK-SAME: %[[EXE:.+]]: !hal.executable
  %exe: !hal.executable
) {
  %c10 = constant 10 : index
  %c11 = constant 11 : index
  %c12 = constant 12 : index
  //      CHECK: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   target(%[[EXE]] : !hal.executable)[2]
  // CHECK-SAME:   workgroups([%c10, %c11, %c12])
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
      target(%exe : !hal.executable)[2]
      workgroups([%c10, %c11, %c12])
  return
}

// -----

hal.executable @exe {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @target, filter="target" {
    hal.executable.entry_point @entry attributes {
      interface = @interface,
      ordinal = 0 : index,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
  }
}

// CHECK-LABEL: @dispatch_indirect_with_nested_references
func @dispatch_indirect_with_nested_references(
  // CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
  %cmd: !hal.command_buffer,
  // CHECK-SAME: %[[BUF:.+]]: !hal.buffer
  %buf: !hal.buffer
) {
  %c10 = constant 10 : index
  // CHECK: %[[DEVICE:.+]] = hal.command_buffer.device<%[[CMD]]
  // CHECK: %[[EXE:.+]] = hal.executable.lookup device(%[[DEVICE]] : !hal.device) executable(@exe)
  // CHECK: hal.command_buffer.dispatch.indirect<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   target(%[[EXE]] : !hal.executable)[0]
  // CHECK-SAME:   workgroups(%[[BUF]] : !hal.buffer)[%c10]
  hal.command_buffer.dispatch.indirect.symbol<%cmd : !hal.command_buffer>
      target(@exe::@target::@entry)
      workgroups(%buf : !hal.buffer)[%c10]
  return
}

// -----

// CHECK-LABEL: @dispatch_indirect_already_using_ordinals
func @dispatch_indirect_already_using_ordinals(
  // CHECK-SAME: %[[CMD:.+]]: !hal.command_buffer
  %cmd: !hal.command_buffer,
  // CHECK-SAME: %[[EXE:.+]]: !hal.executable
  %exe: !hal.executable,
  // CHECK-SAME: %[[BUF:.+]]: !hal.buffer
  %buf: !hal.buffer
) {
  %c10 = constant 10 : index
  // CHECK: hal.command_buffer.dispatch.indirect<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME:   target(%[[EXE]] : !hal.executable)[0]
  // CHECK-SAME:   workgroups(%[[BUF]] : !hal.buffer)[%c10]
  hal.command_buffer.dispatch.indirect<%cmd : !hal.command_buffer>
      target(%exe : !hal.executable)[0]
      workgroups(%buf : !hal.buffer)[%c10]
  return
}
