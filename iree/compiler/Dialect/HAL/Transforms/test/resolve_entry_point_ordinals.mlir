// RUN: iree-opt -split-input-file -iree-hal-resolve-entry-point-ordinals %s | IreeFileCheck %s

// CHECK: module {
module {
  hal.executable @exe {
    hal.interface @interface {
      hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
    }
    hal.executable.target @target, "target" {
      hal.executable.entry_point @entry attributes {
        interface = @interface,
        ordinal = 0 : i32,
        signature = (tensor<4xf32>) -> tensor<4xf32>,
        workgroup_size = [32 : index, 1 : index, 1 : index]
      }
    }
  }

  func @dispatch_with_nested_references() {
    %cmd = "test_hal.command_buffer"() : () -> !hal.command_buffer
    %exe = "test_hal.executable"() : () -> !hal.executable
    %x = "test_hal.workgroup_x"() : () -> index
    %y = "test_hal.workgroup_y"() : () -> index
    %z = "test_hal.workgroup_z"() : () -> index
    // CHECK: hal.command_buffer.dispatch %0, %1, entry_point = 0, workgroup_xyz = [%2, %3, %4]
    hal.command_buffer.dispatch.symbol %cmd, %exe, entry_point = @exe::@target::@entry, workgroup_xyz = [%x, %y, %z]
    return
  }
}

// -----

// CHECK: module {
module {
  hal.executable @exe {
    hal.interface @interface {
      hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
    }
    hal.executable.target @target, "target" {
      hal.executable.entry_point @entry attributes {
        interface = @interface,
        ordinal = 0 : i32,
        signature = (tensor<4xf32>) -> tensor<4xf32>,
        workgroup_size = [32 : index, 1 : index, 1 : index]
      }
    }
  }

  func @dispatch_already_using_ordinals() {
    %cmd = "test_hal.command_buffer"() : () -> !hal.command_buffer
    %exe = "test_hal.executable"() : () -> !hal.executable
    %x = "test_hal.workgroup_x"() : () -> index
    %y = "test_hal.workgroup_y"() : () -> index
    %z = "test_hal.workgroup_z"() : () -> index
    // CHECK: hal.command_buffer.dispatch %0, %1, entry_point = 2, workgroup_xyz = [%2, %3, %4]
    hal.command_buffer.dispatch %cmd, %exe, entry_point = 2, workgroup_xyz = [%x, %y, %z]
    return
  }
}

// -----

// CHECK: module {
module {
  hal.executable @exe {
    hal.interface @interface {
      hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
    }
    hal.executable.target @target, "target" {
      hal.executable.entry_point @entry attributes {
        interface = @interface,
        ordinal = 0 : i32,
        signature = (tensor<4xf32>) -> tensor<4xf32>,
        workgroup_size = [32 : index, 1 : index, 1 : index]
      }
    }
  }

  func @dispatch_indirect_with_nested_references() {
    %cmd = "test_hal.command_buffer"() : () -> !hal.command_buffer
    %exe = "test_hal.executable"() : () -> !hal.executable
    %buffer = "test_hal.buffer"() : () -> !hal.buffer
    %offset = "test_hal.offset"() : () -> index
    // CHECK: hal.command_buffer.dispatch.indirect %0, %1, entry_point = 0, workgroups = %2[%3]
    hal.command_buffer.dispatch.indirect.symbol %cmd, %exe, entry_point = @exe::@target::@entry, workgroups = %buffer[%offset]
    return
  }
}

// -----

// CHECK: module {
module {
  hal.executable @exe {
    hal.interface @interface {
      hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
    }
    hal.executable.target @target, "target" {
      hal.executable.entry_point @entry attributes {
        interface = @interface,
        ordinal = 0 : i32,
        signature = (tensor<4xf32>) -> tensor<4xf32>,
        workgroup_size = [32 : index, 1 : index, 1 : index]
      }
    }
  }

  func @dispatch_indirect_already_using_ordinals() {
    %cmd = "test_hal.command_buffer"() : () -> !hal.command_buffer
    %exe = "test_hal.executable"() : () -> !hal.executable
    %buffer = "test_hal.buffer"() : () -> !hal.buffer
    %offset = "test_hal.offset"() : () -> index
    // CHECK: hal.command_buffer.dispatch.indirect %0, %1, entry_point = 0, workgroups = %2[%3]
    hal.command_buffer.dispatch.indirect %cmd, %exe, entry_point = 0, workgroups = %buffer[%offset]
    return
  }
}
