// RUN: iree-opt --split-input-file --iree-vmvx-link-executables %s | FileCheck %s

#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @vmvx target(#vmvx_target) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c1 = arith.constant 1 : i32
      hal.return %c1 : i32
    }
    hal.executable.export @dispatch_0 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      vm.module @module {
        vm.func @dispatch_0() {
          vm.return
        }
        vm.export @dispatch_0
      }
    }
  }
}
hal.executable private @dispatch_1 {
  hal.executable.variant @vmvx target(#vmvx_target) {
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %c2 = arith.constant 2 : i32
      hal.return %c2 : i32
    }
    hal.executable.export @dispatch_1 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      vm.module @module {
        vm.func @dispatch_1() {
          vm.return
        }
        vm.export @dispatch_1
      }
    }
  }
}
hal.executable private @dispatch_2 {
  hal.executable.variant @vmvx target(#vmvx_target) {
    hal.executable.export @dispatch_2 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      vm.module @module {
        vm.func @dispatch_2() {
          vm.return
        }
        vm.export @dispatch_2
      }
    }
  }
}
func.func @basic_linking() -> () attributes {
  testing.func.a = @dispatch_0,
  testing.func.b = @dispatch_0::@vmvx,
  testing.func.c = @dispatch_0::@vmvx::@dispatch_0
} {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer attributes {
    testing.op.a = @dispatch_0,
    testing.op.b = @dispatch_0::@vmvx,
    testing.op.c = @dispatch_0::@vmvx::@dispatch_0
  }
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@vmvx::@dispatch_2) workgroups([%c1, %c1, %c1])
  return
}
util.initializer {
  %c0 = arith.constant 0 : index
  %device = hal.devices.get %c0 : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@vmvx::@dispatch_2) workgroups([%c1, %c1, %c1])
  util.return
}

// All executables (including their interfaces and entry points) should be
// linked together into a single executable.
// CHECK-NOT: hal.executable private @dispatch_0
// CHECK-NOT: hal.executable private @dispatch_1
// CHECK-NOT: hal.executable private @dispatch_2
// CHECK:       hal.executable private @link_executables_linked_vmvx {
// CHECK-NEXT:    hal.executable.variant public @vmvx_bytecode_fb target(#executable_target_vmvx_bytecode_fb) {
// CHECK-NEXT:      hal.executable.constant.block(%arg0: !hal.device) -> i32 as "foo"
// CHECK-NEXT:        = arith.constant 1
//      CHECK:      hal.executable.export public @dispatch_0 ordinal(0)
//      CHECK:        hal.return %c1, %c1, %c1
//      CHECK:      hal.executable.constant.block(%arg0: !hal.device) -> i32 as "baz"
// CHECK-NEXT:        = arith.constant 2
//      CHECK:      hal.executable.export public @dispatch_1 ordinal(1)
//      CHECK:      hal.executable.export public @dispatch_2 ordinal(2)
//      CHECK:      module {
// CHECK-NEXT:        vm.module public @linked_module {
// CHECK-NEXT:          vm.func @dispatch_0() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_0
// CHECK-NEXT:          vm.func @dispatch_1() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_1
// CHECK-NEXT:          vm.func @dispatch_2() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_2
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
//
// CHECK:       func.func @basic_linking()
// CHECK:           testing.func.a = @link_executables_linked_vmvx
// CHECK-SAME:      testing.func.b = @link_executables_linked_vmvx::@vmvx_bytecode_fb
// CHECK-SAME:      testing.func.c = @link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_0
// CHECK:           testing.op.a = @link_executables_linked_vmvx
// CHECK-SAME:      testing.op.b = @link_executables_linked_vmvx::@vmvx_bytecode_fb
// CHECK-SAME:      testing.op.c = @link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_0
// CHECK:         hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_2) workgroups([%c1, %c1, %c1])
//
// CHECK:       util.initializer
// CHECK:         hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@link_executables_linked_vmvx::@vmvx_bytecode_fb::@dispatch_2) workgroups([%c1, %c1, %c1])

// -----

#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @vmvx target(#vmvx_target) {
    hal.executable.export @dispatch_0 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      vm.module @module {
        vm.rodata public @rodata_a dense<[0]> : tensor<1xi32>
        vm.rodata public @rodata_b dense<[0]> : tensor<1xi32>
        vm.rodata public @rodata_b_0 dense<[0]> : tensor<1xi32>
        vm.rodata public @rodata_c dense<[0]> : tensor<1xi32>
        vm.rodata private @rodata_d dense<[0]> : tensor<1xi32>
        vm.rodata private @rodata_e dense<[0]> : tensor<1xi32>
        vm.func public @dispatch_0() {
          %buf_a = vm.const.ref.rodata @rodata_a : !vm.buffer
          %buf_b = vm.const.ref.rodata @rodata_b : !vm.buffer
          %buf_b_0 = vm.const.ref.rodata @rodata_b_0 : !vm.buffer
          %buf_c = vm.const.ref.rodata @rodata_c : !vm.buffer
          %buf_d = vm.const.ref.rodata @rodata_d : !vm.buffer
          %buf_e = vm.const.ref.rodata @rodata_e : !vm.buffer
          vm.return
        }
      }
    }
  }
}
hal.executable private @dispatch_1 {
  hal.executable.variant @vmvx target(#vmvx_target) {
    hal.executable.export @dispatch_1 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      vm.module @module {
        // Conflict with a public symbol, this should be renamed when linked.
        vm.rodata private @rodata_b dense<[1]> : tensor<1xi32>
        // Conflict and reuses the same naming scheme for conflicts.
        vm.rodata private @rodata_b_0 dense<[1]> : tensor<1xi32>
        // Conflict with a private symbol, this should be renamed when linked.
        vm.rodata private @rodata_d dense<[1]> : tensor<1xi32>
        // Conflict with a private symbol, the other symbol should be renamed.
        vm.rodata public @rodata_e dense<[1]> : tensor<1xi32>
        // No conflict.
        vm.rodata public @rodata_f dense<[1]> : tensor<1xi32>
        vm.func public @dispatch_1() {
          %buf_b = vm.const.ref.rodata @rodata_b : !vm.buffer
          %buf_b_0 = vm.const.ref.rodata @rodata_b_0 : !vm.buffer
          %buf_d = vm.const.ref.rodata @rodata_d : !vm.buffer
          %buf_e = vm.const.ref.rodata @rodata_e : !vm.buffer
          %buf_f = vm.const.ref.rodata @rodata_f : !vm.buffer
          vm.return
        }
      }
    }
  }
}

// Public symbols should keep their names, private symbols can be renamed to
// resolve conflicts.
// References to renamed symbols should be updated.
//
// CHECK-NOT: hal.executable private @dispatch_0
// CHECK-NOT: hal.executable private @dispatch_1
// CHECK:       hal.executable private @link_executables_linked_vmvx {
// CHECK:       hal.executable.variant public @vmvx_bytecode_fb target(#executable_target_vmvx_bytecode_fb) {
// CHECK:           module {
// CHECK-NEXT:        vm.module public @linked_module {
// CHECK-NEXT:          vm.rodata public @rodata_a dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_b dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_b_0 dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_c dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_d dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_e_0 dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.func public @dispatch_0() {
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_a : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_b : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_b_0 : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_c : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_d : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_e_0 : !vm.buffer
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.rodata private @rodata_b_1 dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_b_0_0 dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_d_0 dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_e dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_f dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.func public @dispatch_1() {
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_b_1 : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_b_0_0 : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_d_0 : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_e : !vm.buffer
// CHECK-NEXT:            = vm.const.ref.rodata @rodata_f : !vm.buffer
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
