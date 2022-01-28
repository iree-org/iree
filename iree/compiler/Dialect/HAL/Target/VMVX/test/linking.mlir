// RUN: iree-opt -split-input-file -iree-hal-link-executables %s | FileCheck %s

#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#executable_layout = #hal.executable.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @vmvx, target = #vmvx_target {
    hal.executable.entry_point @dispatch_0 ordinal(0) layout(#executable_layout)
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
  hal.executable.variant @vmvx, target = #vmvx_target {
    hal.executable.entry_point @dispatch_1 ordinal(0) layout(#executable_layout)
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
  hal.executable.variant @vmvx, target = #vmvx_target {
    hal.executable.entry_point @dispatch_2 ordinal(0) layout(#executable_layout)
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
func @basic_linking() -> () {
  %device = hal.ex.shared_device : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@vmvx::@dispatch_2) workgroups([%c1, %c1, %c1])
  return
}
util.initializer {
  %device = hal.ex.shared_device : !hal.device
  %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
  %c1 = arith.constant 1 : index
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
  hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@vmvx::@dispatch_2) workgroups([%c1, %c1, %c1])
  util.initializer.return
}

// All executables (including their interfaces and entry points) should be
// linked together into a single executable.
// CHECK-NOT: hal.executable private @dispatch_0
// CHECK-NOT: hal.executable private @dispatch_1
// CHECK-NOT: hal.executable private @dispatch_2
// CHECK:       hal.executable private @vmvx_linked {
// CHECK-NEXT:    hal.executable.variant public @vmvx_bytecode_fb, target = #executable_target_vmvx_bytecode_fb {
// CHECK-NEXT:      hal.executable.entry_point public @dispatch_0 ordinal(0)
// CHECK-NEXT:      hal.executable.entry_point public @dispatch_1 ordinal(1)
// CHECK-NEXT:      hal.executable.entry_point public @dispatch_2 ordinal(2)
// CHECK-NEXT:      module {
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
// CHECK:       func @basic_linking() {
// CHECK:         hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx_bytecode_fb::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx_bytecode_fb::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx_bytecode_fb::@dispatch_2) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    return
// CHECK-NEXT:  }
//
// CHECK:       util.initializer {
// CHECK:         hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx_bytecode_fb::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx_bytecode_fb::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx_bytecode_fb::@dispatch_2) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    util.initializer.return
// CHECK-NEXT:  }

// -----

#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#executable_layout = #hal.executable.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dispatch_0 {
  hal.executable.variant @vmvx, target = #vmvx_target {
    hal.executable.entry_point @dispatch_0 ordinal(0) layout(#executable_layout)
    builtin.module {
      vm.module @module {
        vm.rodata public @rodata_a dense<[0]> : tensor<1xi32>
        vm.rodata public @rodata_b dense<[0]> : tensor<1xi32>
        vm.rodata public @rodata_b_0 dense<[0]> : tensor<1xi32>
        vm.rodata public @rodata_c dense<[0]> : tensor<1xi32>
        vm.rodata private @rodata_d dense<[0]> : tensor<1xi32>
        vm.rodata private @rodata_e dense<[0]> : tensor<1xi32>

        %buf_a = vm.const.ref.rodata @rodata_a : !vm.buffer
        %buf_b = vm.const.ref.rodata @rodata_b : !vm.buffer
        %buf_b_0 = vm.const.ref.rodata @rodata_b_0 : !vm.buffer
        %buf_c = vm.const.ref.rodata @rodata_c : !vm.buffer
        %buf_d = vm.const.ref.rodata @rodata_d : !vm.buffer
        %buf_e = vm.const.ref.rodata @rodata_e : !vm.buffer
      }
    }
  }
}
hal.executable private @dispatch_1 {
  hal.executable.variant @vmvx, target = #vmvx_target {
    hal.executable.entry_point @dispatch_1 ordinal(0) layout(#executable_layout)
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

        %buf_b = vm.const.ref.rodata @rodata_b : !vm.buffer
        %buf_b_0 = vm.const.ref.rodata @rodata_b_0 : !vm.buffer
        %buf_d = vm.const.ref.rodata @rodata_d : !vm.buffer
        %buf_e = vm.const.ref.rodata @rodata_e : !vm.buffer
        %buf_f = vm.const.ref.rodata @rodata_f : !vm.buffer
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
// CHECK:       hal.executable private @vmvx_linked {
// CHECK:       hal.executable.variant public @vmvx_bytecode_fb, target = #executable_target_vmvx_bytecode_fb {
// CHECK:           module {
// CHECK-NEXT:        vm.module public @linked_module {
// CHECK-NEXT:          vm.rodata public @rodata_a dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_b dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_b_0 dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_c dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_d dense<0> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_e_0 dense<0> : tensor<1xi32>
// CHECK-NEXT:          %[[BUF_rodata_a:.+]] = vm.const.ref.rodata @rodata_a : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_b:.+]] = vm.const.ref.rodata @rodata_b : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_b_0:.+]] = vm.const.ref.rodata @rodata_b_0 : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_c:.+]] = vm.const.ref.rodata @rodata_c : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_d:.+]] = vm.const.ref.rodata @rodata_d : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_e_0:.+]] = vm.const.ref.rodata @rodata_e_0 : !vm.buffer
// CHECK-NEXT:          vm.rodata private @rodata_b_1 dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_b_0_0 dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata private @rodata_d_0 dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_e dense<1> : tensor<1xi32>
// CHECK-NEXT:          vm.rodata public @rodata_f dense<1> : tensor<1xi32>
// CHECK-NEXT:          %[[BUF_rodata_b_1:.+]] = vm.const.ref.rodata @rodata_b_1 : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_b_0_0:.+]] = vm.const.ref.rodata @rodata_b_0_0 : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_d_0:.+]] = vm.const.ref.rodata @rodata_d_0 : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_e:.+]] = vm.const.ref.rodata @rodata_e : !vm.buffer
// CHECK-NEXT:          %[[BUF_rodata_f:.+]] = vm.const.ref.rodata @rodata_f : !vm.buffer
