// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK: vm.rodata private @_exe_binary1_binary {alignment = 16 : i64} dense<[0, 1, 2, 3]> : vector<4xi8>
// CHECK: vm.rodata private @_exe_binary2_binary {alignment = 16 : i64} dense<[4, 5, 6, 7]> : vector<4xi8>
hal.executable @exe {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.binary @binary1 attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = "format1"
  }
  hal.executable.binary @binary2 attributes {
    data = dense<[4, 5, 6, 7]> : vector<4xi8>,
    format = "format2"
  }
}

// CHECK-LABEL: @executableCreate
func @executableCreate(
    // CHECK-SAME: %[[DEV:.+]]: !vm.ref<!hal.device>
    %device : !hal.device,
    // CHECK-SAME: %[[LAYOUT0:.+]]: !vm.ref<!hal.executable_layout>,
    %layout0 : !hal.executable_layout,
    // CHECK-SAME: %[[LAYOUT1:.+]]: !vm.ref<!hal.executable_layout>
    %layout1 : !hal.executable_layout
  ) -> (!hal.executable, !hal.executable) {

  // CHECK-DAG: %[[BINARY1:.+]] = vm.const.ref.rodata @_exe_binary1_binary : !vm.buffer
  // CHECK-DAG: %[[FORMAT1:.+]] = vm.rodata.inline "_utf8_format1_
  // CHECK: %[[EXE1:.+]] = vm.call.variadic @hal.executable.create(
  // CHECK-SAME: %[[DEV]], %[[FORMAT1]], %[[BINARY1]], [%[[LAYOUT0]], %[[LAYOUT1]]]
  // CHECK-SAME: ) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
  %0 = hal.executable.create device(%device : !hal.device) target(@exe::@binary1) layouts([%layout0, %layout1]) : !hal.executable

  // CHECK-DAG: %[[BINARY2:.+]] = vm.const.ref.rodata @_exe_binary2_binary : !vm.buffer
  // CHECK-DAG: %[[FORMAT2:.+]] = vm.rodata.inline "_utf8_format2_
  // CHECK: %[[EXE2:.+]] = vm.call.variadic @hal.executable.create(
  // CHECK-SAME: %[[DEV]], %[[FORMAT2]], %[[BINARY2]], [%[[LAYOUT1]], %[[LAYOUT0]]]
  // CHECK-SAME: ) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer, !vm.ref<!hal.executable_layout> ...) -> !vm.ref<!hal.executable>
  %1 = hal.executable.create device(%device : !hal.device) target(@exe::@binary2) layouts([%layout1, %layout0]) : !hal.executable

  // CHECK: vm.return %[[EXE1]], %[[EXE2]]
  return %0, %1 : !hal.executable, !hal.executable
}

// -----

// CHECK: vm.rodata private @_exe1_binary1_binary {alignment = 16 : i64} dense<[0, 1, 2, 3]> : vector<4xi8>
hal.executable @exe1 {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.binary @binary1 attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = "format"
  }
}
// CHECK: vm.rodata private @_exe2_binary2_binary {alignment = 16 : i64} dense<[4, 5, 6, 7]> : vector<4xi8>
hal.executable @exe2 {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.binary @binary2 attributes {
    data = dense<[4, 5, 6, 7]> : vector<4xi8>,
    format = "format"
  }
}

// CHECK-LABEL: @multipleExecutables
func @multipleExecutables(
    %device : !hal.device,
    %layout0 : !hal.executable_layout,
    %layout1 : !hal.executable_layout
  ) -> (!hal.executable, !hal.executable) {
  // CHECK-DAG: %[[BINARY1:.+]] = vm.const.ref.rodata @_exe1_binary1_binary : !vm.buffer
  // CHECK-DAG: %[[FORMAT1:.+]] = vm.rodata.inline "_utf8_format_
  %0 = hal.executable.create device(%device : !hal.device) target(@exe1::@binary1) layouts([%layout0, %layout1]) : !hal.executable
  // CHECK-DAG: %[[BINARY2:.+]] = vm.const.ref.rodata @_exe2_binary2_binary : !vm.buffer
  // CHECK-DAG: %[[FORMAT2:.+]] = vm.rodata.inline "_utf8_format_
  %1 = hal.executable.create device(%device : !hal.device) target(@exe2::@binary2) layouts([%layout1, %layout0]) : !hal.executable
  return %0, %1 : !hal.executable, !hal.executable
}
