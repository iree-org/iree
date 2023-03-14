// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm %s | FileCheck %s

// CHECK: vm.rodata private @exe_binary1 {alignment = 16 : i64} dense<[0, 1, 2, 3]> : vector<4xi8>
// CHECK: vm.rodata private @exe_binary2 {alignment = 16 : i64} dense<[4, 5, 6, 7]> : vector<4xi8>
hal.executable @exe {
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
func.func @executableCreate(
    // CHECK-SAME: %[[DEV:.+]]: !vm.ref<!hal.device>
    %device: !hal.device,
    // CHECK-SAME: %[[LAYOUT0:.+]]: !vm.ref<!hal.pipeline_layout>,
    %layout0: !hal.pipeline_layout,
    // CHECK-SAME: %[[LAYOUT1:.+]]: !vm.ref<!hal.pipeline_layout>
    %layout1: !hal.pipeline_layout
  ) -> (!hal.executable, !hal.executable) {

  // CHECK-DAG: %[[FORMAT1:.+]] = vm.rodata.inline "_utf8_format1_
  // CHECK-DAG: %[[BINARY1:.+]] = vm.const.ref.rodata @exe_binary1 : !vm.buffer
  // CHECK-DAG: %[[NULL1:.+]] = vm.const.ref.zero : !vm.buffer
  // CHECK: %[[EXE1:.+]] = vm.call.variadic @hal.executable.create(
  // CHECK-SAME: %[[DEV]], %[[FORMAT1]], %[[BINARY1]], %[[NULL1]], [%[[LAYOUT0]], %[[LAYOUT1]]]
  // CHECK-SAME: ) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer, !vm.buffer, !vm.ref<!hal.pipeline_layout> ...) -> !vm.ref<!hal.executable>
  %0 = hal.executable.create device(%device : !hal.device) target(@exe::@binary1) layouts([%layout0, %layout1]) : !hal.executable

  // CHECK-DAG: %[[FORMAT2:.+]] = vm.rodata.inline "_utf8_format2_
  // CHECK-DAG: %[[BINARY2:.+]] = vm.const.ref.rodata @exe_binary2 : !vm.buffer
  // CHECK-DAG: %[[NULL2:.+]] = vm.const.ref.zero : !vm.buffer
  // CHECK: %[[EXE2:.+]] = vm.call.variadic @hal.executable.create(
  // CHECK-SAME: %[[DEV]], %[[FORMAT2]], %[[BINARY2]], %[[NULL2]], [%[[LAYOUT1]], %[[LAYOUT0]]]
  // CHECK-SAME: ) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer, !vm.buffer, !vm.ref<!hal.pipeline_layout> ...) -> !vm.ref<!hal.executable>
  %1 = hal.executable.create device(%device : !hal.device) target(@exe::@binary2) layouts([%layout1, %layout0]) : !hal.executable

  // CHECK: vm.return %[[EXE1]], %[[EXE2]]
  return %0, %1 : !hal.executable, !hal.executable
}

// -----

// CHECK: vm.rodata private @exe1_binary1 {alignment = 16 : i64} dense<[0, 1, 2, 3]> : vector<4xi8>
hal.executable @exe1 {
  hal.executable.binary @binary1 attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = "format"
  }
}
// CHECK: vm.rodata private @exe2_binary2 {alignment = 16 : i64} dense<[4, 5, 6, 7]> : vector<4xi8>
hal.executable @exe2 {
  hal.executable.binary @binary2 attributes {
    data = dense<[4, 5, 6, 7]> : vector<4xi8>,
    format = "format"
  }
}

// CHECK-LABEL: @multipleExecutables
func.func @multipleExecutables(
    %device: !hal.device,
    %layout0: !hal.pipeline_layout,
    %layout1: !hal.pipeline_layout
  ) -> (!hal.executable, !hal.executable) {
  // CHECK-DAG: %[[FORMAT1:.+]] = vm.rodata.inline "_utf8_format_
  // CHECK-DAG: %[[BINARY1:.+]] = vm.const.ref.rodata @exe1_binary1 : !vm.buffer
  %0 = hal.executable.create device(%device : !hal.device) target(@exe1::@binary1) layouts([%layout0, %layout1]) : !hal.executable
  // CHECK-DAG: %[[FORMAT2:.+]] = vm.rodata.inline "_utf8_format_
  // CHECK-DAG: %[[BINARY2:.+]] = vm.const.ref.rodata @exe2_binary2 : !vm.buffer
  %1 = hal.executable.create device(%device : !hal.device) target(@exe2::@binary2) layouts([%layout1, %layout0]) : !hal.executable
  return %0, %1 : !hal.executable, !hal.executable
}

// -----

// CHECK: vm.rodata private @exe_binary {alignment = 16 : i64} dense<[0, 1, 2, 3]> : vector<4xi8>
hal.executable @exe {
  hal.executable.binary @binary attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = "format"
  }
}

// CHECK-LABEL: @executableConstants
func.func @executableConstants(
    // CHECK-SAME: %[[DEV:.+]]: !vm.ref<!hal.device>
    %device: !hal.device,
    // CHECK-SAME: %[[LAYOUT:.+]]: !vm.ref<!hal.pipeline_layout>
    %layout: !hal.pipeline_layout,
    // CHECK-SAME: %[[CONSTANT0:.+]]: i32, %[[CONSTANT1:.+]]: i32
    %constant0: i32, %constant1: i32
  ) -> !hal.executable {
  // CHECK-DAG: %[[FORMAT:.+]] = vm.rodata.inline "_utf8_format_
  // CHECK-DAG: %[[BINARY:.+]] = vm.const.ref.rodata @exe_binary : !vm.buffer

  // CHECK: %[[CONSTANTS:.+]] = vm.buffer.alloc %c12, %c16 : !vm.buffer
  // CHECK-DAG: %[[INDEX0:.+]] = vm.const.i64 0
  // CHECK-DAG: vm.buffer.store.i32 %[[CONSTANT0]], %[[CONSTANTS]][%[[INDEX0]]] : i32 -> !vm.buffer

  // NOTE: there is no INDEX1 as the value is constant zero and is elided.
  %c0 = arith.constant 0 : i32

  // CHECK-DAG: %[[INDEX2:.+]] = vm.const.i64 2
  // CHECK-DAG: vm.buffer.store.i32 %[[CONSTANT1]], %[[CONSTANTS]][%[[INDEX2]]] : i32 -> !vm.buffer

  // CHECK: %[[EXE:.+]] = vm.call.variadic @hal.executable.create(
  // CHECK-SAME: %[[DEV]], %[[FORMAT]], %[[BINARY]], %[[CONSTANTS]], [%[[LAYOUT]]]
  // CHECK-SAME: ) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer, !vm.buffer, !vm.ref<!hal.pipeline_layout> ...) -> !vm.ref<!hal.executable>
  %0 = hal.executable.create
      device(%device : !hal.device)
      target(@exe::@binary)
      layouts([%layout])
      constants([%constant0, %c0, %constant1]) : !hal.executable

  // CHECK: vm.return %[[EXE]]
  return %0 : !hal.executable
}
