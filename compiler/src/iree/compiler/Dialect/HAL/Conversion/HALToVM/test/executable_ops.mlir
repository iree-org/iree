// RUN: iree-opt --split-input-file --iree-vm-conversion %s | FileCheck %s

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
util.func public @executableCreate(
  // CHECK-SAME: %[[DEVICE:.+]]: !vm.ref<!hal.device>,
  %device: !hal.device,
  // CHECK-SAME: %[[AFFINITY:.+]]: i64
  %affinity: i64
) -> (!hal.executable, !hal.executable) {

  // CHECK-DAG: %[[FORMAT1:.+]] = vm.rodata.inline "_utf8_format1_
  // CHECK-DAG: %[[BINARY1:.+]] = vm.rodata.inline "exe_binary1" {alignment = 16 : i64} : !vm.buffer = dense<[0, 1, 2, 3]> : vector<4xi8>
  // CHECK-DAG: %[[NULL1:.+]] = vm.const.ref.zero : !vm.buffer
  // CHECK: %[[EXE1:.+]] = vm.call @hal.executable.create(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]], %[[FORMAT1]], %[[BINARY1]], %[[NULL1]]
  // CHECK-SAME: ) {nosideeffects} : (!vm.ref<!hal.device>, i64, !vm.buffer, !vm.buffer, !vm.buffer) -> !vm.ref<!hal.executable>
  %0 = hal.executable.create device(%device : !hal.device) affinity(%affinity) target(@exe::@binary1) : !hal.executable

  // CHECK-DAG: %[[FORMAT2:.+]] = vm.rodata.inline "_utf8_format2_
  // CHECK-DAG: %[[BINARY2:.+]] = vm.rodata.inline "exe_binary2" {alignment = 16 : i64} : !vm.buffer = dense<[4, 5, 6, 7]> : vector<4xi8>
  // CHECK-DAG: %[[NULL2:.+]] = vm.const.ref.zero : !vm.buffer
  // CHECK: %[[EXE2:.+]] = vm.call @hal.executable.create(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]], %[[FORMAT2]], %[[BINARY2]], %[[NULL2]]
  %1 = hal.executable.create device(%device : !hal.device) affinity(%affinity) target(@exe::@binary2) : !hal.executable

  // CHECK: vm.return %[[EXE1]], %[[EXE2]]
  util.return %0, %1 : !hal.executable, !hal.executable
}

// -----

hal.executable @exe1 {
  hal.executable.binary @binary1 attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = "format"
  }
}
hal.executable @exe2 {
  hal.executable.binary @binary2 attributes {
    data = dense<[4, 5, 6, 7]> : vector<4xi8>,
    format = "format"
  }
}

// CHECK-LABEL: @multipleExecutables
util.func public @multipleExecutables(%device: !hal.device, %affinity: i64) -> (!hal.executable, !hal.executable) {
  // CHECK-DAG: %[[FORMAT1:.+]] = vm.rodata.inline "_utf8_format_
  // CHECK-DAG: %[[BINARY1:.+]] = vm.rodata.inline "exe1_binary1" {alignment = 16 : i64} : !vm.buffer = dense<[0, 1, 2, 3]> : vector<4xi8>
  %0 = hal.executable.create device(%device : !hal.device) affinity(%affinity) target(@exe1::@binary1) : !hal.executable
  // CHECK-DAG: %[[FORMAT2:.+]] = vm.rodata.inline "_utf8_format_
  // CHECK-DAG: %[[BINARY2:.+]] = vm.rodata.inline "exe2_binary2" {alignment = 16 : i64} : !vm.buffer = dense<[4, 5, 6, 7]> : vector<4xi8>
  %1 = hal.executable.create device(%device : !hal.device) affinity(%affinity) target(@exe2::@binary2) : !hal.executable
  util.return %0, %1 : !hal.executable, !hal.executable
}

// -----

hal.executable @exe {
  hal.executable.binary @binary attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = "format"
  }
}

// CHECK-LABEL: @executableConstants
util.func public @executableConstants(
    // CHECK-SAME: %[[DEVICE:.+]]: !vm.ref<!hal.device>
    %device: !hal.device,
    // CHECK-SAME: %[[AFFINITY:.+]]: i64,
    %affinity: i64,
    // CHECK-SAME: %[[CONSTANT0:.+]]: i32, %[[CONSTANT1:.+]]: i32
    %constant0: i32, %constant1: i32
  ) -> !hal.executable {
  // CHECK-DAG: %[[FORMAT:.+]] = vm.rodata.inline "_utf8_format_
  // CHECK-DAG: %[[BINARY:.+]] = vm.rodata.inline "exe_binary" {alignment = 16 : i64} : !vm.buffer = dense<[0, 1, 2, 3]> : vector<4xi8>

  // CHECK: %[[CONSTANTS:.+]] = vm.buffer.alloc %c12, %c16 : !vm.buffer
  // CHECK-DAG: %[[INDEX0:.+]] = vm.const.i64 0
  // CHECK-DAG: vm.buffer.store.i32 %[[CONSTANT0]], %[[CONSTANTS]][%[[INDEX0]]] : i32 -> !vm.buffer

  // NOTE: there is no INDEX1 as the value is constant zero and is elided.
  %c0 = arith.constant 0 : i32

  // CHECK-DAG: %[[INDEX2:.+]] = vm.const.i64 2
  // CHECK-DAG: vm.buffer.store.i32 %[[CONSTANT1]], %[[CONSTANTS]][%[[INDEX2]]] : i32 -> !vm.buffer

  // CHECK: %[[EXE:.+]] = vm.call @hal.executable.create(
  // CHECK-SAME: %[[DEVICE]], %[[AFFINITY]], %[[FORMAT]], %[[BINARY]], %[[CONSTANTS]]
  // CHECK-SAME: ) {nosideeffects} : (!vm.ref<!hal.device>, i64, !vm.buffer, !vm.buffer, !vm.buffer) -> !vm.ref<!hal.executable>
  %0 = hal.executable.create
      device(%device : !hal.device)
      affinity(%affinity)
      target(@exe::@binary)
      constants([%constant0, %c0, %constant1]) : !hal.executable

  // CHECK: vm.return %[[EXE]]
  util.return %0 : !hal.executable
}
