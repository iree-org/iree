// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @executableLoad
// CHECK-SAME: (%[[EXECUTABLE_DATA:.+]]: !vm.buffer)
util.func public @executableLoad(%executable_data: !util.buffer) -> !hal.executable {
  // CHECK-DAG: %[[CONSTANTS:.+]] = vm.const.ref.zero : !vm.buffer
  // CHECK-DAG: %[[FORMAT_STR:.+]] = vm.rodata.inline {{.+}} : !vm.buffer = "executable_format"
  // CHECK: %[[EXECUTABLE:.+]] = vm.call @hal_loader.executable.load(%[[FORMAT_STR]], %[[EXECUTABLE_DATA]], %[[CONSTANTS]])
  %executable = hal_loader.executable.load format("executable_format") data(%executable_data) : !hal.executable
  // CHECK: return %[[EXECUTABLE]]
  util.return %executable : !hal.executable
}

// -----

// CHECK-LABEL: @executableDispatch
// CHECK-SAME: (%[[EXECUTABLE:.+]]: !vm.ref<!hal.executable>,
// CHECK-SAME:  %[[BUFFER0:.+]]: !vm.buffer, %[[BUFFER1:.+]]: !vm.buffer)
util.func public @executableDispatch(%executable: !hal.executable, %buffer0: !util.buffer, %buffer1: !util.buffer) {
  // CHECK-DAG: %[[COUNT_X:.+]] = vm.const.i32 1000
  %count_x = arith.constant 1000 : index
  // CHECK-DAG: %[[COUNT_Y:.+]] = vm.const.i32 1001
  %count_y = arith.constant 1001 : index
  // CHECK-DAG: %[[COUNT_Z:.+]] = vm.const.i32 1002
  %count_z = arith.constant 1002 : index
  // CHECK-DAG: %[[CONSTANT0:.+]] = vm.const.i32 4
  %constant0 = arith.constant 4 : i32
  // CHECK-DAG: %[[CONSTANT1:.+]] = vm.const.i32 5
  %constant1 = arith.constant 5 : i32
  // CHECK-DAG: %[[OFFSET0:.+]] = vm.const.i64 100
  %offset0 = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH0:.+]] = vm.const.i64 128
  %length0 = arith.constant 128 : index
  // CHECK-DAG: %[[OFFSET1:.+]] = vm.const.i64 200
  %offset1 = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH1:.+]] = vm.const.i64 256
  %length1 = arith.constant 256 : index
  // CHECK: vm.call.variadic @hal_loader.executable.dispatch
  hal_loader.executable.dispatch
    // CHECK-SAME: %[[EXECUTABLE]], %c16
    executable(%executable : !hal.executable)[16]
    // CHECK-SAME: %[[COUNT_X]], %[[COUNT_Y]], %[[COUNT_Z]]
    workgroups([%count_x, %count_y, %count_z])
    // CHECK-SAME: [%[[CONSTANT0]], %[[CONSTANT1]]]
    constants([%constant0, %constant1])
    bindings([
      // CHECK-SAME: (%[[BUFFER0]], %[[OFFSET0]], %[[LENGTH0]])
      (%buffer0 : !util.buffer)[%offset0, %length0],
      // CHECK-SAME: (%[[BUFFER1]], %[[OFFSET1]], %[[LENGTH1]])
      (%buffer1 : !util.buffer)[%offset1, %length1]
    ])
  util.return
}
