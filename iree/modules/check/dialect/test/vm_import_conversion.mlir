// RUN: check-opt %s -iree-convert-flow-to-hal -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @expectTrue
func @expectTruei32() {
  // CHECK: [[C:%.+]] = vm.const.i32
  %c = constant 1 : i32
  // CHECK: vm.call @check.expect_true([[C]]
  check.expect_true(%c) : i32
  return
}

// TODO(gcmn) handles other input bit-widths
