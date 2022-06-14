// RUN: iree-opt --split-input-file --iree-vm-conversion %s | FileCheck %s

// CHECK-LABEL: vm.func private @status_check_ok
func.func @status_check_ok() {
  // CHECK: %[[CODE:.+]] =
  %statusCode = arith.constant 1 : i32
  // CHECK: vm.cond_fail %[[CODE]]
  util.status.check_ok %statusCode
  return
}

// -----

// CHECK-LABEL: vm.func private @status_check_ok_with_message
func.func @status_check_ok_with_message() {
  // CHECK: %[[CODE:.+]] =
  %statusCode = arith.constant 1 : i32
  // CHECK: vm.cond_fail %[[CODE]], "failure message"
  util.status.check_ok %statusCode, "failure message"
  return
}
