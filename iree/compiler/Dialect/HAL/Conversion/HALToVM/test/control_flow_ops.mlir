// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: func @check_success
func @check_success() {
    // CHECK: %[[CODE:.+]] =
    %statusCode = constant 1 : i32
    // CHECK: vm.cond_fail %[[CODE]]
    hal.check_success %statusCode
    return
}

// -----

// CHECK-LABEL: func @check_success_with_message
func @check_success_with_message() {
    // CHECK: %[[CODE:.+]] =
    %statusCode = constant 1 : i32
    // CHECK: vm.cond_fail %[[CODE]], "failure message"
    hal.check_success %statusCode, "failure message"
    return
}
