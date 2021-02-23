// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: #include "iree/vm/ops.h"
vm.module @calling_convention_test {
  // CHECK: iree_status_t calling_convention_test_no_in_no_return_impl(calling_convention_test_state_t* state) {
  vm.func @no_in_no_return() -> () {
    // CHECK-NEXT: return iree_ok_status();
    vm.return
  }

  // CHECK: iree_status_t calling_convention_test_i32_in_no_return_impl(int32_t v1, calling_convention_test_state_t* state) {
  vm.func @i32_in_no_return(%arg0 : i32) -> () {
    // CHECK-NEXT: return iree_ok_status();
    vm.return
  }

  // CHECK: iree_status_t calling_convention_test_no_in_i32_return_impl(int32_t *out0, calling_convention_test_state_t* state) {
  vm.func @no_in_i32_return() -> (i32) {
    // CHECK-NEXT: int32_t v1 = vm_const_i32(32);
    %0 = vm.const.i32 32 : i32
    // CHECK-NEXT: *out0 = v1;
    // CHECK-NEXT: return iree_ok_status();
    vm.return %0 : i32
  }

  // CHECK: iree_status_t calling_convention_test_i32_in_i32_return_impl(int32_t v1, int32_t *out0, calling_convention_test_state_t* state) {
  vm.func @i32_in_i32_return(%arg0 : i32) -> (i32) {
    // CHECK-NEXT: int32_t v2 = vm_const_i32(32);
    %0 = vm.const.i32 32 : i32
    // CHECK-NEXT: *out0 = v2;
    // CHECK-NEXT: return iree_ok_status();
    vm.return %0 : i32
  }
}
