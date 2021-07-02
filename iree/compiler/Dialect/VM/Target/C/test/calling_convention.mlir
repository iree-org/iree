// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: #include "iree/vm/ops.h"
vm.module @calling_convention_test {
  // CHECK: static iree_status_t calling_convention_test_no_in_no_return_impl(iree_vm_stack_t* stack, calling_convention_test_state_t* state) {
  vm.func @no_in_no_return() -> () {
    // CHECK-NEXT: VARIABLE DECLARATIONS
    // CHECK-NEXT: RESULTS
    // CHECK-NEXT: BASIC BLOCK ARGUMENTS
    // CHECK-NEXT: END VARIABLE DECLARATIONS
    // CHECK-NEXT: return iree_ok_status();
    vm.return
  }

  // CHECK: static iree_status_t calling_convention_test_i32_in_no_return_impl(int32_t v1, iree_vm_stack_t* stack, calling_convention_test_state_t* state) {
  vm.func @i32_in_no_return(%arg0 : i32) -> () {
    // CHECK-NEXT: VARIABLE DECLARATIONS
    // CHECK-NEXT: RESULTS
    // CHECK-NEXT: BASIC BLOCK ARGUMENTS
    // CHECK-NEXT: END VARIABLE DECLARATIONS
    // CHECK-NEXT: return iree_ok_status();
    vm.return
  }

  // CHECK: static iree_status_t calling_convention_test_no_in_i32_return_impl(int32_t *out0, iree_vm_stack_t* stack, calling_convention_test_state_t* state) {
  vm.func @no_in_i32_return() -> (i32) {
    // CHECK-NEXT: VARIABLE DECLARATIONS
    // CHECK-NEXT: RESULTS
    // CHECK-NEXT: int32_t v1;
    // CHECK-NEXT: BASIC BLOCK ARGUMENTS
    // CHECK-NEXT: END VARIABLE DECLARATIONS
    // CHECK-NEXT: v1 = 32;
    %0 = vm.const.i32 32 : i32
    // CHECK-NEXT: *out0 = v1;
    // CHECK-NEXT: return iree_ok_status();
    vm.return %0 : i32
  }

  // CHECK: static iree_status_t calling_convention_test_i32_in_i32_return_impl(int32_t v1, int32_t *out0, iree_vm_stack_t* stack, calling_convention_test_state_t* state) {
  vm.func @i32_in_i32_return(%arg0 : i32) -> (i32) {
    // CHECK-NEXT: VARIABLE DECLARATIONS
    // CHECK-NEXT: RESULTS
    // CHECK-NEXT: int32_t v2;
    // CHECK-NEXT: BASIC BLOCK ARGUMENTS
    // CHECK-NEXT: END VARIABLE DECLARATIONS
    // CHECK-NEXT: v2 = 32;
    %0 = vm.const.i32 32 : i32
    // CHECK-NEXT: *out0 = v2;
    // CHECK-NEXT: return iree_ok_status();
    vm.return %0 : i32
  }
}
