// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: #include "iree/vm/ops.h"
vm.module @add_module {
  // CHECK: static iree_status_t add_module_add_1_impl(int32_t v1, int32_t v2, int32_t *out0, int32_t *out1, iree_vm_stack_t* stack, add_module_state_t* state) {
  vm.func @add_1(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    // CHECK-NEXT: VARIABLE DECLARATIONS
    // CHECK-NEXT: RESULTS
    // CHECK-NEXT: int32_t v3;
    // CHECK-NEXT: int32_t v4;
    // CHECK-NEXT: BASIC BLOCK ARGUMENTS
    // CHECK-NEXT: END VARIABLE DECLARATIONS
    // CHECK-NEXT: v3 = vm_add_i32(v1, v2);
    %0 = vm.add.i32 %arg0, %arg1 : i32
    // CHECK-NEXT: v4 = vm_add_i32(v3, v3);
    %1 = vm.add.i32 %0, %0 : i32
    // CHECK-NEXT: *out0 = v3;
    // CHECK-NEXT: *out1 = v4;
    // CHECK-NEXT: return iree_ok_status();
    vm.return %0, %1 : i32, i32
  }
}
