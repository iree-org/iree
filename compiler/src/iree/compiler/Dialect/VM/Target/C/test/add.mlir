// RUN: iree-compile --compile-mode=vm --output-format=vm-c %s | FileCheck %s

// CHECK: #include "iree/vm/ops.h"
vm.module @add_module {
  // TODO(simon-camp): Add back check for static modifier
  // CHECK: iree_status_t add_module_add_1(iree_vm_stack_t* v1, struct add_module_t* v2, struct add_module_state_t* v3, int32_t v4, int32_t v5, int32_t* v6, int32_t* v7) {
  vm.func @add_1(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    // CHECK-NEXT: int32_t v8;
    // CHECK-NEXT: int32_t v9;
    // CHECK-NEXT: iree_status_t v10;
    // CHECK-NEXT: v8 = vm_add_i32(v4, v5);
    %0 = vm.add.i32 %arg0, %arg1 : i32
    // CHECK-NEXT: v9 = vm_add_i32(v8, v8);
    %1 = vm.add.i32 %0, %0 : i32
    // CHECK-NEXT: EMITC_DEREF_ASSIGN_VALUE(v6, v8);
    // CHECK-NEXT: EMITC_DEREF_ASSIGN_VALUE(v7, v9);
    // CHECK-NEXT: v10 = iree_ok_status();
    // CHECK-NEXT: return v10;
    vm.return %0, %1 : i32, i32
  }
}
