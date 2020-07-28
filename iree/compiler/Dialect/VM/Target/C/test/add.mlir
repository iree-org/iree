// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: #include "vm_c_funcs.h"
vm.module @add_module {
  // CHECK: iree_status_t test_function(int32_t v1, int32_t v2, int32_t *out_0, int32_t *out_1) {
  vm.func @test_function(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    // CHECK-NEXT: int32_t v3 = vm_add_i32(v1, v2);
    %0 = vm.add.i32 %arg0, %arg1 : i32
    // CHECK-NEXT: int32_t v4 = vm_add_i32(v3, v3);
    %1 = vm.add.i32 %0, %0 : i32
    // CHECK-NEXT: *out0 = v3;
    // CHECK-NEXT: *out1 = v4;
    // CHECK-NEXT: return IREE_STATUS_OK;
    vm.return %0, %1 : i32, i32
  }
}
