// RUN: iree-translate -iree-vm-ir-to-c-module -iree-vm-c-module-optimize=false %s | IreeFileCheck %s

vm.module @global_ops {
  // check the generated state struct
  // CHECK-LABEL: struct global_ops_state_s {
  // CHECK-NEXT: iree_allocator_t allocator;
  // CHECK-NEXT: uint8_t rwdata[8];
  // CHECK-NEXT: iree_vm_ref_t refs[0];
  // CHECK-NEXT: };

  vm.global.i32 @c42 42 : i32
  vm.global.i32 @c107_mut mutable 107 : i32

  vm.export @test_global_load_i32
  // CHECK-LABEL: iree_status_t global_ops_test_global_load_i32_impl(
  vm.func @test_global_load_i32() -> i32 {
    // CHECK-NEXT: VARIABLE DECLARATIONS
    // CHECK-NEXT: RESULTS
    // CHECK-NEXT: int32_t v1;
    // CHECK-NEXT: BASIC BLOCK ARGUMENTS
    // CHECK-NEXT: END VARIABLE DECLARATIONS
    // CHECK-NEXT: v1 = vm_global_load_i32(state->rwdata, 0);
    %value = vm.global.load.i32 @c42 : i32
    vm.return %value : i32
  }

  vm.export @test_global_store_i32
  // CHECK-LABEL: iree_status_t global_ops_test_global_store_i32_impl(
  vm.func @test_global_store_i32() -> i32 {
    // CHECK-NEXT: VARIABLE DECLARATIONS
    // CHECK-NEXT: RESULTS
    // CHECK-NEXT: int32_t v1;
    // CHECK-NEXT: int32_t v2;
    // CHECK-NEXT: BASIC BLOCK ARGUMENTS
    // CHECK-NEXT: END VARIABLE DECLARATIONS
    // CHECK-NEXT: v1 = 17;
    %c17 = vm.const.i32 17 : i32
    // CHECK-NEXT: vm_global_store_i32(state->rwdata, 4, v1);
    vm.global.store.i32 %c17, @c107_mut : i32
    // CHECK-NEXT: v2 = vm_global_load_i32(state->rwdata, 4);
    %value = vm.global.load.i32 @c107_mut : i32
    vm.return %value : i32
  }

  // check state initialization inside the alloc_state function
  // CHECK-LABEL: static iree_status_t global_ops_alloc_state(
  // CHECK: vm_global_store_i32(state->rwdata, 0, 42);
  // CHECK-NEXT: vm_global_store_i32(state->rwdata, 4, 107);
}
