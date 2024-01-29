// RUN: iree-compile --split-input-file --compile-mode=vm --output-format=vm-c --iree-vm-c-module-optimize=false %s | FileCheck %s

vm.module @rodata_ops {
  // Check the generated arrays

  // CHECK: iree_alignas(16) static const uint8_t rodata_ops_buffer_1[] = {1, 2, 3};
  // CHECK-NEXT: iree_alignas(16) static const uint8_t rodata_ops_buffer_2[] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0};

  // Check the generated state struct
  // CHECK-LABEL: struct rodata_ops_state_t {
  // CHECK-SAME: iree_allocator_t allocator;
  // CHECK-SAME: uint8_t rwdata[1];
  // CHECK-SAME: iree_vm_ref_t refs[1];
  // CHECK-SAME: iree_vm_buffer_t rodata_buffers[2];
  // CHECK-SAME: iree_vm_function_t imports[1];
  // CHECK-SAME: };

  // We mark the rodata ops public in this test to explicitly prevent DCE from
  // deleting them.
  vm.rodata public @buffer_1 dense<[1, 2, 3]> : tensor<3xi8>
  vm.rodata public @buffer_2 dense<[1, 2, 3]> : tensor<3xi32>

  // check state initialization inside the alloc_state function
  // CHECK-LABEL: static iree_status_t rodata_ops_alloc_state(
  // CHECK: [[STATE:[^ ]*]] = NULL;

  // CHECK: [[BUFFER_PTR_1:[^ ]*]] = rodata_ops_buffer_1;
  // CHECK-NEXT: [[VOID_PTR_1:[^ ]*]] = (void*) [[BUFFER_PTR_1]];
  // CHECK-NEXT: [[SIZE_1:[^ ]*]] = sizeof(rodata_ops_buffer_1);
  // CHECK-NEXT: [[BYTE_SPAN_1:[^ ]*]] = iree_make_byte_span([[VOID_PTR_1]], [[SIZE_1]]);
  // CHECK-NEXT: [[ALLOCATOR_1:[^ ]*]] = iree_allocator_null();
  // CHECK-NEXT: [[BUFFERS_1:[^ ]*]] = EMITC_STRUCT_PTR_MEMBER([[STATE]], rodata_buffers);
  // CHECK-NEXT: [[BUFFER_1:[^ ]*]] = EMITC_ARRAY_ELEMENT_ADDRESS([[BUFFERS_1]], 0);
  // CHECK-NEXT: iree_vm_buffer_initialize(IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE, [[BYTE_SPAN_1]], [[ALLOCATOR_1]], [[BUFFER_1]]);

  // CHECK: [[BUFFER_PTR_2:[^ ]*]] = rodata_ops_buffer_2;
  // CHECK-NEXT: [[VOID_PTR_2:[^ ]*]] = (void*) [[BUFFER_PTR_2]];
  // CHECK-NEXT: [[SIZE_2:[^ ]*]] = sizeof(rodata_ops_buffer_2);
  // CHECK-NEXT: [[BYTE_SPAN_2:[^ ]*]] = iree_make_byte_span([[VOID_PTR_2]], [[SIZE_2]]);
  // CHECK-NEXT: [[ALLOCATOR_2:[^ ]*]] = iree_allocator_null();
  // CHECK-NEXT: [[BUFFERS_2:[^ ]*]] = EMITC_STRUCT_PTR_MEMBER([[STATE]], rodata_buffers);
  // CHECK-NEXT: [[BUFFER_2:[^ ]*]] = EMITC_ARRAY_ELEMENT_ADDRESS([[BUFFERS_2]], 1);
  // CHECK-NEXT: iree_vm_buffer_initialize(IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE, [[BYTE_SPAN_2]], [[ALLOCATOR_2]], [[BUFFER_2]]);
}

// -----

vm.module @constant_ops {
  // CHECK: constant_ops_neg_constant([[ARGS:[^)]*]]) {
  vm.func @neg_constant() -> i32 {
    // CHECK: int32_t [[CONST:[^ ]*]];
    // CHECK-NOT: [[CONST]] = 4294967292;
    // CHECK: [[CONST]] = -4;
    %c-4_i32 = vm.const.i32 -4
    vm.return %c-4_i32 : i32
  }
}
