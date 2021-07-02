// RUN: iree-translate -iree-vm-ir-to-c-module -iree-vm-c-module-optimize=false %s | IreeFileCheck %s

vm.module @constant_ops {
  // Check the generated arrays

  // CHECK: iree_alignas(16) static const uint8_t constant_ops_buffer_1[] = {1, 2, 3};
  // CHECK-NEXT: iree_alignas(16) static const uint8_t constant_ops_buffer_2[] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0};

  // Check the generated state struct
  // CHECK-LABEL: struct constant_ops_state_t {
  // CHECK-NEXT: iree_allocator_t allocator;
  // CHECK-NEXT: uint8_t rwdata[0];
  // CHECK-NEXT: iree_vm_ref_t refs[0];
  // CHECK-NEXT: iree_vm_buffer_t rodata_buffers[2];
  // CHECK-NEXT: iree_vm_function_t imports[0];
  // CHECK-NEXT: };

  // We mark the rodata ops public in this test to explicitly prevent DCE from
  // deleting them.
  vm.rodata public @buffer_1 dense<[1, 2, 3]> : tensor<3xi8>
  vm.rodata public @buffer_2 dense<[1, 2, 3]> : tensor<3xi32>

  // check state initialization inside the alloc_state function
  // CHECK-LABEL: static iree_status_t constant_ops_alloc_state(
  // CHECK: iree_vm_buffer_initialize(IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE, iree_make_byte_span((void*)constant_ops_buffer_1, sizeof(constant_ops_buffer_1)), iree_allocator_null(), &state->rodata_buffers[0]);
  // CHECK-NEXT: iree_vm_buffer_initialize(IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE, iree_make_byte_span((void*)constant_ops_buffer_2, sizeof(constant_ops_buffer_2)), iree_allocator_null(), &state->rodata_buffers[1]);
}
