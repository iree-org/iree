// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @list_alloc
  vm.func @list_alloc(%arg0: i32) {
    %list = vm.list.alloc %arg0 : (i32) -> !vm.list<i32>
    // CHECK: %[[LIST:.+]] = emitc.call "VM_ARRAY_ELEMENT_ADDRESS"() {args = [#emitc.opaque<"local_refs">, 0 : i32]} : () -> !emitc.opaque<"iree_vm_ref_t*">
    %list_dno = iree.do_not_optimize(%list) : !vm.list<i32>
    // CHECK: iree.do_not_optimize(%[[LIST]]) : !emitc.opaque<"iree_vm_ref_t*">
    vm.return
  }

  // CHECK-LABEL: @list_size
  vm.func @list_size(%arg0: i32) {
    %list = vm.list.alloc %arg0 : (i32) -> !vm.list<i32>
    // CHECK: %[[LIST:.+]] = emitc.call "VM_ARRAY_ELEMENT_ADDRESS"() {args = [#emitc.opaque<"local_refs">, 0 : i32]} : () -> !emitc.opaque<"iree_vm_ref_t*">
    %size = vm.list.size %list : (!vm.list<i32>) -> i32
    // CHECK: %[[SIZE:.+]] = emitc.call "iree_vm_list_size"(%{{.+}})
    %size_dno = iree.do_not_optimize(%size) : i32
    // CHECK: iree.do_not_optimize(%[[SIZE]]) : i32
    vm.return
  }
}

// -----

vm.module @my_module {
  vm.rodata private @byte_buffer dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-LABEL: @ref
  vm.export @ref
  vm.func @ref(%arg0: i32) {
    %buffer = vm.const.ref.rodata @byte_buffer : !vm.buffer
    // CHECK: %[[BUFFER:.+]] = emitc.call "VM_ARRAY_ELEMENT_ADDRESS"() {args = [#emitc.opaque<"local_refs">, 0 : i32]} : () -> !emitc.opaque<"iree_vm_ref_t*">
    %buffer_dno = iree.do_not_optimize(%buffer) : !vm.buffer
    // CHECK: iree.do_not_optimize(%[[BUFFER]]) : !emitc.opaque<"iree_vm_ref_t*">
    vm.return
  }
}
