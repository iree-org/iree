// RUN: iree-opt -split-input-file %s -verify-diagnostics

// -----

vm.module @module {
  vm.func @primitive_ref_list_type_mismatch(%arg0 : !vm.list<!vm.ref<?>>) {
    %c100 = vm.const.i32 100 : i32
    // expected-error @+1 {{but got '!vm.list<!vm.ref<?>>}}
    %1 = vm.list.get.i32 %arg0, %c100 : (!vm.list<!vm.ref<?>>, i32) -> !vm.ref<?>
    vm.return
  }
}

// -----

vm.module @module {
  vm.func @ref_primitive_list_type_mismatch(%arg0 : !vm.list<i32>) {
    %c100 = vm.const.i32 100 : i32
    // expected-error @+1 {{cannot convert between list type 'i32' and result type '!vm.ref<?>'}}
    %1 = vm.list.get.ref %arg0, %c100 : (!vm.list<i32>, i32) -> !vm.ref<?>
    vm.return
  }
}

// -----

vm.module @module {
  vm.func @strongly_typed_ref_type_mismatch(%arg0 : !vm.list<!vm.ref<!iree.byte_buffer>>) {
    %c100 = vm.const.i32 100 : i32
    // expected-error @+1 {{cannot be accessed as '!vm.ref<!iree.mutable_byte_buffer>'}}
    %1 = vm.list.get.ref %arg0, %c100 : (!vm.list<!vm.ref<!iree.byte_buffer>>, i32) -> !vm.ref<!iree.mutable_byte_buffer>
    vm.return
  }
}
