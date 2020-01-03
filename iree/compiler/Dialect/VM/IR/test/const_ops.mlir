// Tests printing and parsing of constant definition ops.

// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @const_i32_zero
  vm.func @const_i32_zero() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    %zero = vm.const.i32.zero : i32
    vm.return %zero : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @const_i32
  vm.func @const_i32() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    %c1 = vm.const.i32 1 : i32
    vm.return %c1 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @const_ref_zero
  vm.func @const_ref_zero() -> !iree.opaque_ref {
    // CHECK: %null = vm.const.ref.zero : !iree.opaque_ref
    %null = vm.const.ref.zero : !iree.opaque_ref
    vm.return %null : !iree.opaque_ref
  }
}

// -----

vm.module @my_module {
  vm.rodata @buf0 dense<[0, 1, 2]> : tensor<3xi8>
  // CHECK-LABEL: @const_ref_rodata
  vm.func @const_ref_rodata() -> !iree.byte_buffer_ref {
    // CHECK: %buf0 = vm.const.ref.rodata @buf0 : !iree.byte_buffer_ref
    %buf0 = vm.const.ref.rodata @buf0 : !iree.byte_buffer_ref
    vm.return %buf0 : !iree.byte_buffer_ref
  }
}
