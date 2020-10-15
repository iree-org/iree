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
  vm.func @const_ref_zero() -> !vm.ref<?> {
    // CHECK: %null = vm.const.ref.zero : !vm.ref<?>
    %null = vm.const.ref.zero : !vm.ref<?>
    vm.return %null : !vm.ref<?>
  }
}

// -----

vm.module @my_module {
  vm.rodata @buf0 dense<[0, 1, 2]> : tensor<3xi8>
  // CHECK-LABEL: @const_ref_rodata
  vm.func @const_ref_rodata() -> !vm.ref<!iree.byte_buffer> {
    // CHECK: %buf0 = vm.const.ref.rodata @buf0 : !vm.ref<!iree.byte_buffer>
    %buf0 = vm.const.ref.rodata @buf0 : !vm.ref<!iree.byte_buffer>
    vm.return %buf0 : !vm.ref<!iree.byte_buffer>
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @inlined_rodata
  vm.func @inlined_rodata() -> !vm.ref<!iree.byte_buffer> {
    // CHECK-NEXT: = vm.rodata.inline : !vm.ref<!iree.byte_buffer> = dense<[0, 1, 2]> : tensor<3xi8>
    %0 = vm.rodata.inline : !vm.ref<!iree.byte_buffer> = dense<[0, 1, 2]> : tensor<3xi8>
    vm.return %0 : !vm.ref<!iree.byte_buffer>
  }
}
