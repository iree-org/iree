// RUN: iree-opt --split-input-file %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @const_i32_zero
  vm.func @const_i32_zero() -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    %zero = vm.const.i32.zero
    vm.return %zero : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @const_i32
  vm.func @const_i32() -> i32 {
    // CHECK: %c1 = vm.const.i32 1
    %c1 = vm.const.i32 1
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
  // CHECK: vm.rodata private @buf0 {alignment = 8 : i64} dense<[0, 1, 2]> : tensor<3xi8>
  vm.rodata private @buf0 {alignment = 8 : i64} dense<[0, 1, 2]> : tensor<3xi8>
  // CHECK-LABEL: @const_ref_rodata
  vm.func @const_ref_rodata() -> !vm.buffer {
    // CHECK: %buf0 = vm.const.ref.rodata @buf0 : !vm.buffer
    %buf0 = vm.const.ref.rodata @buf0 : !vm.buffer
    vm.return %buf0 : !vm.buffer
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @inlined_rodata
  vm.func @inlined_rodata() -> !vm.buffer {
    // CHECK-NEXT: = vm.rodata.inline : !vm.buffer = dense<[0, 1, 2]> : tensor<3xi8>
    %0 = vm.rodata.inline : !vm.buffer = dense<[0, 1, 2]> : tensor<3xi8>
    vm.return %0 : !vm.buffer
  }
}

// -----

// CHECK: #[[$DATA:.+]] = #util.composite<10xi8, [
#table_data = #util.composite<10xi8, [
  dense<[2, 3]> : vector<2xi8>,
  "hello",
  dense<4> : tensor<3xi8>
]>

vm.module @my_module {
  // CHECK-LABEL: @rodata_table
  vm.func @rodata_table() -> !vm.buffer {
    // CHECK-NEXT: = vm.rodata.table : !vm.buffer, !vm.buffer = #[[$DATA]]
    %0:2 = vm.rodata.table : !vm.buffer, !vm.buffer = #table_data
    vm.return %0#1 : !vm.buffer
  }
}
