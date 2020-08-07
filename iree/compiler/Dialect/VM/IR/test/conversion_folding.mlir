// Tests folding and canonicalization of casting/conversion ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s

// CHECK-LABEL: @trunc_folds
vm.module @trunc_folds {
  // CHECK-LABEL: @trunc_i8_const
  vm.func @trunc_i8_const() -> i32 {
    // CHECK: vm.const.i32 255 : i32
    %c = vm.const.i32 0xFFFFFFFF : i32
    %0 = vm.trunc.i32.i8 %c : i32 -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @trunc_i16_const
  vm.func @trunc_i16_const() -> i32 {
    // CHECK: vm.const.i32 65535 : i32
    %c = vm.const.i32 0xFFFFFFFF : i32
    %0 = vm.trunc.i32.i16 %c : i32 -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @trunc_folds_i64
vm.module @trunc_folds_i64 {
  // CHECK-LABEL: @trunc_i8_const
  vm.func @trunc_i8_const() -> i32 {
    // CHECK: vm.const.i32 255 : i32
    %c = vm.const.i64 0xFFFFFFFFFFFFFFFF : i64
    %0 = vm.trunc.i64.i8 %c : i64 -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @trunc_i16_const
  vm.func @trunc_i16_const() -> i32 {
    // CHECK: vm.const.i32 65535 : i32
    %c = vm.const.i64 0xFFFFFFFFFFFFFFFF : i64
    %0 = vm.trunc.i64.i16 %c : i64 -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @trunc_i32_const
  vm.func @trunc_i32_const() -> i32 {
    // CHECK: vm.const.i32 -1 : i32
    %c = vm.const.i64 0xFFFFFFFFFFFFFFFF : i64
    %0 = vm.trunc.i64.i32 %c : i64 -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @ext_folds
vm.module @ext_folds {
  // CHECK-LABEL: @ext_i8_i32_s_const
  vm.func @ext_i8_i32_s_const() -> i32 {
    // CHECK: vm.const.i32 -1 : i32
    %c = vm.const.i32 0x000000FF : i32
    %0 = vm.ext.i8.i32.s %c : i32 -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @ext_i8_i32_u_const
  vm.func @ext_i8_i32_u_const() -> i32 {
    // CHECK: vm.const.i32 255 : i32
    %c = vm.const.i32 0x000000FF : i32
    %0 = vm.ext.i8.i32.u %c : i32 -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @ext_i16_i32_s_const
  vm.func @ext_i16_i32_s_const() -> i32 {
    // CHECK: vm.const.i32 -1 : i32
    %c = vm.const.i32 0x0000FFFF : i32
    %0 = vm.ext.i16.i32.s %c : i32 -> i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @ext_i16_i32_u_const
  vm.func @ext_i16_i32_u_const() -> i32 {
    // CHECK: vm.const.i32 65535 : i32
    %c = vm.const.i32 0x0000FFFF : i32
    %0 = vm.ext.i16.i32.u %c : i32 -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @ext_folds_i64
vm.module @ext_folds_i64 {
  // CHECK-LABEL: @ext_i8_i64_s_const
  vm.func @ext_i8_i64_s_const() -> i64 {
    // CHECK: vm.const.i64 -1 : i64
    %c = vm.const.i32 0x000000FF : i32
    %0 = vm.ext.i8.i64.s %c : i32 -> i64
    vm.return %0 : i64
  }

  // CHECK-LABEL: @ext_i8_i64_u_const
  vm.func @ext_i8_i64_u_const() -> i64 {
    // CHECK: vm.const.i64 255 : i64
    %c = vm.const.i32 0x000000FF : i32
    %0 = vm.ext.i8.i64.u %c : i32 -> i64
    vm.return %0 : i64
  }

  // CHECK-LABEL: @ext_i16_i64_s_const
  vm.func @ext_i16_i64_s_const() -> i64 {
    // CHECK: vm.const.i64 -1 : i64
    %c = vm.const.i32 0x0000FFFF : i32
    %0 = vm.ext.i16.i64.s %c : i32 -> i64
    vm.return %0 : i64
  }

  // CHECK-LABEL: @ext_i16_i64_u_const
  vm.func @ext_i16_i64_u_const() -> i64 {
    // CHECK: vm.const.i64 65535 : i64
    %c = vm.const.i32 0x0000FFFF : i32
    %0 = vm.ext.i16.i64.u %c : i32 -> i64
    vm.return %0 : i64
  }

  // CHECK-LABEL: @ext_i32_i64_s_const
  vm.func @ext_i32_i64_s_const() -> i64 {
    // CHECK: vm.const.i64 -1 : i64
    %c = vm.const.i32 0xFFFFFFFF : i32
    %0 = vm.ext.i32.i64.s %c : i32 -> i64
    vm.return %0 : i64
  }

  // CHECK-LABEL: @ext_i32_i64_u_const
  vm.func @ext_i32_i64_u_const() -> i64 {
    // CHECK: vm.const.i64 4294967295 : i64
    %c = vm.const.i32 0xFFFFFFFF : i32
    %0 = vm.ext.i32.i64.u %c : i32 -> i64
    vm.return %0 : i64
  }
}
