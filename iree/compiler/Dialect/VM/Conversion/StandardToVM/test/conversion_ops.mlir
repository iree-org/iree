// RUN: iree-opt -split-input-file -pass-pipeline='test-iree-convert-std-to-vm' %s | IreeFileCheck %s

// CHECK-LABEL: @t001_bitcast_i32_f32
module @t001_bitcast_i32_f32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: i32
  func @my_fn(%arg0 : i32) -> (f32) {
    // CHECK: vm.bitcast.i32.f32 %[[ARG0]] : i32 -> f32
    %1 = bitcast %arg0 : i32 to f32
    return %1 : f32
  }
}
}

// -----

// CHECK-LABEL: @t002_bitcast_f32_i32
module @t002_bitcast_f32_i32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: f32
  func @my_fn(%arg0 : f32) -> (i32) {
    // CHECK: vm.bitcast.f32.i32 %[[ARG0]] : f32 -> i32
    %1 = bitcast %arg0 : f32 to i32
    return %1 : i32
  }
}
}
