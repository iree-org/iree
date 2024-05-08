// RUN: iree-opt --split-input-file --iree-vm-conversion --cse --iree-vm-target-index-bits=64 --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @t001_bitcast_i32_f32
module @t001_bitcast_i32_f32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: i32
  func.func @my_fn(%arg0 : i32) -> (f32) {
    // CHECK: vm.bitcast.i32.f32 %[[ARG0]] : i32 -> f32
    %1 = arith.bitcast %arg0 : i32 to f32
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
  func.func @my_fn(%arg0 : f32) -> (i32) {
    // CHECK: vm.bitcast.f32.i32 %[[ARG0]] : f32 -> i32
    %1 = arith.bitcast %arg0 : f32 to i32
    return %1 : i32
  }
}
}

// -----

module @t003_sitofp_i8_f32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: i32
  func.func @my_fn(%arg0 : i8) -> (f32) {
    // CHECK: %[[X:.*]] = vm.ext.i8.i32.s %[[ARG0]]
    // CHECK: vm.cast.si32.f32 %[[X]] : i32 -> f32
    %1 = arith.sitofp %arg0 : i8 to f32
    return %1 : f32
  }
}
}

// -----

module @t004_uitofp_i8_f32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: i32
  func.func @my_fn(%arg0 : i8) -> (f32) {
    // CHECK: %[[X:.*]] = vm.ext.i8.i32.u %[[ARG0]]
    // CHECK: vm.cast.ui32.f32 %[[X]] : i32 -> f32
    %1 = arith.uitofp %arg0 : i8 to f32
    return %1 : f32
  }
}
}

// -----

module @t005_uitofp_i32_f32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: i32
  func.func @my_fn(%arg0 : i32) -> (f32) {
    // CHECK: vm.cast.ui32.f32 %[[ARG0]] : i32 -> f32
    %1 = arith.uitofp %arg0 : i32 to f32
    return %1 : f32
  }
}
}

// -----

module @t006_fptosi_fp32_i8 {
module @my_module {
  func.func @my_fn(%arg0 : f32) -> (i8) {
    // CHECK: vm.cast.f32.si32 %[[ARG0]] : f32 -> i32
    %1 = arith.fptosi %arg0 : f32 to i8
    return %1 : i8
  }
}
}

// -----

module @t007_fptosi_fp32_i32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: f32
  func.func @my_fn(%arg0 : f32) -> (i32) {
    // CHECK: vm.cast.f32.si32 %[[ARG0]] : f32 -> i32
    %1 = arith.fptosi %arg0 : f32 to i32
    return %1 : i32
  }
}
}

// -----

// expected-error@+1 {{conversion to vm.module failed}}
module @t008_fptoui_fp32_i8 {
module @my_module {
  func.func @my_fn(%arg0 : f32) -> (i8) {
    // expected-error@+1 {{failed to legalize}}
    %1 = arith.fptoui %arg0 : f32 to i8
    return %1 : i8
  }
}
}

// -----

module @t009_fptoui_fp32_i32 {
module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:.+]]: f32
  func.func @my_fn(%arg0 : f32) -> (i32) {
    // CHECK: vm.cast.f32.ui32 %[[ARG0]] : f32 -> i32
    %1 = arith.fptoui %arg0 : f32 to i32
    return %1 : i32
  }
}
}

// -----

// expected-error@+1 {{conversion to vm.module failed}}
module @t001_bitcast_f16_bf16 {
module @my_module {
  func.func @my_fn(%arg0 : f16) -> (bf16) {
    // expected-error@+1 {{failed to legalize}}
    %1 = arith.bitcast %arg0 : f16 to bf16
    return %1 : bf16
  }
}
}

