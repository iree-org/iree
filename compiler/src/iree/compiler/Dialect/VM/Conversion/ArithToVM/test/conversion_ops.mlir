// RUN: iree-opt --split-input-file --iree-vm-conversion --cse --iree-vm-target-index-bits=64 --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @zext_i1_i32
module @zext_i1_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i1) -> i32 {
    // CHECK: %[[MASK:.+]] = vm.const.i32 1
    // CHECK: vm.and.i32 %[[ARG0]], %[[MASK]] : i32
    %0 = arith.extui %arg0 : i1 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @zext_i8_i32
module @zext_i8_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i8) -> i32 {
    // CHECK: vm.ext.i8.i32.u %[[ARG0]] : i32 -> i32
    %0 = arith.extui %arg0 : i8 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @zext_i16_i32
module @zext_i16_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i16) -> i32 {
    // CHECK: vm.ext.i16.i32.u %[[ARG0]] : i32 -> i32
    %0 = arith.extui %arg0 : i16 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @zext_i1_i64
module @zext_i1_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i1) -> i64 {
    // CHECK: %[[MASK:.+]] = vm.const.i32 1
    // CHECK: %[[MASKED:.+]] = vm.and.i32 %[[ARG0]], %[[MASK]] : i32
    // CHECK: vm.ext.i32.i64.u %[[MASKED]] : i32 -> i64
    %0 = arith.extui %arg0 : i1 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @zext_i8_i64
module @zext_i8_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i8) -> i64 {
    // CHECK: vm.ext.i8.i64.u %[[ARG0]] : i32 -> i64
    %0 = arith.extui %arg0 : i8 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @zext_i16_i64
module @zext_i16_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i16) -> i64 {
    // CHECK: vm.ext.i16.i64.u %[[ARG0]] : i32 -> i64
    %0 = arith.extui %arg0 : i16 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @zext_i32_i64
module @zext_i32_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i32) -> i64 {
    // CHECK: vm.ext.i32.i64.u %[[ARG0]] : i32 -> i64
    %0 = arith.extui %arg0 : i32 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @sext_i1_i32
module @sext_i1_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i1) -> i32 {
    // CHECK-DAG: %[[ZEROS:.+]] = vm.const.i32.zero
    // CHECK-DAG: %[[ONES:.+]] = vm.const.i32 -1
    // CHECK: vm.select.i32 %[[ARG0]], %[[ONES]], %[[ZEROS]] : i32
    %0 = arith.extsi %arg0 : i1 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sext_i8_i32
module @sext_i8_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i8) -> i32 {
    // CHECK: vm.ext.i8.i32.s %[[ARG0]] : i32 -> i32
    %0 = arith.extsi %arg0 : i8 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sext_i16_i32
module @sext_i16_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i16) -> i32 {
    // CHECK: vm.ext.i16.i32.s %[[ARG0]] : i32 -> i32
    %0 = arith.extsi %arg0 : i16 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sext_i1_i64
module @sext_i1_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i1) -> i64 {
    // CHECK-DAG: %[[ZEROS:.+]] = vm.const.i64.zero
    // CHECK-DAG: %[[ONES:.+]] = vm.const.i64 -1
    // CHECK: vm.select.i64 %[[ARG0]], %[[ONES]], %[[ZEROS]] : i64
    %0 = arith.extsi %arg0 : i1 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @sext_i8_i64
module @sext_i8_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i8) -> i64 {
    // CHECK: vm.ext.i8.i64.s %[[ARG0]] : i32 -> i64
    %0 = arith.extsi %arg0 : i8 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @sext_i16_i64
module @sext_i16_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i16) -> i64 {
    // CHECK: vm.ext.i16.i64.s %[[ARG0]] : i32 -> i64
    %0 = arith.extsi %arg0 : i16 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @sext_i32_i64
module @sext_i32_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i32) -> i64 {
    // CHECK: vm.ext.i32.i64.s %[[ARG0]] : i32 -> i64
    %0 = arith.extsi %arg0 : i32 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @trunc_i32_i1
module @trunc_i32_i1 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i32) -> i1 {
    // CHECK-DAG: %[[MASK:.+]] = vm.const.i32 1
    // CHECK: vm.and.i32 %[[ARG0]], %[[MASK]] : i32
    %0 = arith.trunci %arg0 : i32 to i1
    return %0 : i1
  }
}

// -----

// CHECK-LABEL: @trunc_i32_i8
module @trunc_i32_i8 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i32) -> i8 {
    // CHECK: vm.trunc.i32.i8 %[[ARG0]] : i32 -> i32
    %0 = arith.trunci %arg0 : i32 to i8
    return %0 : i8
  }
}

// -----

// CHECK-LABEL: @trunc_i32_i16
module @trunc_i32_i16 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i32) -> i16 {
    // CHECK: vm.trunc.i32.i16 %[[ARG0]] : i32 -> i32
    %0 = arith.trunci %arg0 : i32 to i16
    return %0 : i16
  }
}

// -----

// CHECK-LABEL: @trunc_i64_i1
module @trunc_i64_i1 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i64)
  func.func @fn(%arg0: i64) -> i1 {
    // CHECK-DAG: %[[TRUNC:.+]] = vm.trunc.i64.i32 %[[ARG0]] : i64 -> i32
    // CHECK-DAG: %[[MASK:.+]] = vm.const.i32 1
    // CHECK: vm.and.i32 %[[TRUNC]], %[[MASK]] : i32
    %0 = arith.trunci %arg0 : i64 to i1
    return %0 : i1
  }
}

// -----

// CHECK-LABEL: @trunc_i64_i8
module @trunc_i64_i8 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i64)
  func.func @fn(%arg0: i64) -> i8 {
    // CHECK: vm.trunc.i64.i8 %[[ARG0]] : i64 -> i32
    %0 = arith.trunci %arg0 : i64 to i8
    return %0 : i8
  }
}

// -----

// CHECK-LABEL: @trunc_i64_i16
module @trunc_i64_i16 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i64)
  func.func @fn(%arg0: i64) -> i16 {
    // CHECK: vm.trunc.i64.i16 %[[ARG0]] : i64 -> i32
    %0 = arith.trunci %arg0 : i64 to i16
    return %0 : i16
  }
}

// -----

// CHECK-LABEL: @trunc_i64_i32
module @trunc_i64_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i64)
  func.func @fn(%arg0: i64) -> i32 {
    // CHECK: vm.trunc.i64.i32 %[[ARG0]] : i64 -> i32
    %0 = arith.trunci %arg0 : i64 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sitofp_i8_f32
module @sitofp_i8_f32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i8) -> f32 {
    // CHECK: %[[X:.*]] = vm.ext.i8.i32.s %[[ARG0]]
    // CHECK: vm.cast.si32.f32 %[[X]] : i32 -> f32
    %0 = arith.sitofp %arg0 : i8 to f32
    return %0 : f32
  }
}

// -----

// CHECK-LABEL: @uitofp_i8_f32
module @uitofp_i8_f32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i8) -> f32 {
    // CHECK: %[[X:.*]] = vm.ext.i8.i32.u %[[ARG0]]
    // CHECK: vm.cast.ui32.f32 %[[X]] : i32 -> f32
    %0 = arith.uitofp %arg0 : i8 to f32
    return %0 : f32
  }
}

// -----

// CHECK-LABEL: @uitofp_i32_f32
module @uitofp_i32_f32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i32) -> f32 {
    // CHECK: vm.cast.ui32.f32 %[[ARG0]] : i32 -> f32
    %0 = arith.uitofp %arg0 : i32 to f32
    return %0 : f32
  }
}

// -----

// CHECK-LABEL: @fptosi_fp32_i8
module @fptosi_fp32_i8 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: f32)
  func.func @fn(%arg0: f32) -> i8 {
    // CHECK: vm.cast.f32.si32 %[[ARG0]] : f32 -> i32
    %0 = arith.fptosi %arg0 : f32 to i8
    return %0 : i8
  }
}

// -----

// CHECK-LABEL: @fptosi_fp32_i32
module @fptosi_fp32_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: f32)
  func.func @fn(%arg0: f32) -> i32 {
    // CHECK: vm.cast.f32.si32 %[[ARG0]] : f32 -> i32
    %0 = arith.fptosi %arg0 : f32 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @fptosi_fp32_i64
module @fptosi_fp32_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: f32)
  func.func @fn(%arg0: f32) -> i64 {
    // CHECK: vm.cast.f32.si64 %[[ARG0]] : f32 -> i64
    %0 = arith.fptosi %arg0 : f32 to i64
    return %0 : i64
  }
}

// -----

// expected-error@+1 {{conversion to vm.module failed}}
module @fptoui_fp32_i8 {
  func.func @fn(%arg0: f32) -> i8 {
    // expected-error@+1 {{failed to legalize}}
    %0 = arith.fptoui %arg0 : f32 to i8
    return %0 : i8
  }
}

// -----

// CHECK-LABEL: @fptoui_fp32_i32
module @fptoui_fp32_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: f32)
  func.func @fn(%arg0: f32) -> i32 {
    // CHECK: vm.cast.f32.ui32 %[[ARG0]] : f32 -> i32
    %0 = arith.fptoui %arg0 : f32 to i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: @fptoui_fp32_i64
module @fptoui_fp32_i64 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: f32)
  func.func @fn(%arg0: f32) -> i64 {
    // CHECK: vm.cast.f32.ui64 %[[ARG0]] : f32 -> i64
    %0 = arith.fptoui %arg0 : f32 to i64
    return %0 : i64
  }
}

// -----

// CHECK-LABEL: @bitcast_i32_f32
module @bitcast_i32_f32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: i32)
  func.func @fn(%arg0: i32) -> f32 {
    // CHECK: vm.bitcast.i32.f32 %[[ARG0]] : i32 -> f32
    %0 = arith.bitcast %arg0 : i32 to f32
    return %0 : f32
  }
}

// -----

// CHECK-LABEL: @bitcast_f32_i32
module @bitcast_f32_i32 {
  // CHECK: vm.func private @fn(%[[ARG0:.+]]: f32)
  func.func @fn(%arg0: f32) -> i32 {
    // CHECK: vm.bitcast.f32.i32 %[[ARG0]] : f32 -> i32
    %0 = arith.bitcast %arg0 : f32 to i32
    return %0 : i32
  }
}

// -----

// expected-error@+1 {{conversion to vm.module failed}}
module @bitcast_f16_bf16 {
  func.func @fn(%arg0: f16) -> bf16 {
    // expected-error@+1 {{failed to legalize}}
    %0 = arith.bitcast %arg0 : f16 to bf16
    return %0 : bf16
  }
}
