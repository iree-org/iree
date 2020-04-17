// RUN: iree-opt -split-input-file -pass-pipeline='test-iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_addi
module @t001_addi {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.add.i32 [[ARG0]], [[ARG1]]
    %0 = addi %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_divis
module @t002_divis {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.div.i32.s [[ARG0]], [[ARG1]]
    %0 = divi_signed %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_diviu
module @t002_diviu {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.div.i32.u [[ARG0]], [[ARG1]]
    %0 = divi_unsigned %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t003_muli
module @t003_muli {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.mul.i32 [[ARG0]], [[ARG1]]
    %0 = muli %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t004_remis
module @t004_remis {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.rem.i32.s [[ARG0]], [[ARG1]]
    %0 = remi_signed %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t005_remiu
module @t005_remiu {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.rem.i32.u [[ARG0]], [[ARG1]]
    %0 = remi_unsigned %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t006_subi
module @t006_subi {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.sub.i32 [[ARG0]], [[ARG1]]
    %0 = subi %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t007_and
module @t007_and {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.and.i32 [[ARG0]], [[ARG1]]
    %0 = and %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t008_or
module @t008_or {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.or.i32 [[ARG0]], [[ARG1]]
    %0 = or %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t009_xor
module @t009_xor {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.xor.i32 [[ARG0]], [[ARG1]]
    %0 = xor %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t010_shift
module @t010_shift {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32) -> (i32) {
    %cst = constant 3 : i32
    // CHECK: vm.shl.i32 [[ARG0]], 3 : i32
    %1 = shift_left %arg0, %cst : i32
    return %1 : i32
  }
}

}
