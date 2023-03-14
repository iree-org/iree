// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(test-iree-convert-std-to-vm)" %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_addi
module @t001_addi {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.add.i32 %[[ARG0]], %[[ARG1]]
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_divis
module @t002_divis {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.div.i32.s %[[ARG0]], %[[ARG1]]
    %0 = arith.divsi %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_diviu
module @t002_diviu {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.div.i32.u %[[ARG0]], %[[ARG1]]
    %0 = arith.divui %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t003_muli
module @t003_muli {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.mul.i32 %[[ARG0]], %[[ARG1]]
    %0 = arith.muli %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t004_remis
module @t004_remis {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.rem.i32.s %[[ARG0]], %[[ARG1]]
    %0 = arith.remsi %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t005_remiu
module @t005_remiu {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.rem.i32.u %[[ARG0]], %[[ARG1]]
    %0 = arith.remui %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t006_subi
module @t006_subi {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.sub.i32 %[[ARG0]], %[[ARG1]]
    %0 = arith.subi %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t007_and
module @t007_and {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.and.i32 %[[ARG0]], %[[ARG1]]
    %0 = arith.andi %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t008_or
module @t008_or {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.or.i32 %[[ARG0]], %[[ARG1]]
    %0 = arith.ori %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t009_xor
module @t009_xor {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1: i32) -> (i32) {
    // CHECK: vm.xor.i32 %[[ARG0]], %[[ARG1]]
    %0 = arith.xori %arg0, %arg1 : i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t010_shift_left
module @t010_shift_left {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32) -> (i32) {
    %c3 = arith.constant 3 : i32
    // CHECK: vm.shl.i32 %[[ARG0]], %c3 : i32
    %1 = arith.shli %arg0, %c3 : i32
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t011_shift_right
module @t011_shift_right {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32) -> (i32) {
    %c3 = arith.constant 3 : i32
    // CHECK: %[[T:.+]] = vm.shr.i32.s %[[ARG0]], %c3 : i32
    %1 = arith.shrsi %arg0, %c3 : i32
    // CHECK: vm.shr.i32.u %[[T]], %c3 : i32
    %2 = arith.shrui %1, %c3 : i32
    return %2 : i32
  }
}

}
