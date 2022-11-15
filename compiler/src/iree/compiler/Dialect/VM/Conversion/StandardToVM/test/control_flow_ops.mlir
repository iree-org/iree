// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(test-iree-convert-std-to-vm)" %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_br
module @t001_br {

module {
  func.func @my_fn(%arg0 : i32) -> (i32) {
    // CHECK: vm.br ^bb1
    cf.br ^bb1
  ^bb1:
    return %arg0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_cond_br
module @t002_cond_br {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0 : i1, %arg1 : i32, %arg2 : i32) -> (i32) {
    // CHECK: vm.cond_br %[[ARG0]], ^bb1, ^bb2
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    return %arg1 : i32
  ^bb2:
    return %arg2 : i32
  }
}

}

// -----
// CHECK-LABEL: @t003_br_args
module @t003_br_args {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0 : i32, %arg1 : i32) -> (i32) {
    // CHECK: vm.br ^bb1(%[[ARG0]], %[[ARG1]] : i32, i32)
    cf.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t004_cond_br_args
module @t004_cond_br_args {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0 : i1, %arg1 : i32, %arg2 : i32) -> (i32) {
    // CHECK: vm.cond_br %[[ARG0]], ^bb1(%[[ARG1]] : i32), ^bb2(%[[ARG2]] : i32)
    cf.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    return %0 : i32
  ^bb2(%1 : i32):
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t006_assert
module @t006_assert {

module {
  // CHECK: vm.func private @my_fn
  func.func @my_fn(%arg0: i32) -> i32 {
    %zero = arith.constant 0 : i32
    // CHECK: %[[COND:.+]] = vm.cmp.ne.i32
    %cond = arith.cmpi ne, %arg0, %zero : i32
    // CHECK: %[[STATUS:.+]] = vm.const.i32 9
    // CHECK: %[[INVCOND:.+]] = vm.xor.i32 %[[COND]], %c1
    // CHECK: vm.cond_fail %[[INVCOND]], %[[STATUS]], "Assertion failed"
    cf.assert %cond, "Assertion failed"
    %sum = arith.addi %arg0, %arg0 : i32
    return %sum : i32
  }
}

}
