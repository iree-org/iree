// RUN: iree-opt -split-input-file -pass-pipeline='test-iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_br
module @t001_br {

module {
  func @my_fn(%arg0 : i32) -> (i32) {
    // CHECK: vm.br ^bb1
    br ^bb1
  ^bb1:
    return %arg0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_cond_br
module @t002_cond_br {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func @my_fn(%arg0 : i1, %arg1 : i32, %arg2 : i32) -> (i32) {
    // CHECK: vm.cond_br %[[ARG0]], ^bb1, ^bb2
    cond_br %arg0, ^bb1, ^bb2
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
  // CHECK: func @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func @my_fn(%arg0 : i32, %arg1 : i32) -> (i32) {
    // CHECK: vm.br ^bb1(%[[ARG0]], %[[ARG1]] : i32, i32)
    br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t004_cond_br_args
module @t004_cond_br_args {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]
  func @my_fn(%arg0 : i1, %arg1 : i32, %arg2 : i32) -> (i32) {
    // CHECK: vm.cond_br %[[ARG0]], ^bb1(%[[ARG1]] : i32), ^bb2(%[[ARG2]] : i32)
    cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    return %0 : i32
  ^bb2(%1 : i32):
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t005_call
module @t005_call {

module {
  func private @import_fn(%arg0 : i32) -> i32
  // CHECK: func @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func @my_fn(%arg0 : i32) -> (i32) {
    // CHECK: vm.call @import_fn(%[[ARG0]]) : (i32) -> i32
    %0 = call @import_fn(%arg0) : (i32) -> i32
    return %0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t005_call_int_promotion
module @t005_call_int_promotion {

module {
  func private @import_fn(%arg0 : i1) -> i1
  // CHECK: func @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func @my_fn(%arg0 : i1) -> (i1) {
    // CHECK: vm.call @import_fn(%[[ARG0]]) : (i32) -> i32
    %0 = call @import_fn(%arg0) : (i1) -> i1
    return %0 : i1
  }
}

}

// -----
// CHECK-LABEL: @t006_assert
module @t006_assert {

module {
  // CHECK: vm.func @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  func @my_fn(%arg0 : i32) -> (i32) {
    %zero = constant 0 : i32
    // CHECK: %[[COND:.*]] = vm.cmp.ne.i32
    // CHECK-NEXT: vm.cond_br %[[COND]], ^[[BBTRUE:.*]], ^[[BBFALSE:.*]]
    %cond = cmpi ne, %arg0, %zero : i32
    // CHECK: ^[[BBTRUE]]:
    // CHECK: %[[SUM:.*]] = vm.add.i32
    // CHECK: return %[[SUM]]
    // CHECK: ^[[BBFALSE]]:
    // CHECK: %[[STATUS:.*]] = vm.const.i32
    // CHECK: vm.fail %[[STATUS]]
    assert %cond, "Assertion failed"
    %sum = addi %arg0, %arg0 : i32
    return %sum : i32
  }
}

}
