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
// CHECK-LABEL: @t005_br_table
module @t005_br_table {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%flag: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {
    // CHECK: %[[INDEX:.+]] = vm.sub.i32 %[[FLAG]], %c100
    //      CHECK: vm.br_table %[[INDEX]] {
    // CHECK-NEXT:   default: ^bb1(%[[ARG1]] : i32),
    // CHECK-NEXT:   0: ^bb1(%[[ARG2]] : i32),
    // CHECK-NEXT:   1: ^bb1(%[[ARG1]] : i32),
    // CHECK-NEXT:   2: ^bb1(%[[ARG1]] : i32),
    // CHECK-NEXT:   3: ^bb1(%[[ARG1]] : i32),
    // CHECK-NEXT:   4: ^bb1(%[[ARG3]] : i32),
    // CHECK-NEXT:   5: ^bb1(%[[ARG1]] : i32),
    // CHECK-NEXT:   6: ^bb2
    // CHECK-NEXT: }
    cf.switch %flag : i32, [
      default: ^bb1(%arg1 : i32),
      104: ^bb1(%arg3 : i32),
      100: ^bb1(%arg2 : i32),
      106: ^bb2
    ]
  ^bb1(%0 : i32):
    return %0 : i32
  ^bb2:
    return %arg1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t006_br_table
module @t006_br_table_idx {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%flag: i32) -> i32 {
    // CHECK-DAG: %[[C1:.+]] = vm.const.i64 1
    // CHECK-DAG: %[[C2:.+]] = vm.const.i64 2
    // CHECK-DAG: %[[C2_I32:.+]] = vm.const.i32 2
    // CHECK: %[[INDEX:.+]] = vm.sub.i32 %[[FLAG]], %[[C2_I32]]
    //      CHECK: vm.br_table %[[INDEX]] {
    // CHECK-NEXT:   default: ^bb1(%[[C2]] : i64),
    // CHECK-NEXT:   0: ^bb1(%[[C1]] : i64),
    // CHECK-NEXT:   1: ^bb1(%[[C2]] : i64),
    // CHECK-NEXT:   2: ^bb1(%[[C2]] : i64),
    // CHECK-NEXT:   3: ^bb1(%[[C2]] : i64),
    // CHECK-NEXT:   4: ^bb2
    // CHECK-NEXT: }
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    cf.switch %flag : i32, [
      default: ^bb1(%c2 : index),
      2: ^bb1(%c1 : index),
      6: ^bb2
    ]
  ^bb1(%0 : index):
    %cast = arith.index_cast %0 : index to i32
    return %cast : i32
  ^bb2:
    return %flag : i32
  }
}

}

// -----
// CHECK-LABEL: @t007_assert
module @t007_assert {

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
