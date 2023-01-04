// RUN: iree-opt --split-input-file --iree-import-ml-program %s | FileCheck %s

// CHECK-LABEL: module @globals
builtin.module @globals {
  // CHECK: util.global public mutable @global2 = 51 : i32
  ml_program.global public mutable @global2(51 : i32) : i32
  // CHECK: util.global private mutable @global3 = 52 : i32
  ml_program.global private mutable @global3(52 :i32) : i32
  // CHECK: util.global private @global4 = 53 : i32
  ml_program.global private @global4(53 : i32) : i32
}

// -----
// CHECK-LABEL: module @global_load
builtin.module @global_load {
  ml_program.global private @v_loaded(dense<0> : tensor<4xi32>)  : tensor<4xi32>
  func.func @loaded() {
    // CHECK: util.global.load @v_loaded : tensor<4xi32>
    %0 = ml_program.global_load @v_loaded : tensor<4xi32>
    return
  }
}

// -----
// CHECK-LABEL: module @global_load_const
builtin.module @global_load_const {
  ml_program.global private @v_loaded(dense<0> : tensor<4xi32>)  : tensor<4xi32>
  func.func @loaded() {
    // CHECK: util.global.load @v_loaded : tensor<4xi32>
    %0 = ml_program.global_load_const @v_loaded : tensor<4xi32>
    return
  }
}

// -----
// CHECK-LABEL: module @global_store
builtin.module @global_store {
  ml_program.global private mutable @v_stored : tensor<4xi32>
  func.func @stored() {
    // CHECK: %[[CST:.*]] = arith.constant
    %cst = arith.constant dense<5> : tensor<4xi32>
    // CHECK: util.global.store %[[CST]], @v_stored : tensor<4xi32>
    ml_program.global_store @v_stored = %cst : tensor<4xi32>
    return
  }
}