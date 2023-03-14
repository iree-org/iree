// RUN: iree-opt --split-input-file --iree-import-ml-program %s | FileCheck %s

// CHECK-LABEL: module @globals
builtin.module @globals attributes {
    ml_program.public_global_accessors = {
      get = "global${0}$get", set = "global${0}$set"}} {
  // CHECK: util.global private mutable @global_pubmut = 51 : i32
  // CHECK: func @global$global_pubmut$get() -> i32
  // CHECK: func @global$global_pubmut$set(%{{.*}}: i32)
  // CHECK-NOT: func
  ml_program.global public mutable @global_pubmut(51 : i32) : i32
  // CHECK: util.global private @global_pub = 52 : i32
  // CHECK: func @global$global_pub$get() -> i32
  // CHECK-NOT: func
  ml_program.global public @global_pub(52 : i32) : i32
  // CHECK: util.global private mutable @global_privmut = 53 : i32
  // CHECK-NOT: func
  ml_program.global private mutable @global_privmut(53 : i32) : i32
  // CHECK: util.global private @global_priv = 54 : i32
  ml_program.global private @global_priv(54 : i32) : i32
}

// -----
// CHECK-LABEL: module @globals
builtin.module @globals attributes {
    ml_program.public_global_accessors = {get = "global__{0}__get"}} {
  // CHECK: util.global private mutable @global_pubmut = 51 : i32
  // CHECK: func @global__global_pubmut__get() -> i32
  // CHECK: func @global$global_pubmut$set
  ml_program.global public mutable @global_pubmut(51 : i32) : i32
}

// -----
// CHECK-LABEL: module @no_accessors_globals
builtin.module @no_accessors_globals {
  // CHECK: util.global private mutable @global_pubmut = 51 : i32
  // CHECK: func @global$global_pubmut$get() -> i32
  // CHECK: func @global$global_pubmut$set(%{{.*}}: i32)
  ml_program.global public mutable @global_pubmut(51 : i32) : i32
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
