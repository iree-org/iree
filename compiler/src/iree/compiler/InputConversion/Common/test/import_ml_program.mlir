// RUN: iree-opt --split-input-file --iree-import-ml-program %s | FileCheck %s

// CHECK-LABEL: module @globals
builtin.module @globals attributes {
    ml_program.public_global_accessors = {
      get = "global${0}$get", set = "global${0}$set"}} {
  // CHECK: util.global private mutable @global_pubmut = 51 : i32
  // CHECK: util.func public @global$global_pubmut$get() -> i32
  // CHECK: util.func public @global$global_pubmut$set(%{{.*}}: i32)
  // CHECK-NOT: func
  ml_program.global public mutable @global_pubmut(51 : i32) : i32
  // CHECK: util.global private @global_pub = 52 : i32
  // CHECK: util.func public @global$global_pub$get() -> i32
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
  // CHECK: util.func public @global__global_pubmut__get() -> i32
  // CHECK: util.func public @global$global_pubmut$set
  ml_program.global public mutable @global_pubmut(51 : i32) : i32
}

// -----
// CHECK-LABEL: module @no_accessors_globals
builtin.module @no_accessors_globals {
  // CHECK: util.global private mutable @global_pubmut = 51 : i32
  // CHECK: util.func public @global$global_pubmut$get() -> i32
  // CHECK: util.func public @global$global_pubmut$set(%{{.*}}: i32)
  ml_program.global public mutable @global_pubmut(51 : i32) : i32
}

// -----
// CHECK-LABEL: module @global_load
builtin.module @global_load {
  ml_program.global private @v_loaded(dense<0> : tensor<4xi32>)  : tensor<4xi32>
  util.func @loaded() {
    // CHECK: util.global.load @v_loaded : tensor<4xi32>
    %0 = ml_program.global_load @v_loaded : tensor<4xi32>
    util.return
  }
}

// -----
// CHECK-LABEL: module @global_load_const
builtin.module @global_load_const {
  ml_program.global private @v_loaded(dense<0> : tensor<4xi32>)  : tensor<4xi32>
  util.func @loaded() {
    // CHECK: util.global.load @v_loaded : tensor<4xi32>
    %0 = ml_program.global_load_const @v_loaded : tensor<4xi32>
    util.return
  }
}

// -----
// CHECK-LABEL: module @global_store
builtin.module @global_store {
  ml_program.global private mutable @v_stored : tensor<4xi32>
  util.func @stored() {
    // CHECK: %[[CST:.*]] = arith.constant
    %cst = arith.constant dense<5> : tensor<4xi32>
    // CHECK: util.global.store %[[CST]], @v_stored : tensor<4xi32>
    ml_program.global_store @v_stored = %cst : tensor<4xi32>
    util.return
  }
}

// -----
// CHECK-LABEL: module @globals_extern
builtin.module @globals_extern {
  ml_program.global public @global_pub(#ml_program.extern<i32>) : i32
  ml_program.global public @global_pab(#ml_program.extern<i32>) : i32
  ml_program.global private mutable @global_privmut(#ml_program.extern<i32>) : i32
  ml_program.global private @global_priv(#ml_program.extern<i32>) : i32
}

// CHECK-DAG: util.global private mutable @global_pub : i32
// CHECK-DAG: util.global private mutable @global_pab : i32
// CHECK-DAG: util.global private mutable @global_privmut : i32
// CHECK-DAG: util.global private mutable @global_priv : i32

// CHECK-LABEL: util.func public @ireeMlProgramGlobalsInit(
// CHECK-SAME:    %[[VAL_0:.*]]: !util.list<?>
// CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_2:.*]] = util.list.get %[[VAL_0]]{{\[}}%[[VAL_1]]] : !util.list<?> -> i32
// CHECK:  util.global.store %[[VAL_2]], @global_pab : i32
// CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_4:.*]] = util.list.get %[[VAL_0]]{{\[}}%[[VAL_3]]] : !util.list<?> -> i32
// CHECK:  util.global.store %[[VAL_4]], @global_priv : i32
// CHECK:  %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_6:.*]] = util.list.get %[[VAL_0]]{{\[}}%[[VAL_5]]] : !util.list<?> -> i32
// CHECK:  util.global.store %[[VAL_6]], @global_privmut : i32
// CHECK:  %[[VAL_7:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_8:.*]] = util.list.get %[[VAL_0]]{{\[}}%[[VAL_7]]] : !util.list<?> -> i32
// CHECK:  util.global.store %[[VAL_8]], @global_pub : i32

