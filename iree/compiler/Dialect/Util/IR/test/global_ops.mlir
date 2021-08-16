// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK: util.global public @v_immutable : tensor<i32>
util.global @v_immutable : tensor<i32>
// CHECK: util.global public mutable @v_mutable : tensor<i32>
util.global public mutable @v_mutable : tensor<i32>

// -----

// CHECK: util.global public @v_initialized_const0 = 4 : i32
util.global public @v_initialized_const0 = 4 : i32

// CHECK: util.global public @v_initialized_const1 = 40 : i32
util.global public @v_initialized_const1 = 40 : i32

// CHECK: util.global public @v_initialized_const2 = 40 : i64
util.global public @v_initialized_const2 = 40 : i64

// CHECK: util.global public @v_initialized_const3 = dense<4> : tensor<4xi32>
util.global public @v_initialized_const3 = dense<4> : tensor<4xi32>

// -----

// CHECK: util.global private @v_initialized initializer(@initializer) : tensor<4xi32>
util.global private @v_initialized initializer(@initializer) : tensor<4xi32>
func private @initializer() -> tensor<4xi32>

// -----

util.global private @v_loaded : tensor<4xi32>
// CHECK-LABEL: @loaded
func @loaded() {
  // CHECK-NEXT: = util.global.load @v_loaded : tensor<4xi32>
  %0 = util.global.load @v_loaded : tensor<4xi32>
  return
}

// -----

util.global private mutable @v_stored : tensor<4xi32>
// CHECK-LABEL: @stored
func @stored() {
  // CHECK-NEXT: %[[VAL:.+]] = constant
  %cst = constant dense<5> : tensor<4xi32>
  // CHECK-NEXT: util.global.store %[[VAL]], @v_stored : tensor<4xi32>
  util.global.store %cst, @v_stored : tensor<4xi32>
  return
}

// -----

util.global private @v_loaded : tensor<4xf32>
// CHECK-LABEL: @loaded_indirect
func @loaded_indirect() {
  // CHECK-NEXT: %[[ADDR:.+]] = util.global.address @v_loaded
  %0 = util.global.address @v_loaded : !util.ptr<tensor<4xf32>>
  // CHECK-NEXT: = util.global.load.indirect %[[ADDR]]
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
  return
}

// -----

util.global private mutable @v_stored : tensor<4xf32>
// CHECK-LABEL: @stored_indirect
// CHECK-SAME: (%[[VALUE:.+]]: tensor<4xf32>)
func @stored_indirect(%arg0: tensor<4xf32>) {
  // CHECK-NEXT: %[[ADDR:.+]] = util.global.address @v_stored
  %0 = util.global.address @v_stored : !util.ptr<tensor<4xf32>>
  // CHECK-NEXT: util.global.store.indirect %[[VALUE]], %[[ADDR]]
  util.global.store.indirect %arg0, %0 : tensor<4xf32> -> !util.ptr<tensor<4xf32>>
  return
}
