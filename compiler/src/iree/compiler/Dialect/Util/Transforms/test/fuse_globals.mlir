// RUN: iree-opt --split-input-file --iree-util-fuse-globals --allow-unregistered-dialect %s | FileCheck %s

// CHECK: util.global private mutable @fusable0 : index
util.global private mutable @fusable0 : index
util.global private mutable @fusable1 : index
util.func @foo(%arg0: index) -> (index, index) {
  // CHECK: util.global.store %arg0, @fusable0
  util.global.store %arg0, @fusable0 : index
  // CHECK-NOT: util.global.store %arg0, @fusable1
  util.global.store %arg0, @fusable1 : index
  // CHECK: %[[VALUE0:.+]] = util.global.load @fusable0 : index
  %0 = util.global.load @fusable0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @fusable0 : index
  %1 = util.global.load @fusable1 : index
  // CHECK: util.return %[[VALUE0]], %[[VALUE1]]
  util.return %0, %1 : index, index
}

// -----

// Non-uniform stores.

// CHECK: util.global private mutable @unfusable0 : index
util.global private mutable @unfusable0 : index
// CHECK: util.global private mutable @unfusable1 : index
util.global private mutable @unfusable1 : index
util.func @nonuniform_a(%arg0: index) -> (index, index) {
  // CHECK: util.global.store %arg0, @unfusable0 : index
  util.global.store %arg0, @unfusable0 : index
  // CHECK: util.global.store %arg0, @unfusable1 : index
  util.global.store %arg0, @unfusable1 : index
  // CHECK: %[[VALUE0:.+]] = util.global.load @unfusable0 : index
  %0 = util.global.load @unfusable0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @unfusable1 : index
  %1 = util.global.load @unfusable1 : index
  // CHECK: util.return %[[VALUE0]], %[[VALUE1]]
  util.return %0, %1 : index, index
}
util.func @nonuniform_b(%arg0: index) {
  util.global.store %arg0, @unfusable0 : index
  util.return
}
util.initializer {
  %0 = "some.op"() : () -> index
  util.global.store %0, @unfusable1 : index
  util.return
}

// -----

// Different initializers.

// CHECK: util.global private mutable @unfusableInit0 = 5 : index
util.global private mutable @unfusableInit0 = 5 : index
// CHECK: util.global private mutable @unfusableInit1 = 6 : index
util.global private mutable @unfusableInit1 = 6 : index
util.func @initializer_mix(%arg0: index) -> (index, index) {
  // CHECK: util.global.store %arg0, @unfusableInit0
  util.global.store %arg0, @unfusableInit0 : index
  // CHECK: util.global.store %arg0, @unfusableInit1
  util.global.store %arg0, @unfusableInit1 : index
  // CHECK: %[[VALUE0:.+]] = util.global.load @unfusableInit0 : index
  %0 = util.global.load @unfusableInit0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @unfusableInit1 : index
  %1 = util.global.load @unfusableInit1 : index
  // CHECK: util.return %[[VALUE0]], %[[VALUE1]]
  util.return %0, %1 : index, index
}

// -----

// CHECK: util.global private mutable @unfusableDivergent0
util.global private mutable @unfusableDivergent0 : index
// CHECK: util.global private mutable @unfusableDivergent1
util.global private mutable @unfusableDivergent1 : index
util.func @fn_a(%arg0: index) {
  util.global.store %arg0, @unfusableDivergent0 : index
  util.global.store %arg0, @unfusableDivergent1 : index
  util.return
}
util.func @fn_b(%arg0: index) {
  util.global.store %arg0, @unfusableDivergent0 : index
  util.return
}

// -----

// Tests globals that have some subset fusable and some not.

// CHECK: util.global private mutable @fusableSubset0 : index
util.global private mutable @fusableSubset0 : index
util.global private mutable @fusableSubset1 : index
// CHECK: util.global private mutable @unfusableSubset2 : index
util.global private mutable @unfusableSubset2 : index
// CHECK: util.initializer
util.initializer {
  // CHECK: %[[V:.+]] = arith.constant 4
  %v = arith.constant 4 : index
  // CHECK-NEXT: util.global.store %[[V]], @fusableSubset0
  util.global.store %v, @fusableSubset0 : index
  util.global.store %v, @fusableSubset1 : index
  // CHECK-NEXT: util.global.store %[[V]], @unfusableSubset2
  util.global.store %v, @unfusableSubset2 : index
  util.return
}
// CHECK: util.func public @mutate_unfusable(%[[ARG0:.+]]: index)
util.func public @mutate_unfusable(%arg0: index) {
  // CHECK: util.global.store %[[ARG0]], @unfusableSubset2
  util.global.store %arg0, @unfusableSubset2 : index
  util.return
}
