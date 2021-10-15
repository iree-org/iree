// RUN: iree-opt -split-input-file -iree-util-fuse-globals -allow-unregistered-dialect %s | IreeFileCheck %s

// CHECK: util.global private mutable @fusable0 : index
util.global private mutable @fusable0 : index
util.global private mutable @fusable1 : index
builtin.func @foo(%arg0: index) -> (index, index) {
  // CHECK: util.global.store %arg0, @fusable0
  util.global.store %arg0, @fusable0 : index
  // CHECK-NOT: util.global.store %arg0, @fusable1
  util.global.store %arg0, @fusable1 : index
  // CHECK: %[[VALUE0:.+]] = util.global.load @fusable0 : index
  %0 = util.global.load @fusable0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @fusable0 : index
  %1 = util.global.load @fusable1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]]
  return %0, %1 : index, index
}

// -----

// Non-uniform stores.

// CHECK: util.global private mutable @unfusable0 : index
util.global private mutable @unfusable0 : index
// CHECK: util.global private mutable @unfusable1 : index
util.global private mutable @unfusable1 : index
builtin.func @foo(%arg0: index) -> (index, index) {
  // CHECK: util.global.store %arg0, @unfusable0 : index
  util.global.store %arg0, @unfusable0 : index
  // CHECK: util.global.store %arg0, @unfusable1 : index
  util.global.store %arg0, @unfusable1 : index
  // CHECK: %[[VALUE0:.+]] = util.global.load @unfusable0 : index
  %0 = util.global.load @unfusable0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @unfusable1 : index
  %1 = util.global.load @unfusable1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]]
  return %0, %1 : index, index
}
builtin.func @bar(%arg0: index) {
  util.global.store %arg0, @unfusable0 : index
  return
}
util.initializer {
  %0 = "some.op"() : () -> index
  util.global.store %0, @unfusable1 : index
  util.initializer.return
}

// -----

// Different initializers.

// CHECK: util.global private mutable @unfusableInit0 = 5 : index
util.global private mutable @unfusableInit0 = 5 : index
// CHECK: util.global private mutable @unfusableInit1 = 6 : index
util.global private mutable @unfusableInit1 = 6 : index
builtin.func @foo(%arg0: index) -> (index, index) {
  // CHECK: util.global.store %arg0, @unfusableInit0
  util.global.store %arg0, @unfusableInit0 : index
  // CHECK: util.global.store %arg0, @unfusableInit1
  util.global.store %arg0, @unfusableInit1 : index
  // CHECK: %[[VALUE0:.+]] = util.global.load @unfusableInit0 : index
  %0 = util.global.load @unfusableInit0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @unfusableInit1 : index
  %1 = util.global.load @unfusableInit1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]]
  return %0, %1 : index, index
}
