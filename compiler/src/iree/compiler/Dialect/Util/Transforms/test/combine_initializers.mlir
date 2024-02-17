// RUN: iree-opt --split-input-file --iree-util-combine-initializers %s | FileCheck %s

// Tests that multiple initializers are combined in their module order.

util.func private @extern() -> index

// CHECK: util.global private mutable @global0 : index
util.global private mutable @global0 : index
util.initializer {
  %value0 = util.call @extern() : () -> index
  util.global.store %value0, @global0 : index
  util.return
}
// CHECK-NEXT: util.global private @global1 : index
util.global private @global1 : index
// CHECK-NEXT: util.global private @global2 : index
util.global private @global2 : index
util.initializer {
  %value1 = util.call @extern() : () -> index
  util.global.store %value1, @global1 : index
  %value2 = util.call @extern() : () -> index
  util.global.store %value2, @global2 : index
  util.return
}
// CHECK-NEXT: util.initializer {
// CHECK-NEXT: %[[VALUE0:.+]] = util.call @extern()
// CHECK-NEXT: util.global.store %[[VALUE0]], @global0
// CHECK-NEXT: %[[VALUE1:.+]] = util.call @extern()
// CHECK-NEXT: util.global.store %[[VALUE1]], @global1
// CHECK-NEXT: %[[VALUE2:.+]] = util.call @extern()
// CHECK-NEXT: util.global.store %[[VALUE2]], @global2
// CHECK-NEXT: util.return

// CHECK-LABEL: @orderedCombining
util.func @orderedCombining(%arg0: index) -> (index, index, index) {
  util.global.store %arg0, @global0 : index
  %value0 = util.global.load @global0 : index
  %value1 = util.global.load @global1 : index
  %value2 = util.global.load @global2 : index
  util.return %value0, %value1, %value2 : index, index, index
}

// -----

// Tests that initializers containing CFG ops are inlined into the new combined
// initializer properly.

// CHECK: util.global private mutable @globalA : index
util.global private mutable @globalA : index
util.initializer {
  %cond = arith.constant 1 : i1
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c100 = arith.constant 100 : index
  util.global.store %c100, @globalA : index
  cf.br ^bb3
^bb2:
  %c200 = arith.constant 200 : index
  util.global.store %c200, @globalA : index
  cf.br ^bb3
^bb3:
  util.return
}
// CHECK-NEXT: util.global private @globalB : index
util.global private @globalB : index
util.initializer {
  %c300 = arith.constant 300 : index
  util.global.store %c300, @globalB : index
  util.return
}
// CHECK: util.initializer {
// CHECK: ^bb1:
// CHECK:   cf.br ^bb3
// CHECK: ^bb2:
// CHECK:   cf.br ^bb3
// CHECK: ^bb3:
// CHECK:   cf.br ^bb4
// CHECK: ^bb4:
// CHECK:   util.return
// CHECK: }
