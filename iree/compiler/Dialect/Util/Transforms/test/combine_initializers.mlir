// RUN: iree-opt -split-input-file -iree-util-combine-initializers %s | IreeFileCheck %s

builtin.func private @extern() -> index

// CHECK: util.global private mutable @global0 : index
util.global private mutable @global0 : index
util.initializer {
  %value0 = call @extern() : () -> index
  util.global.store %value0, @global0 : index
  util.initializer.return
}
// CHECK-NEXT: util.global private @global1 : index
util.global private @global1 : index
// CHECK-NEXT: util.global private @global2 : index
util.global private @global2 : index
util.initializer {
  %value1 = call @extern() : () -> index
  util.global.store %value1, @global1 : index
  %value2 = call @extern() : () -> index
  util.global.store %value2, @global2 : index
  util.initializer.return
}
// CHECK-NEXT: util.initializer {
// CHECK-NEXT: %[[VALUE0:.+]] = call @extern()
// CHECK-NEXT: util.global.store %[[VALUE0]], @global0
// CHECK-NEXT: %[[VALUE1:.+]] = call @extern()
// CHECK-NEXT: util.global.store %[[VALUE1]], @global1
// CHECK-NEXT: %[[VALUE2:.+]] = call @extern()
// CHECK-NEXT: util.global.store %[[VALUE2]], @global2
// CHECK-NEXT: util.initializer.return
builtin.func @foo(%arg0: index) -> (index, index, index) {
  util.global.store %arg0, @global0 : index
  %value0 = util.global.load @global0 : index
  %value1 = util.global.load @global1 : index
  %value2 = util.global.load @global2 : index
  return %value0, %value1, %value2 : index, index, index
}
