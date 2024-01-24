// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//      CHECK: util.initializer {
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer {
  util.return
}

// -----

//      CHECK: util.initializer attributes {foo} {
// CHECK-NEXT:   util.return
// CHECK-NEXT: }
util.initializer attributes {foo} {
  util.return
}

// -----

// CHECK: util.initializer {
util.initializer {
  // CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0 : i32
  %zero = arith.constant 0 : i32
  // CHECK-NEXT:   cf.br ^bb1(%[[ZERO]] : i32)
  cf.br ^bb1(%zero: i32)
  // CHECK-NEXT: ^bb1(%0: i32):
^bb1(%0: i32):
  // CHECK-NEXT:   util.return
  util.return
}
