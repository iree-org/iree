// RUN: iree-opt --split-input-file --iree-util-annotate-op-ordinals %s | FileCheck %s

// CHECK: module attributes {util.ordinal = 0 : index}
module {

// CHECK-NEXT: util.initializer attributes {util.ordinal = 1 : index}
util.initializer {
  // CHECK-NEXT: arith.constant {util.ordinal = 2 : index} true
  %cond = arith.constant true
  // CHECK-NEXT: cf.cond_br {{.+}} {util.ordinal = 3 : index}
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  // CHECK: arith.constant {util.ordinal = 4 : index} 100
  %c100 = arith.constant 100 : index
  // CHECK-NEXT: cf.br {{.+}} {util.ordinal = 5 : index}
  cf.br ^bb2
^bb2:
  // CHECK: util.initializer.return {util.ordinal = 6 : index}
  util.initializer.return
}

// CHECK: util.global private mutable @globalB {util.ordinal = 7 : index}
util.global private mutable @globalB : index
// CHECK-NEXT: func.func @setterFunc() attributes {util.ordinal = 8 : index}
func.func @setterFunc() {
  // CHECK-NEXT: arith.constant {util.ordinal = 9 : index} 300
  %c300 = arith.constant 300 : index
  // CHECK-NEXT: util.global.store {{.+}} {util.ordinal = 10 : index}
  util.global.store %c300, @globalB : index
  // CHECK-NEXT: return {util.ordinal = 11 : index}
  return
}

}
