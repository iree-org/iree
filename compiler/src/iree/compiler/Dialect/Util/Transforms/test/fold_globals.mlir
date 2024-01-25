// RUN: iree-opt --split-input-file --iree-util-fold-globals %s | FileCheck %s

// CHECK: util.global public mutable @uniformConstants = 5 : index
util.global public mutable @uniformConstants : index
func.func @foo() {
  %c5 = arith.constant 5 : index
  // CHECK-NOT: util.global.store %c5, @uniformConstants : index
  util.global.store %c5, @uniformConstants : index
  return
}
func.func @bar() {
  %c5 = arith.constant 5 : index
  // CHECK-NOT: util.global.store %c5, @uniformConstants : index
  util.global.store %c5, @uniformConstants : index
  return
}

// -----

// CHECK: util.global public mutable @nonuniformConstants : index
util.global public mutable @nonuniformConstants : index
func.func @foo() {
  %c5 = arith.constant 5 : index
  // CHECK: util.global.store %c5, @nonuniformConstants : index
  util.global.store %c5, @nonuniformConstants : index
  return
}
func.func @bar() {
  %c6 = arith.constant 6 : index
  // CHECK: util.global.store %c6, @nonuniformConstants : index
  util.global.store %c6, @nonuniformConstants : index
  return
}

// -----

// CHECK: util.global private @chained0 : index
util.global private mutable @chained0 : index
// CHECK-NOT: util.global private mutable @chained1 : index
util.global private mutable @chained1 : index
func.func @foo() -> index {
  // CHECK: %[[VALUE:.+]] = util.global.load @chained0 : index
  %0 = util.global.load @chained0 : index
  // CHECK-NOT: util.global.store
  util.global.store %0, @chained1 : index
  // CHECK-NEXT: return %[[VALUE]]
  return %0 : index
}

// -----

// CHECK: util.global public mutable @unchained0 : index
util.global public mutable @unchained0 : index
// CHECK: util.global public mutable @unchained1 : index
util.global public mutable @unchained1 : index
func.func @foo() {
  // CHECK: %[[VALUE:.+]] = util.global.load @unchained0 : index
  %0 = util.global.load @unchained0 : index
  // CHECK: util.global.store %[[VALUE]], @unchained1 : index
  util.global.store %0, @unchained1 : index
  return
}
func.func @bar(%arg0: index) {
  // CHECK: util.global.store %arg0, @unchained1 : index
  util.global.store %arg0, @unchained1 : index
  return
}

// -----

// NOTE: we're indirectly testing the mutable -> immutable change as the
// patterns will inline the constants iff the globals are made immutable.

// CHECK-NOT: @immutable0
util.global private mutable @immutable0 = 5 : index
// CHECK-NOT: @immutable1
util.global private mutable @immutable1 : index
// CHECK: util.global private mutable @mutable : index
util.global private mutable @mutable : index
// CHECK-NOT: util.initializer
util.initializer {
  %c6 = arith.constant 6 : index
  util.global.store %c6, @immutable1 : index
  util.return
}
func.func @foo(%arg0: index) -> (index, index, index) {
  // CHECK-DAG: %[[C5:.+]] = arith.constant 5
  %0 = util.global.load @immutable0 : index
  // CHECK-DAG: %[[C6:.+]] = arith.constant 6
  %1 = util.global.load @immutable1 : index
  // CHECK: %[[MUTABLE:.+]] = util.global.load @mutable
  %2 = util.global.load @mutable : index
  // CHECK: util.global.store %arg0, @mutable
  util.global.store %arg0, @mutable : index
  // CHECK: return %[[C5]], %[[C6]], %[[MUTABLE]]
  return %0, %1, %2 : index, index, index
}

// -----

// CHECK: util.global private mutable @used0 = 5 : index
util.global private mutable @used0 = 5 : index
// CHECK: util.global private mutable @used1 : index
util.global private mutable @used1 : index
func.func @foo(%arg0: index, %arg1: index) -> (index, index) {
  // CHECK: %[[VALUE0:.+]] = util.global.load @used0 : index
  %0 = util.global.load @used0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @used1 : index
  %1 = util.global.load @used1 : index
  // CHECK: util.global.store %arg0, @used0 : index
  util.global.store %arg0, @used0 : index
  // CHECK: util.global.store %arg1, @used1 : index
  util.global.store %arg1, @used1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]]
  return %0, %1 : index, index
}

// -----

// CHECK-NOT: @unused0
util.global private mutable @unused0 = 5 : index
// CHECK-NOT: @unused1
util.global private mutable @unused1 : index
util.initializer {
  %c6 = arith.constant 6 : index
  // CHECK-NOT: util.global.store %c6, @unused1 : index
  util.global.store %c6, @unused1 : index
  util.return
}

// -----

// CHECK: util.global private @dupeCst0 {inlining_policy = #util.inline.never} = 5 : index
util.global private @dupeCst0 {inlining_policy = #util.inline.never} = 5 : index
// CHECK-NOT: util.global private @dupeCst1
util.global private @dupeCst1 {inlining_policy = #util.inline.never} = 5 : index
func.func @foo() -> (index, index) {
  // CHECK-DAG: %[[VALUE0:.+]] = util.global.load @dupeCst0
  %0 = util.global.load @dupeCst0 : index
  // CHECK-DAG: %[[VALUE1:.+]] = util.global.load @dupeCst0
  %1 = util.global.load @dupeCst1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]]
  return %0, %1 : index, index
}

// -----

// CHECK-NOT: util.global private @nondupeCst0
util.global private @nondupeCst0 = 6 : index
// CHECK-NOT: util.global private @nondupeCst1
util.global private @nondupeCst1 = 6 : index
// CHECK-NOT: util.initializer
util.initializer {
  %c7 = arith.constant 7 : index
  util.global.store %c7, @nondupeCst1 : index
  util.return
}
func.func @foo() -> (index, index) {
  // CHECK-DAG: %[[C6:.+]] = arith.constant 6 : index
  %0 = util.global.load @nondupeCst0 : index
  // CHECK-DAG: %[[C7:.+]] = arith.constant 7 : index
  %1 = util.global.load @nondupeCst1 : index
  // CHECK: return %[[C6]], %[[C7]]
  return %0, %1 : index, index
}
