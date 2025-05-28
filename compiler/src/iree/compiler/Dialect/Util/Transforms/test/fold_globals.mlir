// RUN: iree-opt --split-input-file --iree-util-fold-globals %s | FileCheck %s

// CHECK: util.global public mutable @uniformConstants = 5 : index
util.global public mutable @uniformConstants : index
util.func @foo() {
  %c5 = arith.constant 5 : index
  // CHECK-NOT: util.global.store %c5, @uniformConstants : index
  util.global.store %c5, @uniformConstants : index
  util.return
}
util.func @bar() {
  %c5 = arith.constant 5 : index
  // CHECK-NOT: util.global.store %c5, @uniformConstants : index
  util.global.store %c5, @uniformConstants : index
  util.return
}

// -----

// CHECK: util.global public mutable @nonuniformConstants : index
util.global public mutable @nonuniformConstants : index
util.func @foo() {
  %c5 = arith.constant 5 : index
  // CHECK: util.global.store %c5, @nonuniformConstants : index
  util.global.store %c5, @nonuniformConstants : index
  util.return
}
util.func @bar() {
  %c6 = arith.constant 6 : index
  // CHECK: util.global.store %c6, @nonuniformConstants : index
  util.global.store %c6, @nonuniformConstants : index
  util.return
}

// -----

// CHECK: util.global private @chained0 : index
util.global private mutable @chained0 : index
// CHECK-NOT: util.global private mutable @chained1 : index
util.global private mutable @chained1 : index
util.func @foo() -> index {
  // CHECK: %[[VALUE:.+]] = util.global.load immutable @chained0 : index
  %0 = util.global.load @chained0 : index
  // CHECK-NOT: util.global.store
  util.global.store %0, @chained1 : index
  // CHECK-NEXT: return %[[VALUE]]
  util.return %0 : index
}

// -----

// CHECK: util.global public mutable @unchained0 : index
util.global public mutable @unchained0 : index
// CHECK: util.global public mutable @unchained1 : index
util.global public mutable @unchained1 : index
util.func @foo() {
  // CHECK: %[[VALUE:.+]] = util.global.load @unchained0 : index
  %0 = util.global.load @unchained0 : index
  // CHECK: util.global.store %[[VALUE]], @unchained1 : index
  util.global.store %0, @unchained1 : index
  util.return
}
util.func @bar(%arg0: index) {
  // CHECK: util.global.store %arg0, @unchained1 : index
  util.global.store %arg0, @unchained1 : index
  util.return
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
util.func @foo(%arg0: index) -> (index, index, index) {
  // CHECK-DAG: %[[C5:.+]] = arith.constant 5
  %0 = util.global.load @immutable0 : index
  // CHECK-DAG: %[[C6:.+]] = arith.constant 6
  %1 = util.global.load @immutable1 : index
  // CHECK: %[[MUTABLE:.+]] = util.global.load @mutable
  %2 = util.global.load @mutable : index
  // CHECK: util.global.store %arg0, @mutable
  util.global.store %arg0, @mutable : index
  // CHECK: return %[[C5]], %[[C6]], %[[MUTABLE]]
  util.return %0, %1, %2 : index, index, index
}

// -----

// CHECK: util.global private @immutable_initializer_local
util.global private mutable @immutable_initializer_local : index
// CHECK: util.global private @immutable_initializer_callee
util.global private mutable @immutable_initializer_callee : index
// CHECK: util.global private mutable @mutable : index
util.global private mutable @mutable : index
util.func private @generate_value() -> index
util.initializer {
  %value = util.call @generate_value() : () -> index
  util.global.store %value, @immutable_initializer_local : index
  util.return
}
util.func @public_func() -> (index, index, index) {
  util.call @public_callee() : () -> ()
  // CHECK-DAG: %[[LOCAL:.+]] = util.global.load immutable @immutable_initializer_local
  %0 = util.global.load @immutable_initializer_local : index
  // CHECK-DAG: %[[CALLEE:.+]] = util.global.load immutable @immutable_initializer_callee
  %1 = util.global.load @immutable_initializer_callee : index
  // CHECK-DAG: %[[MUTABLE:.+]] = util.global.load @mutable
  %2 = util.global.load @mutable : index
  // CHECK: return %[[LOCAL]], %[[CALLEE]], %[[MUTABLE]]
  util.return %0, %1, %2 : index, index, index
}
util.func private @public_callee() {
  %value = util.call @generate_value() : () -> index
  util.global.store %value, @mutable : index
  util.return
}

// -----

// CHECK: util.global private mutable @used0 = 5 : index
util.global private mutable @used0 = 5 : index
// CHECK: util.global private mutable @used1 : index
util.global private mutable @used1 : index
// CHECK: util.global private @referenced : index
util.global private @referenced : index
util.func @foo(%arg0: index, %arg1: index) -> (index, index) attributes {
  some.attr = @referenced
} {
  // CHECK: %[[VALUE0:.+]] = util.global.load @used0 : index
  %0 = util.global.load @used0 : index
  // CHECK: %[[VALUE1:.+]] = util.global.load @used1 : index
  %1 = util.global.load @used1 : index
  // CHECK: util.global.store %arg0, @used0 : index
  util.global.store %arg0, @used0 : index
  // CHECK: util.global.store %arg1, @used1 : index
  util.global.store %arg1, @used1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]]
  util.return %0, %1 : index, index
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

builtin.module attributes {
  some.attr = @only_ref_on_module
} {
  // CHECK: @only_ref_on_module
  util.global private @only_ref_on_module : index
}

// -----

builtin.module @named_module attributes {
  some.attr = @named_module::@only_ref_on_module
} {
  // CHECK: @only_ref_on_module
  util.global private @only_ref_on_module : index
}

// -----

// CHECK: util.global private @dupeCst0 {inlining_policy = #util.inline.never} = 5 : index
util.global private @dupeCst0 {inlining_policy = #util.inline.never} = 5 : index
// CHECK-NOT: util.global private @dupeCst1
util.global private @dupeCst1 {inlining_policy = #util.inline.never} = 5 : index
util.func @foo() -> (index, index) attributes {
  some.attr = @dupeCst1
} {
  // CHECK-DAG: %[[VALUE0:.+]] = util.global.load immutable @dupeCst0
  %0 = util.global.load @dupeCst0 : index
  // CHECK-DAG: %[[VALUE1:.+]] = util.global.load immutable @dupeCst0
  %1 = util.global.load @dupeCst1 : index
  // CHECK-DAG: util.optimization_barrier
  // CHECK-SAME: op.attr = @dupeCst0
  util.optimization_barrier {op.attr = @dupeCst1} %1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]]
  util.return %0, %1 : index, index
}

// -----

// Tests that uninitialized globals have stores folded into their initializer.
// Initialized globals are not changed. This rule relies on globals having
// undefined values until stored - if the initializer was 0 by default then this
// would have correctness issues unless we could guarantee that no other
// initialization-time code has potentially read the value before the store.

// CHECK: util.global private @dontFoldInitialized = 6 : index
util.global private @dontFoldInitialized = 6 : index
// CHECK-NOT: util.global private @foldUnintialized
util.global private @foldUnintialized : index
// CHECK: util.initializer
util.initializer {
  // CHECK-DAG: %[[C7:.+]] = arith.constant 7
  %c7 = arith.constant 7 : index
  // CHECK-DAG: util.global.store %[[C7]], @dontFoldInitialized
  util.global.store %c7, @dontFoldInitialized : index
  %c8 = arith.constant 8 : index
  // CHECK-NOT: util.global.store %{{.+}}, @foldUnintialized
  util.global.store %c8, @foldUnintialized : index
  util.return
}
util.func @foo() -> (index, index) {
  // CHECK-DAG: %[[UNFOLDED:.+]] = util.global.load immutable @dontFoldInitialized
  %0 = util.global.load @dontFoldInitialized : index
  // CHECK-DAG: %[[FOLDED:.+]] = arith.constant 8 : index
  %1 = util.global.load @foldUnintialized : index
  // CHECK: return %[[UNFOLDED]], %[[FOLDED]]
  util.return %0, %1 : index, index
}

// -----

// Tests that globals with the same value and dialect attrs fold while ones
// with different dialect attrs do not. We could have an interface to make this
// controllable by the attributes.

// CHECK: util.global private @dupeCst0
// CHECK-SAME: some.attr = 100 : index
util.global private @dupeCst0 {
  inlining_policy = #util.inline.never,
  some.attr = 100 : index
} = 5 : index
// CHECK-NOT: util.global private @dupeCst1
util.global private @dupeCst1 {
  inlining_policy = #util.inline.never,
  some.attr = 100 : index
} = 5 : index
// CHECK: util.global private @nondupeCst0
util.global private @nondupeCst0 {
  inlining_policy = #util.inline.never
} = 5 : index
// CHECK: util.global private @nondupeCst1
util.global private @nondupeCst1 {
  inlining_policy = #util.inline.never,
  some.attr = 123 : index
} = 5 : index
util.func @foo() -> (index, index, index, index) {
  // CHECK-DAG: %[[VALUE0:.+]] = util.global.load immutable @dupeCst0
  %0 = util.global.load @dupeCst0 : index
  // CHECK-DAG: %[[VALUE1:.+]] = util.global.load immutable @dupeCst0
  %1 = util.global.load @dupeCst1 : index
  // CHECK-DAG: %[[VALUE2:.+]] = util.global.load immutable @nondupeCst0
  %2 = util.global.load @nondupeCst0 : index
  // CHECK-DAG: %[[VALUE3:.+]] = util.global.load immutable @nondupeCst1
  %3 = util.global.load @nondupeCst1 : index
  // CHECK: return %[[VALUE0]], %[[VALUE1]], %[[VALUE2]], %[[VALUE3]]
  util.return %0, %1, %2, %3 : index, index, index, index
}
