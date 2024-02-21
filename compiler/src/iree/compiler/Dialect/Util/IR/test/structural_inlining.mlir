// RUN: iree-opt --split-input-file --inline %s | FileCheck %s

// Tests inlining a call into an initializer.

// CHECK: util.initializer
util.initializer {
  // CHECK: %[[VALUE:.+]] = arith.constant 123
  %0 = arith.constant 123 : i32
  // CHECK-NOT: util.call @from_initializer
  %1 = util.call @from_initializer(%0) : (i32) -> i32
  // CHECK: util.optimization_barrier %[[VALUE]]
  util.optimization_barrier %1 : i32
  util.return
}
// CHECK-NOT: util.func private @from_initializer
util.func private @from_initializer(%arg0: i32) -> i32 {
  util.return %arg0 : i32
}

// -----

// Tests inlining a call into another call.

// CHECK: util.func public @caller(%[[ARG0:.+]]: i32) -> i32
util.func public @caller(%arg0: i32) -> i32 {
  // CHECK-NOT: util.call @callee
  %0 = util.call @callee(%arg0) : (i32) -> i32
  // CHECK: util.return %[[ARG0]]
  util.return %0 : i32
}
// CHECK-NOT: util.func private @callee
util.func private @callee(%arg0: i32) -> i32 {
  util.return %arg0 : i32
}

// -----

// Tests that `#util.inline.never` blocks inlining a call into another call
// when placed on the callee.

// CHECK: util.func public @noinline_caller
util.func public @noinline_caller(%arg0: i32) -> i32 {
  // CHECK: util.call @noinline_callee
  %0 = util.call @noinline_callee(%arg0) : (i32) -> i32
  util.return %0 : i32
}
// CHECK: util.func private @noinline_callee
util.func private @noinline_callee(%arg0: i32) -> i32 attributes {
  inlining_policy = #util.inline.never
} {
  util.return %arg0 : i32
}

// -----

// Tests inlining a recursive function doesn't explode.

// CHECK: util.func public @recursive_fn(%[[ARG0:.+]]: i32) -> i32
util.func public @recursive_fn(%arg0: i32) -> i32 {
  // CHECK: util.call @recursive_fn
  %0 = util.call @recursive_fn(%arg0) : (i32) -> i32
  util.return %0 : i32
}
