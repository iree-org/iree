// RUN: iree-opt --split-input-file --iree-vmvx-materialize-constants %s | FileCheck %s

// No constant block required.

// CHECK-NOT: @__set_constants
// CHECK: func private @no_constants_required()
func.func private @no_constants_required() {
  return
}

// -----

// Constant block required - globals and setter function should be added.

// CHECK: util.global private mutable @__constant_foo_ordinal {hal.executable.constant.key = "foo"} : i32
// CHECK: util.global private mutable @__constant_foo : i32
// CHECK: util.global private mutable @__constant_bar_ordinal {hal.executable.constant.key = "bar"} : i32
// CHECK: util.global private mutable @__constant_bar : i32
// CHECK: func @__set_constants(%[[BUFFER:.+]]: !util.buffer) {
// CHECK:   %[[BUFFER_SIZE:.+]] = arith.constant 8
// CHECK:   %[[FOO_ORDINAL:.+]] = util.global.load @__constant_foo_ordinal
// CHECK:   %[[FOO_OFFSET:.+]] = arith.muli %[[FOO_ORDINAL]], %c4
// CHECK:   %[[FOO_OFFSET_IDX:.+]] = arith.index_cast %[[FOO_OFFSET]]
// CHECK:   %[[FOO_VALUE:.+]] = util.buffer.load %[[BUFFER]][%[[FOO_OFFSET_IDX]] for {{.+}}] : !util.buffer{%[[BUFFER_SIZE]]}
// CHECK:   util.global.store %[[FOO_VALUE]], @__constant_foo : i32
// CHECK:   %[[BAR_ORDINAL:.+]] = util.global.load @__constant_bar_ordinal
// CHECK:   %[[BAR_OFFSET:.+]] = arith.muli %[[BAR_ORDINAL]], %c4
// CHECK:   %[[BAR_OFFSET_IDX:.+]] = arith.index_cast %[[BAR_OFFSET]]
// CHECK:   %[[BAR_VALUE:.+]] = util.buffer.load %[[BUFFER]][%[[BAR_OFFSET_IDX]] for {{.+}}] : !util.buffer{%[[BUFFER_SIZE]]}
// CHECK:   util.global.store %[[BAR_VALUE]], @__constant_bar : i32
// CHECK:   return
// CHECK: }

// CHECK-LABEL: func private @constant_user
func.func private @constant_user() {
  // CHECK: %[[FOO_LOADED:.+]] = util.global.load @__constant_foo
  %value_0 = hal.executable.constant.load "foo" : i32
  // CHECK: util.optimization_barrier %[[FOO_LOADED]]
  util.optimization_barrier %value_0 : i32

  // CHECK: %[[BAR_LOADED:.+]] = util.global.load @__constant_bar
  %value_1 = hal.executable.constant.load "bar" : i32
  // CHECK: util.optimization_barrier %[[BAR_LOADED]]
  util.optimization_barrier %value_1 : i32

  return
}
