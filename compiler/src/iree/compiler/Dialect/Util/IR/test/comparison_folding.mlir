// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @cmp_eq_same
util.func @cmp_eq_same(%value: !util.buffer) -> i1 {
  // CHECK-NOT: util.cmp.eq
  %result = util.cmp.eq %value, %value : !util.buffer
  // CHECK: util.return %true
  util.return %result : i1
}

// -----

// CHECK-LABEL: @cmp_eq_null
util.func @cmp_eq_null() -> i1 {
  %null0 = util.null : !util.buffer
  %null1 = util.null : !util.buffer
  // CHECK-NOT: util.cmp.eq
  %result = util.cmp.eq %null0, %null1 : !util.buffer
  // CHECK: util.return %true
  util.return %result : i1
}

// -----

// CHECK-LABEL: @cmp_ne_same
util.func @cmp_ne_same(%value: !util.buffer) -> i1 {
  // CHECK-NOT: util.cmp.ne
  %result = util.cmp.ne %value, %value : !util.buffer
  // CHECK: util.return %false
  util.return %result : i1
}

// -----

// CHECK-LABEL: @cmp_ne_null
util.func @cmp_ne_null() -> i1 {
  %null0 = util.null : !util.buffer
  %null1 = util.null : !util.buffer
  // CHECK-NOT: util.cmp.ne
  %result = util.cmp.ne %null0, %null1 : !util.buffer
  // CHECK: util.return %false
  util.return %result : i1
}
