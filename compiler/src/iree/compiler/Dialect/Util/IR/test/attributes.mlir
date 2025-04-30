// RUN: iree-opt --split-input-file --mlir-print-local-scope %s | iree-opt --split-input-file --mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: @assume_int
builtin.module @assume_int attributes {
  // CHECK: util.all = #util.int.assumption<umin = 1, umax = 2, udiv = 16>
  // CHECK-SAME: util.udiv = #util.int.assumption<udiv = 32>
  // CHECK-SAME: util.umax = #util.int.assumption<umax = 10>
  // CHECK-SAME: util.umin = #util.int.assumption<umin = 5>
  util.all = #util.int.assumption<umin = 1, umax = 2, udiv = 16>,
  util.udiv = #util.int.assumption<udiv = 32>,
  util.umax = #util.int.assumption<umax = 10>,
  util.umin = #util.int.assumption<umin = 5>
} {}

// -----

// CHECK-LABEL: @byte_pattern
builtin.module @byte_pattern attributes {
  // CHECK: r0 = #util.byte_pattern<0> : i8
  util.r0 = #util.byte_pattern<0> : i8,
  // CHECK: r1 = #util.byte_pattern<0> : tensor<28xi8>
  util.r1 = #util.byte_pattern<0> : tensor<28xi8>,
  // CHECK: r2 = #util.byte_pattern<6> : tensor<33x4xi4>
  util.r2 = #util.byte_pattern<6> : tensor<33x4xi4>
} {}

// -----

// CHECK-LABEL: @byte_range
builtin.module @byte_range attributes {
  // CHECK: br0 = #util.byte_range<123, 456>
  util.br0 = #util.byte_range<123, 456>
} {}

// -----

// CHECK-LABEL: @composite
builtin.module @composite attributes {
  // CHECK: c0 = #util.composite<0xi8, []>
  util.c0 = #util.composite<0xi8, []>,
  //      CHECK: c1 = #util.composite<1xi8, [
  // CHECK-NEXT:   dense<0> : vector<1xi8>,
  // CHECK-NEXT: ]>
  util.c1 = #util.composite<1xi8, [dense<0> : vector<1xi8>]>,
  //      CHECK: c2 = #util.composite<4xi8, [
  // CHECK-NEXT:   dense<0> : vector<1xi8>,
  // CHECK-NEXT:   dense<1> : vector<1xi8>,
  // CHECK-NEXT:   dense<[2, 3]> : vector<2xi8>,
  // CHECK-NEXT: ]>
  util.c2 = #util.composite<4xi8, [
    dense<0> : vector<1xi8>,
    dense<1> : vector<1xi8>,
    dense<[2, 3]> : vector<2xi8>,
  ]>
} {}

// -----

// CHECK-LABEL: @null
builtin.module @null attributes {
  // CHECK: util.buffer = #util.null : !util.buffer
  util.buffer = #util.null : !util.buffer
} {}

// -----

// CHECK-LABEL: @uninitialized
builtin.module @uninitialized attributes {
  // CHECK: util.i32 = #util.uninitialized : i32
  util.i32 = #util.uninitialized : i32,
  // CHECK: util.tensor = #util.uninitialized : tensor<4xf32>
  util.tensor = #util.uninitialized : tensor<4xf32>
} {}

// -----

// CHECK-LABEL: @preprocessing_pipeline
builtin.module @preprocessing_pipeline {
  // CHECK: util.func public @main() attributes {preprocessing_pipeline = #util.preprocessing_pipeline<"some-iree-preprocessing-pass-pipeline">}
  util.func public @main() attributes {preprocessing_pipeline = #util.preprocessing_pipeline<"some-iree-preprocessing-pass-pipeline">} {
    util.return
  }
}
