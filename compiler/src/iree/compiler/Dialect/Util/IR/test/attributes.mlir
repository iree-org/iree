// RUN: iree-opt --split-input-file --mlir-print-local-scope %s | iree-opt --split-input-file --mlir-print-local-scope | FileCheck %s

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

// CHECK-LABEL: @uninitialized
builtin.module @uninitialized attributes {
  // CHECK: util.i32 = #util.uninitialized : i32
  util.i32 = #util.uninitialized : i32,
  // CHECK: util.tensor = #util.uninitialized : tensor<4xf32>
  util.tensor = #util.uninitialized : tensor<4xf32>
} {}
