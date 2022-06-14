// RUN: iree-opt --split-input-file --mlir-print-local-scope %s | iree-opt --split-input-file --mlir-print-local-scope | FileCheck %s

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
