// RUN: iree-opt --transform-interpreter %s --split-input-file --verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.util.lookup_nearest_symbol_from_self @__transform_main : !transform.any_op
    transform.print %0 : !transform.any_op
    transform.yield
  }
}

// CHECK: IR printer:
// CHECK-NEXT: transform.named_sequence @__transform_main

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
// expected-error@+1 {{could not find symbol @non_existent_symbol}}
    %0 = transform.util.lookup_nearest_symbol_from_self @non_existent_symbol : !transform.any_op
    transform.print %0 : !transform.any_op
    transform.yield
  }
}
