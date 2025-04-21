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

// -----

util.func public @times_two(%in: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %0 = arith.muli %in, %c2 : i32
  util.return %0 : i32
}

module attributes {transform.with_named_sequence} {
  util.func public @double_func(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    util.return %0 : i32
  }
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.util.lookup_nearest_symbol_from_self @double_func : !transform.any_op
    %mul = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %in = transform.get_operand %mul[0] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %mul[0] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call inline_call %func(%in) -> %out after %mul {}
      : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
    transform.yield
  }
}

// CHECK-LABEL: util.func public @times_two
//  CHECK-SAME:   %[[IN:.+]]: i32
//       CHECK:   %[[ADD:.+]] = arith.addi %[[IN]], %[[IN]]
//       CHECK:   util.return %[[ADD]]

// Verify that the original function wasn't mangled by the call.
//       CHECK: module attributes {transform.with_named_sequence}
//       CHECK:   util.func public @double_func
//  CHECK-NEXT:     arith.addi

// -----

util.func public @pow_two(%in: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // An example with scf.for that requires a single block body.
  %pow2 = scf.for %i = %c0 to %in step %c1 iter_args(%iter = %c1) -> i32 : i32 {
    %0 = arith.muli %iter, %c2 : i32
    scf.yield %0 : i32
  }
  util.return %pow2 : i32
}

module attributes {transform.with_named_sequence} {
  util.func public @double_func(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    util.return %0 : i32
  }
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.util.lookup_nearest_symbol_from_self @double_func : !transform.any_op
    %mul = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %in = transform.get_operand %mul[0] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %mul[0] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call inline_call %func(%in) -> %out after %mul {}
      : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> ()
    transform.yield
  }
}

// CHECK-LABEL: util.func public @pow_two
//       CHECK:   scf.for {{.*}} iter_args(%[[ITER:.+]] =
//       CHECK:     %[[ADD:.+]] = arith.addi %[[ITER]], %[[ITER]]
//       CHECK:     scf.yield %[[ADD]]

//       CHECK: module attributes {transform.with_named_sequence}
//       CHECK:   util.func public @double_func
//  CHECK-NEXT:     arith.addi
