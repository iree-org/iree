// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-unroll-annotated-loops))" \
// RUN:   --allow-unregistered-dialect | FileCheck %s

func.func @basic_unroll() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    "unregistered.loop_body"(%i) : (index) -> ()
  } {unroll_loop}
  return
}

// CHECK-LABEL:   func.func @basic_unroll
//       CHECK:     "unregistered.loop_body"(%c0)
//       CHECK:     "unregistered.loop_body"(%c1)
//       CHECK:     "unregistered.loop_body"(%c2)

// -----

func.func @no_annotation_no_unroll() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    "unregistered.loop_body"(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @no_annotation_no_unroll
//       CHECK:     scf.for
//       CHECK:       "unregistered.loop_body"

// -----

func.func @no_unroll_dynamic_trip(%x: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %x step %c1 {
    "unregistered.loop_body"(%i) : (index) -> ()
  } {unroll_loop}
  return
}

// CHECK-LABEL:   func.func @no_unroll_dynamic_trip
//       CHECK:     scf.for
//       CHECK:       "unregistered.loop_body"
//   CHECK-NOT:     unroll_loop

// -----

func.func @unroll_non_normalized() {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c5 to %c10 step %c2 {
    "unregistered.loop_body"(%i) : (index) -> ()
  } {unroll_loop}
  return
}

// CHECK-LABEL:   func.func @unroll_non_normalized
//       CHECK:     "unregistered.loop_body"(%c5)
//       CHECK:     "unregistered.loop_body"(%c7)
//       CHECK:     "unregistered.loop_body"(%c9)

// -----

func.func @unroll_iter_arg() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init = arith.constant 1 : i32
  %0 = scf.for %i = %c0 to %c3 step %c1 iter_args(%it = %init) -> i32 {
    %1 = "unregistered.loop_body"(%it) : (i32) -> (i32)
    scf.yield %1 : i32
  } {unroll_loop}
  return %0 : i32
}

// CHECK-LABEL:   func.func @unroll_iter_arg
//       CHECK:     %[[INIT:.+]] = arith.constant 1 : i32
//       CHECK:     %[[IT0:.+]] = "unregistered.loop_body"(%[[INIT]])
//       CHECK:     %[[IT1:.+]] = "unregistered.loop_body"(%[[IT0]])
//       CHECK:     %[[IT2:.+]] = "unregistered.loop_body"(%[[IT1]])
//       CHECK:     return %[[IT2]]

// -----

func.func @nested_unroll() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      "unregistered.loop_body"(%i, %j) : (index, index) -> ()
    } {unroll_loop}
  } {unroll_loop}
  return
}

// CHECK-LABEL:   func.func @nested_unroll
//       CHECK:     "unregistered.loop_body"(%c0, %c0)
//       CHECK:     "unregistered.loop_body"(%c0, %c1)
//       CHECK:     "unregistered.loop_body"(%c1, %c0)
//       CHECK:     "unregistered.loop_body"(%c1, %c1)
