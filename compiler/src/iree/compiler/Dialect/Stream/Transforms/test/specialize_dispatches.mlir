// RUN: iree-opt --split-input-file --iree-stream-specialize-dispatches %s | FileCheck %s

// Tests that operands of different types are placed into lookup tables of the
// appropriate type within the dispatch region and extracted based on a unique
// dispatch site identifier. Bindings and dynamic operands are untouched.

// CHECK-LABEL: @specializeEx
stream.executable private @specializeEx {
  stream.executable.export public @dispatch
  builtin.module  {
    // CHECK:  util.func public @dispatch(%[[BINDING:.+]]: !stream.binding, %[[A:.+]]: i32, %[[SITE:.+]]: index)
     util.func public @dispatch(%binding: !stream.binding, %a: i32, %b: index, %c: i1, %d: i1) {
      // CHECK-NEXT: %[[LUT_I32:.+]] = arith.constant dense<[
      // CHECK-SAME:   [20],
      // CHECK-SAME:   [40]
      // CHECK-SAME: ]> : tensor<2x1xi32>
      // CHECK-NEXT: %[[LUT_I1:.+]] = arith.constant dense<[
      // CHECK-SAME:   [false, true],
      // CHECK-SAME:   [true, false]
      // CHECK-SAME: ]> : tensor<2x2xi1>

      // CHECK: %[[B_I32:.+]] = tensor.extract %[[LUT_I32]][%[[SITE]], %c0] : tensor<2x1xi32>
      // CHECK: %[[B:.+]] = arith.index_cast %[[B_I32]] : i32 to index

      // CHECK: %[[C:.+]] = tensor.extract %[[LUT_I1]][%[[SITE]], %c0]
      // CHECK: %[[D:.+]] = tensor.extract %[[LUT_I1]][%[[SITE]], %c1]

      // CHECK-NEXT: util.optimization_barrier %[[BINDING]] : !stream.binding
      util.optimization_barrier %binding : !stream.binding
      // CHECK-NEXT: util.optimization_barrier %[[A]] : i32
      util.optimization_barrier %a : i32
      // CHECK-NEXT: util.optimization_barrier %[[B]] : index
      util.optimization_barrier %b : index
      // CHECK-NEXT: util.optimization_barrier %[[C]] : i1
      util.optimization_barrier %c : i1
      // CHECK-NEXT: util.optimization_barrier %[[D]] : i1
      util.optimization_barrier %d : i1
      util.return
    }
  }
}
// CHECK:  util.func public @specialize(%[[A:.+]]: i32)
util.func public @specialize(%a: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c20}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c20}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[A]], %c0_0 : i32, index)
    stream.cmd.dispatch @specializeEx::@dispatch[%c1, %c1, %c1](%a, %c20, %false, %true : i32, index, i1, i1) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
    // CHECK: stream.cmd.dispatch {{.+}}(%[[A]], %c1_1 : i32, index)
    stream.cmd.dispatch @specializeEx::@dispatch[%c1, %c1, %c1](%a, %c40, %true, %false : i32, index, i1, i1) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
  } => !stream.timepoint
  util.return
}
