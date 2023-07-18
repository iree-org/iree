// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-stream-fuse-dispatch-bindings)' %s | FileCheck %s

// Test that bindings that are unique are rebased to the widest possible access
// and to have a 0 offset by passing in the actual offset as operands. The
// --iree-stream-fold-uniform-operands pass will take care of inlining/deduping
// the operands that we insert.
//
// TODO(benvanik): do this in a canonicalize bindings pass. This should not be
// happening here!

#aliasConfig = #stream.resource_config<{
  alias_mutable_bindings = true
}>

// CHECK-LABEL: @rebaseBindingsEx
stream.executable private @rebaseBindingsEx {
  stream.executable.export public @dispatch attributes {stream.resources = #aliasConfig}
  builtin.module  {
    // CHECK: func.func @dispatch(%[[BINDING_A:.+]]: !stream.binding, %[[BINDING_B:.+]]: !stream.binding,
    // CHECK-SAME:           %[[OFFSET_A:.+]]: index, %[[OFFSET_B:.+]]: index, %[[OPERAND:.+]]: index)
    func.func @dispatch(%binding_a: !stream.binding, %binding_b: !stream.binding, %operand: index) {
      %c0 = arith.constant 0 : index
      %c20 = arith.constant 20 : index

      // CHECK: %[[SUBSPAN_A:.+]] = stream.binding.subspan %[[BINDING_A]][%[[OFFSET_A]]]
      %subspan_a = stream.binding.subspan %binding_a[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_A]]
      util.optimization_barrier %subspan_a : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_B:.+]] = arith.addi %c20, %[[OFFSET_B]]
      // CHECK-NEXT: %[[SUBSPAN_B:.+]] = stream.binding.subspan %[[BINDING_B]][%[[SUM_OFFSET_B]]]
      %subspan_b = stream.binding.subspan %binding_b[%c20] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_B]]
      util.optimization_barrier %subspan_b : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK-NEXT: util.optimization_barrier %[[OPERAND]] : index
      util.optimization_barrier %operand : index
      return
    }
  }
}
// CHECK: func.func @rebaseBindings(%[[OPERAND:.+]]: index)
func.func @rebaseBindings(%operand: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  %c120 = arith.constant 120 : index
  %c160 = arith.constant 160 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[ALLOC0:.+]] = stream.resource.alloc
  %alloc0 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  // CHECK-NEXT: %[[ALLOC1:.+]] = stream.resource.alloc
  %alloc1 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  // CHECK: stream.cmd.execute
  %result_timepoint = stream.cmd.execute
      // CHECK-SAME: with(%[[ALLOC0]] as %[[CAPTURE0:.+]]: !stream.resource<transient>{%c200},
      // CHECK-SAME:      %[[ALLOC1]] as %[[CAPTURE1:.+]]: !stream.resource<transient>{%c200})
      with(%alloc0 as %capture0: !stream.resource<transient>{%c200},
           %alloc1 as %capture1: !stream.resource<transient>{%c200}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%c40, %c80, %[[OPERAND]] : index, index, index)
    stream.cmd.dispatch @rebaseBindingsEx::@dispatch[%c1, %c1, %c1](%operand : index) {
      // CHECK-NEXT: rw %[[CAPTURE0]][%c0
      rw %capture0[%c40 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NEXT: rw %[[CAPTURE1]][%c0
      rw %capture1[%c80 for %c20] : !stream.resource<transient>{%c200}
    }
    // CHECK: stream.cmd.dispatch {{.+}}(%c120, %c160, %[[OPERAND]] : index, index, index)
    stream.cmd.dispatch @rebaseBindingsEx::@dispatch[%c1, %c1, %c1](%operand : index) {
      // CHECK-NEXT: rw %[[CAPTURE0]][%c0
      rw %capture0[%c120 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NEXT: rw %[[CAPTURE1]][%c0
      ro %capture1[%c160 for %c20] : !stream.resource<transient>{%c200}
    }
  } => !stream.timepoint
  return
}

// -----

// Tests that bindings that are duplicated at all dispatch sites are folded.
// This will happen even if the offsets differ as we are rebasing them to 0 and
// moving the offsetting into the dispatched function. To ensure that we are
// offseting things correctly we have pre-offset subspans in the dispatch region
// that must have the host binding offsets added to them.
//
// NOTE: the offset operands are inserted at the start of the operand list and
// in the order of the deduplicated bindings instead of the original order.
// This is a bit weird and it would be nice to preserve the order in the future.

#aliasConfig = #stream.resource_config<{
  alias_mutable_bindings = true
}>

// CHECK-LABEL: @deduplicateBindingsEx
stream.executable private @deduplicateBindingsEx {
  stream.executable.export public @dispatch attributes {stream.resources = #aliasConfig}
  builtin.module  {
    // CHECK: func.func @dispatch(%[[BINDING_A:.+]]: !stream.binding, %[[BINDING_B:.+]]: !stream.binding,
    // CHECK-SAME:           %[[OFFSET_A:.+]]: index, %[[OFFSET_C:.+]]: index, %[[OFFSET_B:.+]]: index, %[[OPERAND:.+]]: index)
    func.func @dispatch(%binding_a: !stream.binding, %binding_b: !stream.binding, %binding_c: !stream.binding, %operand: index) {
      %c0 = arith.constant 0 : index
      %c20 = arith.constant 20 : index
      %c40 = arith.constant 40 : index

      // CHECK: %[[SUBSPAN_A:.+]] = stream.binding.subspan %[[BINDING_A]][%[[OFFSET_A]]]
      %subspan_a = stream.binding.subspan %binding_a[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_A]]
      util.optimization_barrier %subspan_a : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_B:.+]] = arith.addi %c20, %[[OFFSET_B]]
      // CHECK-NEXT: %[[SUBSPAN_B:.+]] = stream.binding.subspan %[[BINDING_B]][%[[SUM_OFFSET_B]]]
      %subspan_b = stream.binding.subspan %binding_b[%c20] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_B]]
      util.optimization_barrier %subspan_b : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_C:.+]] = arith.addi %c40, %[[OFFSET_C]]
      // CHECK-NEXT: %[[SUBSPAN_C:.+]] = stream.binding.subspan %[[BINDING_A]][%[[SUM_OFFSET_C]]]
      %subspan_c = stream.binding.subspan %binding_c[%c40] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_C]]
      util.optimization_barrier %subspan_c : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK-NEXT: util.optimization_barrier %[[OPERAND]] : index
      util.optimization_barrier %operand : index
      return
    }
  }
}
// CHECK: func.func @deduplicateBindings(%[[OPERAND:.+]]: index)
func.func @deduplicateBindings(%operand: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  %c120 = arith.constant 120 : index
  %c160 = arith.constant 160 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[ALLOC0:.+]] = stream.resource.alloc
  %alloc0 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  // CHECK-NEXT: %[[ALLOC1:.+]] = stream.resource.alloc
  %alloc1 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  // CHECK: stream.cmd.execute
  %result_timepoint = stream.cmd.execute
      // CHECK-SAME: with(%[[ALLOC0]] as %[[CAPTURE0:.+]]: !stream.resource<transient>{%c200},
      // CHECK-SAME:      %[[ALLOC1]] as %[[CAPTURE1:.+]]: !stream.resource<transient>{%c200})
      with(%alloc0 as %capture0: !stream.resource<transient>{%c200},
           %alloc1 as %capture1: !stream.resource<transient>{%c200}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%c40, %c0, %c80, %[[OPERAND]] : index, index, index, index)
    stream.cmd.dispatch @deduplicateBindingsEx::@dispatch[%c1, %c1, %c1](%operand : index) {
      // CHECK-NEXT: rw %[[CAPTURE0]][%c0
      rw %capture0[%c40 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NEXT: rw %[[CAPTURE1]][%c0
      rw %capture1[%c80 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NOT: rw %[[CAPTURE0]]
      ro %capture0[%c0 for %c20] : !stream.resource<transient>{%c200}
    }
    // CHECK: stream.cmd.dispatch {{.+}}(%c120, %c20, %c160, %[[OPERAND]] : index, index, index, index)
    stream.cmd.dispatch @deduplicateBindingsEx::@dispatch[%c1, %c1, %c1](%operand : index) {
      // CHECK-NEXT: rw %[[CAPTURE0]][%c0
      rw %capture0[%c120 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NEXT: rw %[[CAPTURE1]][%c0
      ro %capture1[%c160 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NOT: rw %[[CAPTURE0]]
      rw %capture0[%c20 for %c20] : !stream.resource<transient>{%c200}
    }
  } => !stream.timepoint
  return
}
