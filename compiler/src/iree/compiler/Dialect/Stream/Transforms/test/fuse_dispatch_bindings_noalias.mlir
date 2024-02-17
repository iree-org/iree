// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-stream-fuse-dispatch-bindings)' %s | FileCheck %s

// TODO(benvanik): remove this file when aliasing mutable bindings is fixed.

// Tests that bindings that are duplicated at all dispatch sites are folded
// so long as they are not mutable.

#noaliasConfig = #stream.resource_config<{
  alias_mutable_bindings = false
}>

// CHECK-LABEL: @deduplicateBindingsEx
stream.executable private @deduplicateBindingsEx {
  stream.executable.export public @dispatch attributes {stream.resources = #noaliasConfig}
  builtin.module  {
    // CHECK:  util.func public @dispatch(%[[BINDING_A:.+]]: !stream.binding, %[[BINDING_C:.+]]: !stream.binding,
    // CHECK-SAME:           %[[OFFSET_A:.+]]: index, %[[OFFSET_B:.+]]: index, %[[OFFSET_C:.+]]: index, %[[OPERAND:.+]]: index)
     util.func public @dispatch(%binding_a: !stream.binding, %binding_b: !stream.binding, %binding_c: !stream.binding, %operand: index) {
      %c0 = arith.constant 0 : index
      %c20 = arith.constant 20 : index
      %c40 = arith.constant 40 : index

      // CHECK: %[[SUBSPAN_A:.+]] = stream.binding.subspan %[[BINDING_A]][%[[OFFSET_A]]]
      %subspan_a = stream.binding.subspan %binding_a[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_A]]
      util.optimization_barrier %subspan_a : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_B:.+]] = arith.addi %[[OFFSET_B]], %c20
      // CHECK-NEXT: %[[SUBSPAN_B:.+]] = stream.binding.subspan %[[BINDING_A]][%[[SUM_OFFSET_B]]]
      %subspan_b = stream.binding.subspan %binding_b[%c20] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_B]]
      util.optimization_barrier %subspan_b : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_C:.+]] = arith.addi %[[OFFSET_C]], %c40
      // CHECK-NEXT: %[[SUBSPAN_C:.+]] = stream.binding.subspan %[[BINDING_C]][%[[SUM_OFFSET_C]]]
      %subspan_c = stream.binding.subspan %binding_c[%c40] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_C]]
      util.optimization_barrier %subspan_c : !flow.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK-NEXT: util.optimization_barrier %[[OPERAND]] : index
      util.optimization_barrier %operand : index
      util.return
    }
  }
}
// CHECK:  util.func public @deduplicateBindings(%[[OPERAND:.+]]: index)
util.func public @deduplicateBindings(%operand: index) {
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
  // CHECK: stream.cmd.execute
  %result_timepoint = stream.cmd.execute
      // CHECK-SAME: with(%[[ALLOC0]] as %[[CAPTURE0:.+]]: !stream.resource<transient>{%c200})
      with(%alloc0 as %capture0: !stream.resource<transient>{%c200}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%c40, %c80, %c0, %[[OPERAND]] : index, index, index, index)
    stream.cmd.dispatch @deduplicateBindingsEx::@dispatch[%c1, %c1, %c1](%operand : index) {
      // CHECK-NEXT: ro %[[CAPTURE0]][%c0
      ro %capture0[%c40 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NOT: ro %[[CAPTURE0]][%c0
      ro %capture0[%c80 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NEXT: rw %[[CAPTURE0]]
      rw %capture0[%c0 for %c20] : !stream.resource<transient>{%c200}
    }
    // CHECK: stream.cmd.dispatch {{.+}}(%c120, %c160, %c20, %[[OPERAND]] : index, index, index, index)
    stream.cmd.dispatch @deduplicateBindingsEx::@dispatch[%c1, %c1, %c1](%operand : index) {
      // CHECK-NEXT: ro %[[CAPTURE0]][%c0
      ro %capture0[%c120 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NOT: ro %[[CAPTURE0]][%c0
      ro %capture0[%c160 for %c20] : !stream.resource<transient>{%c200},
      // CHECK-NEXT: rw %[[CAPTURE0]]
      rw %capture0[%c20 for %c20] : !stream.resource<transient>{%c200}
    }
  } => !stream.timepoint
  util.return
}
