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
    // CHECK: util.func public @dispatch(%[[BINDING_A:.+]]: !stream.binding, %[[BINDING_B:.+]]: !stream.binding,
    // CHECK-SAME: %[[OFFSET_A:.+]]: index, %[[OFFSET_B:.+]]: index, %[[OPERAND:.+]]: index)
     util.func public @dispatch(%binding_a: !stream.binding, %binding_b: !stream.binding, %operand: index) {
      %c0 = arith.constant 0 : index
      %c20 = arith.constant 20 : index

      // CHECK: %[[SUBSPAN_A:.+]] = stream.binding.subspan %[[BINDING_A]][%[[OFFSET_A]]]
      %subspan_a = stream.binding.subspan %binding_a[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_A]]
      util.optimization_barrier %subspan_a : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_B:.+]] = arith.addi %[[OFFSET_B]], %c20
      // CHECK-NEXT: %[[SUBSPAN_B:.+]] = stream.binding.subspan %[[BINDING_B]][%[[SUM_OFFSET_B]]]
      %subspan_b = stream.binding.subspan %binding_b[%c20] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_B]]
      util.optimization_barrier %subspan_b : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK-NEXT: util.optimization_barrier %[[OPERAND]] : index
      util.optimization_barrier %operand : index
      util.return
    }
  }
}
// CHECK: util.func public @rebaseBindings(%[[OPERAND:.+]]: index)
util.func public @rebaseBindings(%operand: index) {
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
  util.return
}

// -----

// Tests that bindings that are duplicated at all dispatch sites are folded.
// This will happen even if the offsets differ as we are rebasing them to 0 and
// moving the offsetting into the dispatched function. To ensure that we are
// offsetting things correctly we have pre-offset subspans in the dispatch region
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
    // CHECK: util.func public @dispatch(%[[BINDING_A:.+]]: !stream.binding, %[[BINDING_B:.+]]: !stream.binding,
    // CHECK-SAME: %[[OFFSET_A:.+]]: index, %[[OFFSET_C:.+]]: index, %[[OFFSET_B:.+]]: index, %[[OPERAND:.+]]: index)
     util.func public @dispatch(%binding_a: !stream.binding, %binding_b: !stream.binding, %binding_c: !stream.binding, %operand: index) {
      %c0 = arith.constant 0 : index
      %c20 = arith.constant 20 : index
      %c40 = arith.constant 40 : index

      // CHECK: %[[SUBSPAN_A:.+]] = stream.binding.subspan %[[BINDING_A]][%[[OFFSET_A]]]
      %subspan_a = stream.binding.subspan %binding_a[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_A]]
      util.optimization_barrier %subspan_a : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_B:.+]] = arith.addi %[[OFFSET_B]], %c20
      // CHECK-NEXT: %[[SUBSPAN_B:.+]] = stream.binding.subspan %[[BINDING_B]][%[[SUM_OFFSET_B]]]
      %subspan_b = stream.binding.subspan %binding_b[%c20] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_B]]
      util.optimization_barrier %subspan_b : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK: %[[SUM_OFFSET_C:.+]] = arith.addi %[[OFFSET_C]], %c40
      // CHECK-NEXT: %[[SUBSPAN_C:.+]] = stream.binding.subspan %[[BINDING_A]][%[[SUM_OFFSET_C]]]
      %subspan_c = stream.binding.subspan %binding_c[%c40] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      // CHECK-NEXT: util.optimization_barrier %[[SUBSPAN_C]]
      util.optimization_barrier %subspan_c : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>

      // CHECK-NEXT: util.optimization_barrier %[[OPERAND]] : index
      util.optimization_barrier %operand : index
      util.return
    }
  }
}
// CHECK: util.func public @deduplicateBindings(%[[OPERAND:.+]]: index)
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
  util.return
}

// -----

// Differing alias patterns get separate executables; repeated patterns reuse one.
// CHECK-LABEL: stream.executable private @specializeBindingsEx {
// CHECK: util.func public @dispatch(%[[SPEC_A:[^:,]+]]: !stream.binding, %[[SPEC_B:[^:,]+]]: !stream.binding,
// CHECK-SAME: %[[SPEC_OFF_A:[^:,]+]]: index, %[[SPEC_OFF_C:[^:,]+]]: index, %[[SPEC_OFF_B:[^:,]+]]: index, %{{[^:,]+}}: index)
// CHECK: stream.binding.subspan %[[SPEC_A]][%[[SPEC_OFF_A]]]
// CHECK: %[[SPEC_SUM_B:.*]] = arith.addi %[[SPEC_OFF_B]], %c20
// CHECK: stream.binding.subspan %[[SPEC_B]][%[[SPEC_SUM_B]]]
// CHECK: %[[SPEC_SUM_C:.*]] = arith.addi %[[SPEC_OFF_C]], %c40
// CHECK: stream.binding.subspan %[[SPEC_A]][%[[SPEC_SUM_C]]]
// CHECK-LABEL: stream.executable private @specializeBindingsEx_0 {
// CHECK-NEXT: }
// CHECK-LABEL: util.func public @specializeBindings
// CHECK-SAME: (%[[SPEC_TAG:[^:,]+]]: index)
// CHECK: stream.cmd.execute
// CHECK-SAME: with(%{{.+}} as %[[SPEC_CAPTURE_A:[^:,]+]]: !stream.resource<transient>{%c200}, %{{.+}} as %[[SPEC_CAPTURE_B:[^:,]+]]: !stream.resource<transient>{%c200})
// CHECK: stream.cmd.dispatch @specializeBindingsEx::@dispatch
// CHECK-SAME: (%c40, %c0, %c80, %[[SPEC_TAG]] : index, index, index, index)
// CHECK-NEXT: ro %[[SPEC_CAPTURE_A]][%{{.+}} for %c200]{{.*}}{{$}}
// CHECK-NEXT: rw %[[SPEC_CAPTURE_B]][%{{.+}} for %c200]{{.*}}{{$}}
// CHECK-NEXT: }
// CHECK: stream.cmd.dispatch @specializeBindingsEx_1::@dispatch
// CHECK-SAME: (%c120, %c160, %c20, %[[SPEC_TAG]] : index, index, index, index)
// CHECK-NEXT: rw %[[SPEC_CAPTURE_A]][%{{.+}} for %c200]{{.*}}{{$}}
// CHECK-NEXT: rw %[[SPEC_CAPTURE_B]][%{{.+}} for %c200]{{.*}}{{$}}
// CHECK-NEXT: }
// CHECK: stream.cmd.dispatch @specializeBindingsEx_1::@dispatch
// CHECK-SAME: (%c0, %c40, %c80, %[[SPEC_TAG]] : index, index, index, index)
// CHECK-NEXT: rw %[[SPEC_CAPTURE_B]][%{{.+}} for %c200]{{.*}}{{$}}
// CHECK-NEXT: rw %[[SPEC_CAPTURE_A]][%{{.+}} for %c200]{{.*}}{{$}}
// CHECK-NEXT: }
// CHECK-LABEL: stream.executable private @specializeBindingsEx_1 {
// CHECK: util.func public @dispatch(%[[SPEC_CLONE_A:[^:,]+]]: !stream.binding, %[[SPEC_CLONE_B:[^:,]+]]: !stream.binding,
// CHECK-SAME: %[[SPEC_CLONE_OFF_A:[^:,]+]]: index, %[[SPEC_CLONE_OFF_B:[^:,]+]]: index, %[[SPEC_CLONE_OFF_C:[^:,]+]]: index, %{{[^:,]+}}: index)
// CHECK: stream.binding.subspan %[[SPEC_CLONE_A]][%[[SPEC_CLONE_OFF_A]]]
// CHECK: %[[SPEC_CLONE_SUM_B:.*]] = arith.addi %[[SPEC_CLONE_OFF_B]], %c20
// CHECK: stream.binding.subspan %[[SPEC_CLONE_B]][%[[SPEC_CLONE_SUM_B]]]
// CHECK: %[[SPEC_CLONE_SUM_C:.*]] = arith.addi %[[SPEC_CLONE_OFF_C]], %c40
// CHECK: stream.binding.subspan %[[SPEC_CLONE_B]][%[[SPEC_CLONE_SUM_C]]]
// CHECK-NOT: stream.executable private
#config = #stream.resource_config<{alias_mutable_bindings = true}>
stream.executable private @specializeBindingsEx {
  stream.executable.export public @dispatch attributes {stream.resources = #config}
  builtin.module {
    util.func public @dispatch(%a: !stream.binding, %b: !stream.binding, %c: !stream.binding, %tag: index) {
      %c0 = arith.constant 0 : index
      %c20 = arith.constant 20 : index
      %c40 = arith.constant 40 : index
      %span_a = stream.binding.subspan %a[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      util.optimization_barrier %span_a : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>
      %span_b = stream.binding.subspan %b[%c20] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      util.optimization_barrier %span_b : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>
      %span_c = stream.binding.subspan %c[%c40] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>{%c20}
      util.optimization_barrier %span_c : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<20xi8>>
      util.optimization_barrier %tag : index
      util.return
    }
  }
}
stream.executable private @specializeBindingsEx_0 {}
util.func public @specializeBindings(%tag: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  %c120 = arith.constant 120 : index
  %c160 = arith.constant 160 : index
  %c200 = arith.constant 200 : index
  %alloc0 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  %alloc1 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  %done = stream.cmd.execute with(%alloc0 as %capture0: !stream.resource<transient>{%c200},
      %alloc1 as %capture1: !stream.resource<transient>{%c200}) {
    stream.cmd.dispatch @specializeBindingsEx::@dispatch[%c1, %c1, %c1](%tag : index) {
      ro %capture0[%c40 for %c20] : !stream.resource<transient>{%c200},
      rw %capture1[%c80 for %c20] : !stream.resource<transient>{%c200},
      ro %capture0[%c0 for %c20] : !stream.resource<transient>{%c200}
    }
    stream.cmd.dispatch @specializeBindingsEx::@dispatch[%c1, %c1, %c1](%tag : index) {
      rw %capture0[%c120 for %c20] : !stream.resource<transient>{%c200},
      ro %capture1[%c160 for %c20] : !stream.resource<transient>{%c200},
      rw %capture1[%c20 for %c20] : !stream.resource<transient>{%c200}
    }
    stream.cmd.dispatch @specializeBindingsEx::@dispatch[%c1, %c1, %c1](%tag : index) {
      ro %capture1[%c0 for %c20] : !stream.resource<transient>{%c200},
      rw %capture0[%c40 for %c20] : !stream.resource<transient>{%c200},
      rw %capture0[%c80 for %c20] : !stream.resource<transient>{%c200}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Disabled mutable aliasing must not introduce specializations.
// CHECK-LABEL: stream.executable private @disabledAliasesEx {
// CHECK: util.func public @dispatch(%{{[^:,]+}}: !stream.binding, %{{[^:,]+}}: !stream.binding, %{{[^:,]+}}: !stream.binding,
// CHECK-NOT: stream.executable private
// CHECK-LABEL: util.func public @disabledAliases
// CHECK: stream.cmd.dispatch @disabledAliasesEx::@dispatch
// CHECK: stream.cmd.dispatch @disabledAliasesEx::@dispatch
// CHECK-NOT: stream.executable private
#config = #stream.resource_config<{alias_mutable_bindings = false}>
stream.executable private @disabledAliasesEx {
  stream.executable.export public @dispatch attributes {stream.resources = #config}
  builtin.module {
    util.func public @dispatch(%a: !stream.binding, %b: !stream.binding, %c: !stream.binding, %tag: index) {
      util.return
    }
  }
}
util.func public @disabledAliases(%tag: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  %c120 = arith.constant 120 : index
  %c160 = arith.constant 160 : index
  %c200 = arith.constant 200 : index
  %alloc0 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  %alloc1 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  %done = stream.cmd.execute with(%alloc0 as %capture0: !stream.resource<transient>{%c200},
      %alloc1 as %capture1: !stream.resource<transient>{%c200}) {
    stream.cmd.dispatch @disabledAliasesEx::@dispatch[%c1, %c1, %c1](%tag : index) {
      rw %capture0[%c40 for %c20] : !stream.resource<transient>{%c200},
      rw %capture1[%c80 for %c20] : !stream.resource<transient>{%c200},
      ro %capture0[%c0 for %c20] : !stream.resource<transient>{%c200}
    }
    stream.cmd.dispatch @disabledAliasesEx::@dispatch[%c1, %c1, %c1](%tag : index) {
      rw %capture0[%c120 for %c20] : !stream.resource<transient>{%c200},
      ro %capture1[%c160 for %c20] : !stream.resource<transient>{%c200},
      rw %capture1[%c20 for %c20] : !stream.resource<transient>{%c200}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Executables with multiple exports retain the existing fusion behavior.
// CHECK-LABEL: stream.executable private @multipleExportsEx {
// CHECK: stream.executable.export public @dispatch
// CHECK: stream.executable.export public @other
// CHECK: util.func public @dispatch(%{{[^:,]+}}: !stream.binding, %{{[^:,]+}}: !stream.binding, %{{[^:,]+}}: !stream.binding,
// CHECK-NOT: stream.executable private
// CHECK-LABEL: util.func public @multipleExports
// CHECK: stream.cmd.dispatch @multipleExportsEx::@dispatch
// CHECK: stream.cmd.dispatch @multipleExportsEx::@dispatch
// CHECK-NOT: stream.executable private
#config = #stream.resource_config<{alias_mutable_bindings = true}>
stream.executable private @multipleExportsEx {
  stream.executable.export public @dispatch attributes {stream.resources = #config}
  stream.executable.export public @other attributes {stream.resources = #config}
  builtin.module {
    util.func public @dispatch(%a: !stream.binding, %b: !stream.binding, %c: !stream.binding, %tag: index) {
      util.return
    }
    util.func public @other(%a: !stream.binding, %b: !stream.binding, %c: !stream.binding, %tag: index) {
      util.return
    }
  }
}
util.func public @multipleExports(%tag: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c80 = arith.constant 80 : index
  %c120 = arith.constant 120 : index
  %c160 = arith.constant 160 : index
  %c200 = arith.constant 200 : index
  %alloc0 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  %alloc1 = stream.resource.alloc uninitialized : !stream.resource<transient>{%c200}
  %done = stream.cmd.execute with(%alloc0 as %capture0: !stream.resource<transient>{%c200},
      %alloc1 as %capture1: !stream.resource<transient>{%c200}) {
    stream.cmd.dispatch @multipleExportsEx::@dispatch[%c1, %c1, %c1](%tag : index) {
      rw %capture0[%c40 for %c20] : !stream.resource<transient>{%c200},
      rw %capture1[%c80 for %c20] : !stream.resource<transient>{%c200},
      ro %capture0[%c0 for %c20] : !stream.resource<transient>{%c200}
    }
    stream.cmd.dispatch @multipleExportsEx::@dispatch[%c1, %c1, %c1](%tag : index) {
      rw %capture0[%c120 for %c20] : !stream.resource<transient>{%c200},
      ro %capture1[%c160 for %c20] : !stream.resource<transient>{%c200},
      rw %capture1[%c20 for %c20] : !stream.resource<transient>{%c200}
    }
  } => !stream.timepoint
  util.return
}
