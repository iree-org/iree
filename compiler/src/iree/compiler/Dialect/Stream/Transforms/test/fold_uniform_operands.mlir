// RUN: iree-opt --split-input-file --iree-stream-fold-uniform-operands %s | FileCheck %s

// Tests that the pass doesn't crash if there is no inner module.

// CHECK-LABEL: @noInnerModuleEx
stream.executable private @noInnerModuleEx {
  // CHECK: stream.executable.export public @dispatch workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    // CHECK: iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG1]])
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
    stream.return %x, %y, %z : index, index, index
  }
}
// CHECK: util.func public @no_inner_module(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
util.func public @no_inner_module(%arg0: !hal.buffer_view, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c20}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c20}) {
    // CHECK: stream.cmd.dispatch @noInnerModuleEx::@dispatch
    // CHECK-SAME: [%[[ARG1]], %[[ARG1]]]
    // CHECK-SAME: (%[[ARG1]], %[[ARG2]], %[[ARG2]] : index, index, index)
    stream.cmd.dispatch @noInnerModuleEx::@dispatch[%arg1, %arg1](%arg1, %arg2, %arg2 : index, index, index) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Tests that duplicate primitive operands are folded if they are uniform at all
// dispatch sites.
//
// In this test (%a, %a) -> (%a0, %a1) at both dispatch sites and can be
// folded, while (%b, %b) -> (%b0, %b1) and (%c20, %b) -> (%b0, %b1) prevents
// the folding due to divergent values. The binding and %c should be untouched.

// CHECK-LABEL: @deduplicateOperandsEx
stream.executable private @deduplicateOperandsEx {
  stream.executable.export public @dispatch
  builtin.module  {
    // CHECK:  util.func public @dispatch(%[[BINDING:.+]]: !stream.binding, %[[A01:.+]]: i32, %[[B0:.+]]: index, %[[C:.+]]: i1, %[[B1:.+]]: index)
     util.func public @dispatch(%binding: !stream.binding, %a0: i32, %b0: index, %c: i1, %a1: i32, %b1: index) {
      // CHECK-NEXT: util.optimization_barrier %[[BINDING]] : !stream.binding
      util.optimization_barrier %binding : !stream.binding
      // CHECK-NEXT: util.optimization_barrier %[[A01]] : i32
      util.optimization_barrier %a0 : i32
      // CHECK-NEXT: util.optimization_barrier %[[A01]] : i32
      util.optimization_barrier %a1 : i32
      // CHECK-NEXT: util.optimization_barrier %[[B0]] : index
      util.optimization_barrier %b0 : index
      // CHECK-NEXT: util.optimization_barrier %[[B1]] : index
      util.optimization_barrier %b1 : index
      // CHECK-NEXT: util.optimization_barrier %[[C]] : i1
      util.optimization_barrier %c : i1
      util.return
    }
  }
}
// CHECK:  util.func public @deduplicateOperands(%[[A:.+]]: i32, %[[B:.+]]: index, %[[C:.+]]: i1)
util.func public @deduplicateOperands(%a: i32, %b: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c20}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c20}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[A]], %[[B]], %[[C]], %[[B]] : i32, index, i1, index)
    stream.cmd.dispatch @deduplicateOperandsEx::@dispatch[%c1, %c1, %c1](%a, %b, %c, %a, %b : i32, index, i1, i32, index) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
    // CHECK: stream.cmd.dispatch {{.+}}(%[[A]], %c20, %[[C]], %[[B]] : i32, index, i1, index)
    stream.cmd.dispatch @deduplicateOperandsEx::@dispatch[%c1, %c1, %c1](%a, %c20, %c, %a, %b : i32, index, i1, i32, index) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Tests that operands that are uniformly constant at all dispatch sites are
// inlined into the dispatch regions.
//
// In this test %a is dynamic and ignored, %b is uniformly %c20 and inlined, and
// %c is divergent (%false/%true) and skipped.

// CHECK-LABEL: @inlineConstantOperandsEx
stream.executable private @inlineConstantOperandsEx {
  stream.executable.export public @dispatch
  builtin.module  {
    // CHECK:  util.func public @dispatch(%[[BINDING:.+]]: !stream.binding, %[[A:.+]]: i32, %[[C:.+]]: i1)
     util.func public @dispatch(%binding: !stream.binding, %a: i32, %b: index, %c: i1) {
      // CHECK: %[[B:.+]] = arith.constant 20 : index
      // CHECK-NEXT: util.optimization_barrier %[[BINDING]] : !stream.binding
      util.optimization_barrier %binding : !stream.binding
      // CHECK-NEXT: util.optimization_barrier %[[A]] : i32
      util.optimization_barrier %a : i32
      // CHECK-NEXT: util.optimization_barrier %[[B]] : index
      util.optimization_barrier %b : index
      // CHECK-NEXT: util.optimization_barrier %[[C]] : i1
      util.optimization_barrier %c : i1
      util.return
    }
  }
}
// CHECK:  util.func public @inlineConstantOperands(%[[A:.+]]: i32)
util.func public @inlineConstantOperands(%a: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c20}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c20}) {
    // CHECK: stream.cmd.dispatch {{.+}}(%[[A]], %false : i32, i1)
    stream.cmd.dispatch @inlineConstantOperandsEx::@dispatch[%c1, %c1, %c1](%a, %c20, %false : i32, index, i1) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
    // CHECK: stream.cmd.dispatch {{.+}}(%[[A]], %true : i32, i1)
    stream.cmd.dispatch @inlineConstantOperandsEx::@dispatch[%c1, %c1, %c1](%a, %c20, %true : i32, index, i1) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
  } => !stream.timepoint
  util.return
}

// -----

// Tests that duplicate workloads are folded if they are uniform at all dispatch sites.

// CHECK-LABEL: @deduplicateWorkloadsEx
stream.executable private @deduplicateWorkloadsEx {
  // CHECK: stream.executable.export public @dispatch workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index) -> (index, index, index) {
    // CHECK: iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG0]], %[[ARG1]], %[[ARG0]], %[[ARG2]])
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4)
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module  {
    // CHECK: util.func public @dispatch(%[[ARG0:.+]]: !stream.binding, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
    util.func public @dispatch(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: index) {
      // CHECK-NEXT: util.optimization_barrier %[[ARG0]] : !stream.binding
      util.optimization_barrier %arg0 : !stream.binding
      // CHECK-NEXT: %[[ASSUME:.+]]:5 = util.assume.int
      %0:5 = util.assume.int
          %arg1<umin = 4, umax = 524160, udiv = 4>,
          %arg1<umin = 4, umax = 524160, udiv = 4>,
          %arg2<umin = 4, umax = 524160, udiv = 4>,
          %arg1<umin = 4, umax = 524160, udiv = 4>,
          %arg3<umin = 4, umax = 524160, udiv = 4>
        : index, index, index, index, index
      // CHECK: %[[ORDINAL0:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[ASSUME]]#0, 0 : index
      %1 = iree_tensor_ext.dispatch.workload.ordinal %0#0, 0 : index
      // CHECK-NEXT: util.optimization_barrier %[[ORDINAL0]] : index
      util.optimization_barrier %1 : index
      // CHECK-NEXT: %[[ORDINAL1:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[ASSUME]]#1, 0 : index
      %2 = iree_tensor_ext.dispatch.workload.ordinal %0#1, 1 : index
      // CHECK-NEXT: util.optimization_barrier %[[ORDINAL1]] : index
      util.optimization_barrier %2 : index
      // CHECK-NEXT: %[[ORDINAL2:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[ASSUME]]#2, 1 : index
      %3 = iree_tensor_ext.dispatch.workload.ordinal %0#2, 2 : index
      // CHECK-NEXT: util.optimization_barrier %[[ORDINAL2]] : index
      util.optimization_barrier %3 : index
      // CHECK-NEXT: %[[ORDINAL3:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[ASSUME]]#3, 0 : index
      %4 = iree_tensor_ext.dispatch.workload.ordinal %0#3, 3 : index
      // CHECK-NEXT: util.optimization_barrier %[[ORDINAL3]] : index
      util.optimization_barrier %4 : index
      // CHECK-NEXT: %[[ORDINAL4:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[ASSUME]]#4, 2 : index
      %5 = iree_tensor_ext.dispatch.workload.ordinal %0#4, 4 : index
      // CHECK-NEXT: util.optimization_barrier %[[ORDINAL4]] : index
      util.optimization_barrier %5 : index
      util.return
    }
  }
}
// CHECK: util.func public @deduplicateWorkloads(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
util.func public @deduplicateWorkloads(%arg0: !hal.buffer_view, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %alloc = stream.resource.alloc uninitialized : !stream.resource<transient>{%c20}
  %result_timepoint = stream.cmd.execute with(%alloc as %capture: !stream.resource<transient>{%c20}) {
    // CHECK: stream.cmd.dispatch @deduplicateWorkloadsEx::@dispatch
    // CHECK-SAME: [%[[ARG1]], %[[ARG1]], %[[ARG3]]]
    // CHECK-SAME: (%[[ARG1]], %[[ARG2]], %[[ARG3]] : index, index, index)
    stream.cmd.dispatch @deduplicateWorkloadsEx::@dispatch[%arg1, %arg1, %arg1, %arg1, %arg3](%arg1, %arg2, %arg3 : index, index, index) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
    // CHECK: stream.cmd.dispatch @deduplicateWorkloadsEx::@dispatch
    // CHECK-SAME: [%[[ARG1]], %[[ARG2]], %[[ARG3]]]
    // CHECK-SAME: (%[[ARG1]], %[[ARG2]], %[[ARG3]] : index, index, index)
    stream.cmd.dispatch @deduplicateWorkloadsEx::@dispatch[%arg1, %arg1, %arg2, %arg1, %arg3](%arg1, %arg2, %arg3 : index, index, index) {
      rw %capture[%c0 for %c20] : !stream.resource<transient>{%c20}
    }
  } => !stream.timepoint
  util.return
}
