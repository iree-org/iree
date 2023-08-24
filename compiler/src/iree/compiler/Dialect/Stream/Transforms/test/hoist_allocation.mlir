// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-stream-hoist-allocation))' %s | FileCheck %s

// Test that resource allocatiion housted into the loop predecessor block.

// CHECK-LABEL: @hoistResourceAlloc
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<external>)
func.func @hoistResourceAlloc(%arg0: !stream.resource<external>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<staging>
  // CHECK-NEXT: cf.br ^bb1
  cf.br ^bb1
^bb1:
 // CHECK: stream.cmd.execute
 // CHECK-SAME: %[[OPERAND]] as %[[ARG0:.+]]: !stream.resource<external>
 // CHECK-SAME: %[[ALLOC]] as %[[ARG1:.+]]: !stream.resource<staging>
 // CHECK: stream.cmd.copy %[[ARG0]][%[[C0]]], %[[ARG1]][%[[C0]]]
 // CHECK: stream.cmd.flush %[[ARG1]]
  %0 = stream.resource.alloc uninitialized : !stream.resource<staging>{%c1}
  %1 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<external>{%c1}, %0 as %arg3: !stream.resource<staging>{%c1}) {
    stream.cmd.copy %arg2[%c0], %arg3[%c0], %c1 : !stream.resource<external>{%c1} -> !stream.resource<staging>{%c1}
    stream.cmd.flush %arg3[%c0 for %c1] : !stream.resource<staging>{%c1}
  } => !stream.timepoint
  %2 = stream.timepoint.await %1 => %0 : !stream.resource<staging>{%c1}
  %3 = stream.resource.load %2[%c0] : !stream.resource<staging>{%c1} -> i1
  cf.cond_br %3, ^bb1, ^bb2
^bb2:
  return
}
