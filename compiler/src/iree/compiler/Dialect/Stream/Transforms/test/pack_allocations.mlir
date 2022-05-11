// RUN: iree-opt --split-input-file --pass-pipeline='func.func(iree-stream-pack-allocations)' %s | FileCheck %s

// CHECK-LABEL: @packAllocations
// CHECK-SAME: (%[[SIZE_A:.+]]: index, %[[SIZE_B:.+]]: index)
func.func @packAllocations(%size_a: index, %size_b: index) {
  //      CHECK: %[[SLICES:.+]]:3 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[SIZE_A]],
  // CHECK-NEXT:   [0, 0] = %[[SIZE_B]]
  // CHECK-NEXT: }) : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SLICES]]#0}
  %0:2 = stream.resource.alloc uninitialized :
      !stream.resource<transient>{%size_a},
      !stream.resource<transient>{%size_b}

  // CHECK: %[[SLICE_A:.+]] = stream.resource.subview %[[ALLOC]][%[[SLICES]]#1]
  // CHECK-SAME: !stream.resource<transient>{%[[SLICES]]#0} -> !stream.resource<transient>{%[[SIZE_A]]}
  // CHECK: %[[SLICE_B:.+]] = stream.resource.subview %[[ALLOC]][%[[SLICES]]#2]
  // CHECK-SAME: !stream.resource<transient>{%[[SLICES]]#0} -> !stream.resource<transient>{%[[SIZE_B]]}

  // CHECK: util.do_not_optimize(%[[SLICE_A]])
  util.do_not_optimize(%0#0) : !stream.resource<transient>
  // CHECK: util.do_not_optimize(%[[SLICE_B]])
  util.do_not_optimize(%0#1) : !stream.resource<transient>
  return
}

// -----

// CHECK-LABEL: @packEmpty
func.func @packEmpty() {
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc : !stream.resource<transient>{%c0}
  %c0 = arith.constant 0 : index
  %0 = stream.resource.alloc : !stream.resource<transient>{%c0}

  // CHECK: util.do_not_optimize(%[[ALLOC]])
  util.do_not_optimize(%0) : !stream.resource<transient>
  return
}
