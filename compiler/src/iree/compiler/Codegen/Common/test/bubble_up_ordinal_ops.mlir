// RUN: iree-opt --iree-codegen-bubble-up-ordinal-ops --allow-unregistered-dialect %s | FileCheck %s

// Test that uses of a cast op are replaced with uses of the ordinal op
// when the cast result has multiple uses (one being the ordinal op).
func.func @bubble_ordinal_through_index_castui(%arg0: i32) -> (index, index) {
  %0 = arith.index_castui %arg0 : i32 to index
  %1 = iree_tensor_ext.dispatch.workload.ordinal %0, 4 : index
  // Note: this uses %0 (cast result) directly, not %1 (ordinal result)
  %2 = "foo.op"(%0) : (index) -> (index)
  return %1, %2 : index, index
}
// CHECK-LABEL: func @bubble_ordinal_through_index_castui(
//  CHECK-SAME:     %[[ARG0:.+]]: i32)
//       CHECK:   %[[CAST:.+]] = arith.index_castui %[[ARG0]] : i32 to index
//       CHECK:   %[[ORDINAL:.+]] = iree_tensor_ext.dispatch.workload.ordinal %[[CAST]], 4
// CHECK:   %[[RESULT:.+]] = "foo.op"(%[[ORDINAL]])
// CHECK:   return %[[ORDINAL]], %[[RESULT]]
