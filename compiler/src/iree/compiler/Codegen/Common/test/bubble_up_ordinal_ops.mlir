// RUN: iree-opt --iree-codegen-bubble-up-ordinal-ops --allow-unregistered-dialect %s | FileCheck %s

func.func @castui(%arg0: i32) -> index {
  %0 = arith.index_castui %arg0 : i32 to index
  %1 = flow.dispatch.workload.ordinal %0 4 : index
  %2 = "foo.op"(%1) : (index) -> (index)
  return %2 : index
}
// CHECK-LABEL: func @castui(
//  CHECK-SAME:     %[[ARG0:.+]]: i32)
//       CHECK:   %[[CAST:.+]] = arith.index_castui %[[ARG0]] : i32 to index
//       CHECK:   %[[ORDINAL:.+]] = flow.dispatch.workload.ordinal %[[CAST]] 4
//       CHECK:   "foo.op"(%[[ORDINAL]])
