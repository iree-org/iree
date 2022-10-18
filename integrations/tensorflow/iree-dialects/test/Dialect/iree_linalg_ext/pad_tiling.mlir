// RUN: iree-dialects-opt --iree-linalg-ext-tile --split-input-file %s | FileCheck  %s
// XFAIL: *
// TODO: Re-enable when upstream tensor.pad op properly implements the tiling
// interface.

func.func @pad_tensor(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index,
    %arg3 : index, %arg4 : index, %arg5 : f32) -> tensor<?x?xf32> {
  %0 = tensor.pad %arg0 low[%arg1, %arg2] high[%arg3, %arg4] {
    ^bb0(%arg6 : index, %arg7 : index):
      tensor.yield %arg5 : f32
  } {__internal_iree_linalg_transform__ = "tiling_input"}
      :  tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2] -> (s2 + s0 + s1)>
//      CHECK: func.func @pad_tensor
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
// CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: f32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C10:.+]] = arith.constant 10 : index
//  CHECK-DAG:   %[[C20:.+]] = arith.constant 20 : index
//  CHECK-DAG:   %[[INIT:.+]] = tensor.empty()
//      CHECK:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:   %[[UBY:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[ARG3]], %[[D0]]]
//      CHECK:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[UBX:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]], %[[ARG4]], %[[D1]]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[UBY]] step %[[C10]]
// CHECK-SAME:       iter_args(%[[ARG7:.+]] = %[[INIT]])
//      CHECK:     %[[YIELD:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[UBX]] step %[[C20]]
// CHECK-SAME:         iter_args(%[[ARG9:.+]] = %[[ARG7]])
//      CHECK:       %[[PAD_TILE:.+]] = scf.if
//      CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[PAD_TILE]] into %[[ARG9]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]]
//      CHECK:       scf.yield %[[INSERT]]
//      CHECK:     scf.yield %[[YIELD]]
//      CHECK:   return %[[RESULT]]
