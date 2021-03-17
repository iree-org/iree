// RUN: iree-opt -iree-flow-convert-to-flow-tensor-ops-pass -canonicalize -cse -split-input-file %s | IreeFileCheck %s

func @subtensor_convert(%arg0 : tensor<?x24x48xf32>, %arg1 : index) ->
    (tensor<?x12x12xf32>, tensor<?x12x24xf32>, tensor<?x?x48xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = subtensor %arg0[0, 0, 0] [%arg1, 24, 48] [1, 1, 1]
      : tensor<?x24x48xf32> to tensor<?x24x48xf32>
  %2 = memref.dim %arg0, %c0 : tensor<?x24x48xf32>
  %3 = subi %2, %arg1 : index
  %4 = subtensor %arg0[%arg1, 0, 0] [%3, 24, 48] [1, 1, 1]
      : tensor<?x24x48xf32> to tensor<?x24x48xf32>
  %5 = subtensor %0[0, 0, 0] [%arg1, 12, 12] [1, 1, 1]
      : tensor<?x24x48xf32> to tensor<?x12x12xf32>
  %6 = subtensor %4[0, 0, 0] [%3, 12, 24] [1, 2, 2]
      : tensor<?x24x48xf32> to tensor<?x12x24xf32>
  %7 = memref.dim %arg0, %c1 : tensor<?x24x48xf32>
  %8 = subi %7, %arg1 : index
  %9 = subtensor %arg0[0, %arg1, 0] [%2, %8, 48] [1, 1, 1]
      : tensor<?x24x48xf32> to tensor<?x?x48xf32>
  return %5, %6, %9 : tensor<?x12x12xf32>, tensor<?x12x24xf32>, tensor<?x?x48xf32>
}
// CHECK-LABEL: func @subtensor_convert
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x24x48xf32>
//  CHECK-SAME:   %[[ARG1:.+]]: index
//   CHECK-DAG:   %[[C24:.+]] = constant 24
//   CHECK-DAG:   %[[C48:.+]] = constant 48
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//       CHECK:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[SLICE1:.+]] = flow.tensor.slice %[[ARG0]]
//  CHECK-SAME:       [%[[C0]], %[[C0]], %[[C0]] for %[[ARG1]], %[[C24]], %[[C48]]]
//  CHECK-SAME:       : tensor<?x24x48xf32>{%[[D0]], %[[C24]], %[[C48]]}
//  CHECK-SAME:       -> tensor<?x24x48xf32>{%[[ARG1]], %[[C24]], %[[C48]]}
//       CHECK:   %[[D1:.+]] = subi %[[D0]], %[[ARG1]]
//       CHECK:   %[[SLICE2:.+]] = flow.tensor.slice %[[ARG0]]
//  CHECK-SAME:       [%[[ARG1]], %[[C0]], %[[C0]] for %[[D1]], %[[C24]], %[[C48]]]
//  CHECK-SAME:       : tensor<?x24x48xf32>{%[[D0]], %[[C24]], %[[C48]]}
//  CHECK-SAME:       -> tensor<?x24x48xf32>{%[[D1]], %[[C24]], %[[C48]]}
//   CHECK-DAG:   %[[UNMODIFIED1:.+]] = subtensor %[[SLICE1]][0, 0, 0] [%[[ARG1]], 12, 12] [1, 1, 1]
//   CHECK-DAG:   %[[UNMODIFIED2:.+]] = subtensor %[[SLICE2]][0, 0, 0] [%[[D1]], 12, 24] [1, 2, 2]
//   CHECK-DAG:   %[[UNMODIFIED3:.+]] = subtensor %[[ARG0]][0, %[[ARG1]], 0]
//       CHECK:   return %[[UNMODIFIED1]], %[[UNMODIFIED2]], %[[UNMODIFIED3]]
