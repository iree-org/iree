// RUN: iree-opt -split-input-file -iree-flow-convert-to-flow-tensor-ops-pass %s | IreeFileCheck %s

func @insert_slice_convert
    (%arg0 : tensor<?x24x48xf32>, %arg1 : tensor<1x4x48xf32>) ->
    tensor<?x24x48xf32> {
  %c0 = constant 0 : index
  %0 = tensor.insert_slice %arg1 into %arg0[4, 2, 0] [1, 4, 48] [1, 1, 1] :
      tensor<1x4x48xf32> into tensor<?x24x48xf32>
  return %0 : tensor<?x24x48xf32>
}
// CHECK-LABEL: func @insert_slice_convert
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.update %[[ARG1]], %[[ARG0]][%[[C4]], %[[C2]], %[[C0]]]
//  CHECK-SAME:     : tensor<1x4x48xf32> -> tensor<?x24x48xf32>{%[[DIM0]]}

// -----

func @insert_slice_convert_rank_reducing
    (%arg0 : tensor<?x24x48xf32>, %arg1 : tensor<4x48xf32>) ->
    tensor<?x24x48xf32> {
  %c0 = constant 0 : index
  %0 = tensor.insert_slice %arg1 into %arg0[4, 2, 0] [1, 4, 48] [1, 1, 1] :
      tensor<4x48xf32> into tensor<?x24x48xf32>
  return %0 : tensor<?x24x48xf32>
}
// CHECK-LABEL: func @insert_slice_convert_rank_reducing
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[RESHAPE:.+]] = flow.tensor.reshape %[[ARG1]] : tensor<4x48xf32> -> tensor<1x4x48xf32>
//   CHECK-DAG:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.update %[[RESHAPE]], %[[ARG0]][%[[C4]], %[[C2]], %[[C0]]]
//  CHECK-SAME:     : tensor<1x4x48xf32> -> tensor<?x24x48xf32>{%[[DIM]]}

// -----

func @rank_reducing_insert_slice_trailing_unit_dims
   (%arg0 : tensor<49x20xf32>, %arg1 : tensor<1x50x20x1xf32>) -> tensor<1x50x20x1xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, 1, 0, 0] [1, 49, 20, 1] [1, 1, 1, 1] : tensor<49x20xf32> into tensor<1x50x20x1xf32>
  return %0 : tensor<1x50x20x1xf32>
}
// CHECK-LABEL: func @rank_reducing_insert_slice_trailing_unit_dims
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//       CHECK:   %[[RESHAPE:.+]] = flow.tensor.reshape %{{.+}} : tensor<49x20xf32> -> tensor<1x49x20x1xf32>
//       CHECK:   flow.tensor.update %[[RESHAPE]], %{{.+}}[%[[C0]], %[[C1]], %[[C0]], %[[C0]]] : tensor<1x49x20x1xf32> -> tensor<1x50x20x1xf32>
