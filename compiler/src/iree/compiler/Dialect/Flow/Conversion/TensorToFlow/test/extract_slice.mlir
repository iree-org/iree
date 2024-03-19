// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

 util.func public @extract_slice1(%arg0 : tensor<5x24x48xf32>) -> tensor<4xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 4] [1, 1, 4] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<4xf32>
  util.return %0 : tensor<4xf32>
}
// CHECK-LABEL:  util.func public @extract_slice1(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x24x48xf32>)
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C4]] for %[[C1]], %[[C1]], %[[C4]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   util.return %[[RESULT]]

// -----

 util.func public @extract_slice2(%arg0 : tensor<5x24x48xf32>) -> tensor<2x48xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 0] [1, 2, 48] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<2x48xf32>
  util.return %0 : tensor<2x48xf32>
}
// CHECK-LABEL:  util.func public @extract_slice2
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x24x48xf32>)
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C48:.+]] = arith.constant 48 : index
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C0]] for %[[C1]], %[[C2]], %[[C48]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   util.return %[[RESULT]]

// -----

 util.func public @extract_slice3(%arg0 : tensor<5x24x48xf32>) -> tensor<2x24xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 0] [1, 2, 24] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<2x24xf32>
  util.return %0 : tensor<2x24xf32>
}
// CHECK-LABEL:  util.func public @extract_slice3
//       CHECK:   tensor.extract_slice

// -----

 util.func public @extract_slice4(%arg0 : tensor<5x24x48xf32>, %arg1 : index) -> tensor<2x24xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 0] [1, 2, 24] [1, %arg1, 1]
      : tensor<5x24x48xf32> to tensor<2x24xf32>
  util.return %0 : tensor<2x24xf32>
}
// CHECK-LABEL:  util.func public @extract_slice4
//       CHECK:   tensor.extract_slice

// -----

 util.func public @extract_slice5(%arg0 : tensor<5x24x48xf32>, %arg1 : index) -> tensor<2x48xf32> {
  %0 = tensor.extract_slice %arg0[2, %arg1, 0] [1, 2, 48] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<2x48xf32>
  util.return %0 : tensor<2x48xf32>
}
// CHECK-LABEL:  util.func public @extract_slice5(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x24x48xf32>
//  CHECK-SAME:   %[[ARG1:.+]]: index)
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C48:.+]] = arith.constant 48 : index
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[ARG1]], %[[C0]] for %[[C1]], %[[C2]], %[[C48]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   util.return %[[RESULT]]

// -----

 util.func public @extract_slice6(%arg0 : tensor<5x24x48xf32>, %arg1 : index) -> tensor<?x48xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 0] [1, %arg1, 48] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<?x48xf32>
  util.return %0 : tensor<?x48xf32>
}
// CHECK-LABEL:  util.func public @extract_slice6(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x24x48xf32>
//  CHECK-SAME:   %[[ARG1:.+]]: index)
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C48:.+]] = arith.constant 48 : index
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C0]] for %[[C1]], %[[ARG1]], %[[C48]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   util.return %[[RESULT]]

// -----

 util.func public @extract_slice7(%arg0 : tensor<5x?x48xf32>, %arg1 : index) -> tensor<2x48xf32> {
  %0 = tensor.extract_slice %arg0[2, 3, 0] [1, 2, 48] [1, 1, 1]
      : tensor<5x?x48xf32> to tensor<2x48xf32>
  util.return %0 : tensor<2x48xf32>
}
// CHECK-LABEL:  util.func public @extract_slice7(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x?x48xf32>
//  CHECK-SAME:   %[[ARG1:.+]]: index)
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C48:.+]] = arith.constant 48 : index
//   CHECK-DAG:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<5x?x48xf32>
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C0]] for %[[C1]], %[[C2]], %[[C48]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   util.return %[[RESULT]]

// -----

 util.func public @rank_reducing_extract_slice(%arg0: tensor<?x513xi32>) -> tensor<513xi32> {
  %0 = tensor.extract_slice %arg0[4, 0] [1, 513] [1, 1] : tensor<?x513xi32> to tensor<513xi32>
  util.return %0 : tensor<513xi32>
}
// CHECK-LABEL:  util.func public @rank_reducing_extract_slice
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C513:.+]] = arith.constant 513 : index
//       CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]]
//  CHECK-SAME:       [%[[C4]], %[[C0]] for %[[C1]], %[[C513]]]
//  CHECK-SAME:       : tensor<?x513xi32>{%[[DIM]]} -> tensor<1x513xi32>
//       CHECK:   %[[RESHAPE:.+]] = flow.tensor.reshape %[[SLICE]] : tensor<1x513xi32> -> tensor<513xi32>
//       CHECK:   util.return %[[RESHAPE]] : tensor<513xi32>

// -----

 util.func public @rank_reducing_extract_slice_trailing_unit_dims
   (%arg0 : tensor<1x50x20x1xf32>) -> tensor<49x20xf32> {
  %0 = tensor.extract_slice %arg0[0, 1, 0, 0] [1, 49, 20, 1] [1, 1, 1, 1] : tensor<1x50x20x1xf32> to tensor<49x20xf32>
  util.return %0 : tensor<49x20xf32>
}
// CHECK-LABEL:  util.func public @rank_reducing_extract_slice_trailing_unit_dims
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C49:.+]] = arith.constant 49 : index
//   CHECK-DAG:   %[[C20:.+]] = arith.constant 20 : index
//       CHECK:   %[[extract_slice:.+]] = flow.tensor.slice %{{.+}}[%[[C0]], %[[C1]], %[[C0]], %[[C0]] for %[[C1]], %[[C49]], %[[C20]], %[[C1]]] : tensor<1x50x20x1xf32> -> tensor<1x49x20x1xf32>
//       CHECK:   flow.tensor.reshape %[[extract_slice]] : tensor<1x49x20x1xf32> -> tensor<49x20xf32>

// -----

 util.func public @extract_slice_within_dispatch_workgroups_not_converted() -> tensor<f32> {
  %x = arith.constant 100 : index
  %0 = flow.dispatch.workgroups[%x]() : () -> (tensor<f32>) = () {
    // CHECK: = tensor.extract_slice %[[source:.+]][2, 3, 4] [1, 1, 4] [1, 1, 1] : tensor<5x24x48xf32> to tensor<4xf32>
    %1 = "test.source"() : () -> (tensor<5x24x48xf32>)
    %2 = tensor.extract_slice %1[2, 3, 4] [1, 1, 4] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<4xf32>
    "test.sink"(%2) : (tensor<4xf32>) -> ()
    flow.return
  }
  util.return %0 : tensor<f32>
}
