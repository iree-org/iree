// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -canonicalize -cse --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @fuse_collapse_shape_into_forall(%arg0: tensor<8x8xf32>) -> tensor<64xf32> {
    %0 = tensor.empty() : tensor<8x8xf32>
    %1 = scf.forall (%arg1) in (4) shared_outs(%arg2 = %0) -> (tensor<8x8xf32>) {
      %2 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg0[%2, 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
      %extracted_slice_0 = tensor.extract_slice %arg2[%2, 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
      %3 = linalg.copy ins(%extracted_slice : tensor<2x8xf32>) outs(%extracted_slice_0 : tensor<2x8xf32>) -> tensor<2x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg2[%2, 0] [2, 8] [1, 1] : tensor<2x8xf32> into tensor<8x8xf32>
      }
    } {mapping = [#gpu.thread<x>]}
    %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<8x8xf32> into tensor<64xf32>
    return %collapsed : tensor<64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %producer = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer = transform.structured.match ops{["tensor.collapse_shape"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.iree.fuse_collapse_shape_into_forall %consumer into %producer
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_collapse_shape_into_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<8x8xf32>

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<8x8xf32>
//       CHECK:   %[[COLLAPSED_OUT:.+]] = tensor.collapse_shape %[[EMPTY]] {{\[}}[0, 1]] : tensor<8x8xf32> into tensor<64xf32>
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall (%[[IDX:.+]]) in (4) shared_outs(%[[COLLAPSED_BBARG:.+]] = %[[COLLAPSED_OUT]]) -> (tensor<64xf32>) {
//   CHECK-DAG:     %[[EXPANDED_BBARG:.+]] = tensor.expand_shape %[[COLLAPSED_BBARG]]
//  CHECK-SAME:       {{\[}}[0, 1]] output_shape [8, 8] : tensor<64xf32> into tensor<8x8xf32>
//   CHECK-DAG:     %[[SLICE_IDX_0:.+]] = affine.apply #[[$MAP]](%[[IDX]])
//   CHECK-DAG:     %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[SLICE_IDX_0]], 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
//   CHECK-DAG:     %[[OUT_SLICE:.+]] = tensor.extract_slice %[[EXPANDED_BBARG]][%[[SLICE_IDX_0]], 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
//   CHECK-DAG:     %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<2x8xf32>) outs(%[[OUT_SLICE]] : tensor<2x8xf32>) -> tensor<2x8xf32>
//   CHECK-DAG:     %[[LINEAR_SLICE_IDX:.+]] = affine.linearize_index disjoint [%[[SLICE_IDX_0]], %[[C0]]] by (8, 8) : index
//   CHECK-DAG:     %[[COLLAPSED_COPY:.+]] = tensor.collapse_shape %[[COPY]] {{\[}}[0, 1]] : tensor<2x8xf32> into tensor<16xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[COLLAPSED_COPY]] into %[[COLLAPSED_BBARG]][%[[LINEAR_SLICE_IDX]]] [16] [1] : tensor<16xf32> into tensor<64xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   return %[[FORALL_RESULT]]

// -----

#map = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
module {
  func.func @fuse_dynamic_collapse_shape_into_forall(%arg0: tensor<?x?x8xf32>, %arg1: index, %arg2: index) -> tensor<?x?xf32> {
    %0 = tensor.empty(%arg1, %arg2) : tensor<?x?x8xf32>
    %1 = scf.forall (%arg3, %arg4) = (0, 0) to (%arg1, %arg2) step (4, 4) shared_outs(%arg5 = %0) -> (tensor<?x?x8xf32>) {
      %2 = affine.min #map(%arg3)[%arg1]
      %3 = affine.min #map(%arg4)[%arg2]
      %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg4, 0] [%2, %3, 8] [1, 1, 1] : tensor<?x?x8xf32> to tensor<?x?x8xf32>
      %extracted_slice_0 = tensor.extract_slice %arg5[%arg3, %arg4, 0] [%2, %3, 8] [1, 1, 1] : tensor<?x?x8xf32> to tensor<?x?x8xf32>
      %4 = linalg.copy ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_0 : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg5[%arg3, %arg4, 0] [%2, %3, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<?x?x8xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %collapsed = tensor.collapse_shape %1 [[0], [1, 2]] : tensor<?x?x8xf32> into tensor<?x?xf32>
    return %collapsed : tensor<?x?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.collapse_shape"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.iree.fuse_collapse_shape_into_forall %1 into %0 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 8)>

// CHECK-LABEL: func @fuse_dynamic_collapse_shape_into_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?x?x8xf32>
//  CHECK-SAME:   %[[SIZE0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[SIZE1:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[SIZE0]], %[[SIZE1]]) : tensor<?x?x8xf32>
//       CHECK:   %[[COLLAPSED_OUT:.+]] = tensor.collapse_shape %[[EMPTY]] {{\[}}[0], [1, 2]] : tensor<?x?x8xf32> into tensor<?x?xf32>
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall (%[[IDX0:.+]], %[[IDX1:.+]]) = (0, 0) to (%[[SIZE0]], %[[SIZE1]]) step (4, 4)
//  CHECK-SAME:     shared_outs(%[[COLLAPSED_BBARG:.+]] = %[[COLLAPSED_OUT]]) -> (tensor<?x?xf32>) {
//   CHECK-DAG:   %[[EXPANDED_BBARG:.+]] = tensor.expand_shape %[[COLLAPSED_BBARG]]
//  CHECK-SAME:     {{\[}}[0], [1, 2]] output_shape [%[[SIZE0]], %[[SIZE1]], 8] : tensor<?x?xf32> into tensor<?x?x8xf32>
//   CHECK-DAG:   %[[SLICE_SIZE_0:.+]] = affine.min #map(%[[IDX0]])[%[[SIZE0]]]
//   CHECK-DAG:   %[[SLICE_SIZE_1:.+]] = affine.min #map(%[[IDX1]])[%[[SIZE1]]]
//   CHECK-DAG:   %[[LINEAR_SLICE_IDX:.+]] = affine.linearize_index disjoint [%[[IDX1]], %[[C0]]] by (%[[SIZE1]], 8) : index
//   CHECK-DAG:   %[[COLLAPSED_SLICE_SIZE:.+]] = affine.apply #[[$MAP1]](%[[SLICE_SIZE_1]])
//   CHECK-DAG:   %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [%[[IDX0]], %[[IDX1]], 0]{{.*}}[%[[SLICE_SIZE_0]], %[[SLICE_SIZE_1]], 8] [1, 1, 1] : tensor<?x?x8xf32> to tensor<?x?x8xf32>
//   CHECK-DAG:   %[[OUT_SLICE:.+]] = tensor.extract_slice %[[EXPANDED_BBARG]]
//  CHECK-SAME:     [%[[IDX0]], %[[IDX1]], 0] [%[[SLICE_SIZE_0]], %[[SLICE_SIZE_1]], 8] [1, 1, 1] : tensor<?x?x8xf32> to tensor<?x?x8xf32>
//   CHECK-DAG:   %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<?x?x8xf32>) outs(%[[OUT_SLICE]] : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
//   CHECK-DAG:   %[[COLLAPSED_COPY:.+]] = tensor.collapse_shape %[[COPY]] {{\[}}[0], [1, 2]] : tensor<?x?x8xf32> into tensor<?x?xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[COLLAPSED_COPY]] into %[[COLLAPSED_BBARG]]
//  CHECK-SAME:         [%[[IDX0]], %[[LINEAR_SLICE_IDX]]] [%[[SLICE_SIZE_0]], %[[COLLAPSED_SLICE_SIZE]]] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
//       CHECK:   return %[[FORALL_RESULT]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @fuse_collapse_shape_into_multi_result_forall(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf16>) -> (tensor<64xf32>, tensor<8x8xf16>) {
    %0 = tensor.empty() : tensor<8x8xf32>
    %1 = tensor.empty() : tensor<8x8xf16>
    %2:2 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %0, %arg4 = %1) -> (tensor<8x8xf32>, tensor<8x8xf16>) {
      %3 = affine.apply #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[%3, 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
      %extracted_slice_0 = tensor.extract_slice %arg3[%3, 0] [2, 8] [1, 1] : tensor<8x8xf32> to tensor<2x8xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%3, 0] [2, 8] [1, 1] : tensor<8x8xf16> to tensor<2x8xf16>
      %extracted_slice_2 = tensor.extract_slice %arg4[%3, 0] [2, 8] [1, 1] : tensor<8x8xf16> to tensor<2x8xf16>
      %4 = linalg.copy ins(%extracted_slice : tensor<2x8xf32>) outs(%extracted_slice_0 : tensor<2x8xf32>) -> tensor<2x8xf32>
      %5 = linalg.copy ins(%extracted_slice_1 : tensor<2x8xf16>) outs(%extracted_slice_2 : tensor<2x8xf16>) -> tensor<2x8xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg3[%3, 0] [2, 8] [1, 1] : tensor<2x8xf32> into tensor<8x8xf32>
        tensor.parallel_insert_slice %5 into %arg4[%3, 0] [2, 8] [1, 1] : tensor<2x8xf16> into tensor<8x8xf16>
      }
    } {mapping = [#gpu.thread<x>]}
    %collapsed = tensor.collapse_shape %2#0 [[0, 1]] : tensor<8x8xf32> into tensor<64xf32>
    return %collapsed, %2#1 : tensor<64xf32>, tensor<8x8xf16>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.collapse_shape"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.iree.fuse_collapse_shape_into_forall %1 into %0 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @fuse_collapse_shape_into_multi_result_forall
//       CHECK:   %[[FORALL_RESULT:.+]]:2 = scf.forall {{.*}} -> (tensor<64xf32>, tensor<8x8xf16>) {
//       CHECK:     scf.forall.in_parallel {
//   CHECK-DAG:       tensor.parallel_insert_slice {{.*}} : tensor<16xf32> into tensor<64xf32>
//   CHECK-DAG:       tensor.parallel_insert_slice {{.*}} : tensor<2x8xf16> into tensor<8x8xf16>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   return %[[FORALL_RESULT]]#0, %[[FORALL_RESULT]]#1
