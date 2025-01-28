// RUN: iree-opt %s -iree-transform-dialect-interpreter --verify-diagnostics -transform-dialect-drop-schedule -canonicalize -cse --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @fuse_extract_slice_into_forall(%arg0: tensor<8xf32>, %arg1: index) -> tensor<?xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %0) -> (tensor<8xf32>) {
      %2 = affine.apply #map(%arg2)
      %extracted_slice_0 = tensor.extract_slice %arg0[%2] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%2] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %3 = linalg.copy ins(%extracted_slice_0 : tensor<2xf32>) outs(%extracted_slice_1 : tensor<2xf32>) -> tensor<2xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg3[%2] [2] [1] : tensor<2xf32> into tensor<8xf32>
      }
    } {mapping = [#gpu.thread<x>]}
    %extracted_slice = tensor.extract_slice %1[0] [%arg1] [1] : tensor<8xf32> to tensor<?xf32>
    return %extracted_slice : tensor<?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %producer = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer = transform.get_consumers_of_result %producer[0] : (!transform.any_op) -> !transform.any_op
    %2 = transform.iree.fuse_extract_slice_into_forall %consumer into %producer
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (d0 * -2 + s0, 0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (2, d0)>

// CHECK-LABEL: func @fuse_extract_slice_into_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
//   CHECK-DAG:   %[[SLICED_OUT:.+]] = tensor.extract_slice %[[EMPTY]][0] [%[[ARG1]]] [1] : tensor<8xf32> to tensor<?xf32>
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall (%[[IDX:.+]]) in (4) shared_outs(%[[SLICED_BBARG:.+]] = %[[SLICED_OUT]]) -> (tensor<?xf32>) {

//   CHECK-DAG:     %[[SLICE_IDX:.+]] = affine.apply #[[$MAP]](%[[IDX]])
//   CHECK-DAG:     %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[SLICE_IDX]]] [2] [1] : tensor<8xf32> to tensor<2xf32>
//   CHECK-DAG:     %[[OUT_SLICE:.+]] = tensor.extract_slice %[[EMPTY]][%[[SLICE_IDX]]] [2] [1] : tensor<8xf32> to tensor<2xf32>
//   CHECK-DAG:     %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<2xf32>) outs(%[[OUT_SLICE]] : tensor<2xf32>) -> tensor<2xf32>

//   CHECK-DAG:     %[[SIZE_CLAMPED_LOW:.+]] = affine.max #[[$MAP1]](%[[IDX]])[%[[ARG1]]]
//   CHECK-DAG:     %[[SIZE_CLAMPED_HIGH:.+]] = affine.min #[[$MAP2]](%[[SIZE_CLAMPED_LOW]])

//       CHECK:     %[[SLICED_COPY:.+]] = tensor.extract_slice %[[COPY]][0] [%[[SIZE_CLAMPED_HIGH]]] [1] : tensor<2xf32> to tensor<?xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[SLICED_COPY]] into %[[SLICED_BBARG]][%[[SLICE_IDX]]] [%[[SIZE_CLAMPED_HIGH]]] [1] : tensor<?xf32> into tensor<?xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   return %[[FORALL_RESULT]]

// -----

module {
  module {
    func.func @fuse_dynamic_extract_slice_into_forall(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<?x?xf32> {
      %0 = tensor.empty(%arg1, %arg2) : tensor<?x?xf32>
      %1 = scf.forall (%arg7, %arg8) = (0, 0) to (%arg1, %arg2) step (%arg5, %arg6) shared_outs(%arg9 = %0) -> (tensor<?x?xf32>) {
        %extracted_slice_0 = tensor.extract_slice %arg0[%arg7, %arg8] [%arg5, %arg6] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_1 = tensor.extract_slice %arg9[%arg7, %arg8] [%arg5, %arg6] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %2 = linalg.copy ins(%extracted_slice_0 : tensor<?x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %2 into %arg9[%arg7, %arg8] [%arg5, %arg6] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %extracted_slice = tensor.extract_slice %1[0, 0] [%arg3, %arg4] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      return %extracted_slice : tensor<?x?xf32>
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_consumers_of_result %0[0] : (!transform.any_op) -> !transform.any_op
      %2 = transform.iree.fuse_extract_slice_into_forall %1 into %0 : (!transform.any_op, !transform.any_op) -> !transform.any_op
      transform.yield
    }
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 0)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (s0, d0)>

// CHECK-LABEL: func @fuse_dynamic_extract_slice_into_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[SIZE0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[SIZE1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[EXTRACT_SIZE0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[EXTRACT_SIZE1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[STEP0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[STEP1:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[SIZE0]], %[[SIZE1]]) : tensor<?x?xf32>
//   CHECK-DAG:   %[[SLICED_OUT:.+]] = tensor.extract_slice %[[EMPTY]][0, 0] [%[[EXTRACT_SIZE0]], %[[EXTRACT_SIZE1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall (%[[IDX0:.+]], %[[IDX1:.+]]) = (0, 0) to (%[[SIZE0]], %[[SIZE1]]) step (%[[STEP0]], %[[STEP1]])
//  CHECK-SAME:       shared_outs(%[[SLICED_BBARG:.+]] = %[[SLICED_OUT]]) -> (tensor<?x?xf32>) {

//   CHECK-DAG:     %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]]{{.*}}[%[[IDX0]], %[[IDX1]]] [%[[STEP0]], %[[STEP1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//   CHECK-DAG:     %[[OUT_SLICE:.+]] = tensor.extract_slice %[[EMPTY]]{{.*}}[%[[IDX0]], %[[IDX1]]] [%[[STEP0]], %[[STEP1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//   CHECK-DAG:     %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<?x?xf32>) outs(%[[OUT_SLICE]] : tensor<?x?xf32>) -> tensor<?x?xf32>

//   CHECK-DAG:     %[[SIZE_CLAMPED_LOW0:.+]] = affine.max #[[$MAP]](%[[IDX0]])[%[[EXTRACT_SIZE0]]]
//   CHECK-DAG:     %[[SIZE_CLAMPED_HIGH0:.+]] = affine.min #[[$MAP1]](%[[SIZE_CLAMPED_LOW0]])[%[[STEP0]]]
//   CHECK-DAG:     %[[SIZE_CLAMPED_LOW1:.+]] = affine.max #[[$MAP]](%[[IDX1]])[%[[EXTRACT_SIZE1]]]
//   CHECK-DAG:     %[[SIZE_CLAMPED_HIGH1:.+]] = affine.min #[[$MAP1]](%[[SIZE_CLAMPED_LOW1]])[%[[STEP1]]]

//       CHECK:     %[[SLICED_COPY:.+]] = tensor.extract_slice %[[COPY]][0, 0] [%[[SIZE_CLAMPED_HIGH0]], %[[SIZE_CLAMPED_HIGH1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[SLICED_COPY]] into %[[SLICED_BBARG]]
//  CHECK-SAME:         [%[[IDX0]], %[[IDX1]]] [%[[SIZE_CLAMPED_HIGH0]], %[[SIZE_CLAMPED_HIGH1]]] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
//       CHECK:   return %[[FORALL_RESULT]]

// -----

module {
  func.func @fuse_rank_reduced_extract_slice_into_forall(%arg0: tensor<4x8xf32>, %arg1: index) -> tensor<?xf32> {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = scf.forall (%arg2, %arg3) = (0, 0) to (4, 8) step (2, 2) shared_outs(%arg4 = %0) -> (tensor<4x8xf32>) {
      %extracted_slice_0 = tensor.extract_slice %arg0[%arg2, %arg3] [2, 2] [1, 1] : tensor<4x8xf32> to tensor<2x2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3] [2, 2] [1, 1] : tensor<4x8xf32> to tensor<2x2xf32>
      %2 = linalg.copy ins(%extracted_slice_0 : tensor<2x2xf32>) outs(%extracted_slice_1 : tensor<2x2xf32>) -> tensor<2x2xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg4[%arg2, %arg3] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x8xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %extracted_slice = tensor.extract_slice %1[0, 0] [1, %arg1] [1, 1] : tensor<4x8xf32> to tensor<?xf32>
    return %extracted_slice : tensor<?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_consumers_of_result %0[0] : (!transform.any_op) -> !transform.any_op
    %2 = transform.iree.fuse_extract_slice_into_forall %1 into %0 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (2, d0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 0)>

// CHECK-LABEL: func @fuse_rank_reduced_extract_slice_into_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<4x8xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
//   CHECK-DAG:   %[[SLICED_OUT:.+]] = tensor.extract_slice %[[EMPTY]][0, 0] [1, %[[ARG1]]] [1, 1] : tensor<4x8xf32> to tensor<1x?xf32>
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall (%[[IDX0:.+]], %[[IDX1:.+]]) = (0, 0) to (4, 8) step (2, 2)
//  CHECK-SAME:       shared_outs(%[[SLICED_BBARG:.+]] = %[[SLICED_OUT]]) -> (tensor<1x?xf32>) {

//   CHECK-DAG:     %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]]{{.*}}[%[[IDX0]], %[[IDX1]]] [2, 2] [1, 1] : tensor<4x8xf32> to tensor<2x2xf32>
//   CHECK-DAG:     %[[OUT_SLICE:.+]] = tensor.extract_slice %[[EMPTY]]{{.*}}[%[[IDX0]], %[[IDX1]]] [2, 2] [1, 1] : tensor<4x8xf32> to tensor<2x2xf32>
//   CHECK-DAG:     %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<2x2xf32>) outs(%[[OUT_SLICE]] : tensor<2x2xf32>) -> tensor<2x2xf32>

//   CHECK-DAG:     %[[SIZE_CLAMPED_LOW0:.+]] = affine.max #[[$MAP]](%[[IDX0]])
//   CHECK-DAG:     %[[SIZE_CLAMPED_HIGH0:.+]] = affine.min #[[$MAP1]](%[[SIZE_CLAMPED_LOW0]])
//   CHECK-DAG:     %[[SIZE_CLAMPED_LOW1:.+]] = affine.max #[[$MAP2]](%[[IDX1]])[%[[ARG1]]]
//   CHECK-DAG:     %[[SIZE_CLAMPED_HIGH1:.+]] = affine.min #[[$MAP1]](%[[SIZE_CLAMPED_LOW1]])

//       CHECK:     %[[SLICED_COPY:.+]] = tensor.extract_slice %[[COPY]]
//  CHECK-SAME:       [0, 0] [%[[SIZE_CLAMPED_HIGH0]], %[[SIZE_CLAMPED_HIGH1]]] [1, 1] : tensor<2x2xf32> to tensor<?x?xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[SLICED_COPY]] into %[[SLICED_BBARG]]
//  CHECK-SAME:         [%[[IDX0]], %[[IDX1]]] [%[[SIZE_CLAMPED_HIGH0]], %[[SIZE_CLAMPED_HIGH1]]] [1, 1] : tensor<?x?xf32> into tensor<1x?xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FORALL_RESULT]] {{\[}}[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
//       CHECK:   return %[[COLLAPSE]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @fuse_extract_slice_into_rank_reduced_forall_slices(%arg0: tensor<4x8xf32>, %size1: index) -> tensor<4x?xf32> {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %0) -> (tensor<4x8xf32>) {
      %2 = affine.apply #map(%arg2)
      %extracted_slice_0 = tensor.extract_slice %arg0[0, %2] [1, 2] [1, 1] : tensor<4x8xf32> to tensor<2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[0, %2] [1, 2] [1, 1] : tensor<4x8xf32> to tensor<2xf32>
      %3 = linalg.copy ins(%extracted_slice_0 : tensor<2xf32>) outs(%extracted_slice_1 : tensor<2xf32>) -> tensor<2xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg3[0, %2] [1, 2] [1, 1] : tensor<2xf32> into tensor<4x8xf32>
      }
    } {mapping = [#gpu.thread<x>]}
    %extracted_slice = tensor.extract_slice %1[0, 0] [4, %size1] [1, 1] : tensor<4x8xf32> to tensor<4x?xf32>
    return %extracted_slice : tensor<4x?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %producer = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer = transform.get_consumers_of_result %producer[0] : (!transform.any_op) -> !transform.any_op
    %2 = transform.iree.fuse_extract_slice_into_forall %consumer into %producer
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (d0 * -2 + s0, 0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (2, d0)>

// CHECK-LABEL: func @fuse_extract_slice_into_rank_reduced_forall_slices
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<4x8xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
//   CHECK-DAG:   %[[SLICED_OUT:.+]] = tensor.extract_slice %[[EMPTY]][0, 0] [4, %[[ARG1]]] [1, 1] : tensor<4x8xf32> to tensor<4x?xf32>
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall (%[[IDX:.+]]) in (4)
//  CHECK-SAME:       shared_outs(%[[SLICED_BBARG:.+]] = %[[SLICED_OUT]]) -> (tensor<4x?xf32>) {

//   CHECK-DAG:     %[[SLICE_IDX:.+]] = affine.apply #[[$MAP]](%[[IDX]])
//   CHECK-DAG:     %[[IN_SLICE:.+]] = tensor.extract_slice %[[ARG0]]{{.*}}[0, %[[SLICE_IDX]]] [1, 2] [1, 1] : tensor<4x8xf32> to tensor<2xf32>
//   CHECK-DAG:     %[[OUT_SLICE:.+]] = tensor.extract_slice %[[EMPTY]]{{.*}}[0, %[[SLICE_IDX]]] [1, 2] [1, 1] : tensor<4x8xf32> to tensor<2xf32>
//   CHECK-DAG:     %[[COPY:.+]] = linalg.copy ins(%[[IN_SLICE]] : tensor<2xf32>) outs(%[[OUT_SLICE]] : tensor<2xf32>) -> tensor<2xf32>

//   CHECK-DAG:     %[[SIZE_CLAMPED_LOW:.+]] = affine.max #[[$MAP1]](%[[IDX]])[%[[ARG1]]]
//   CHECK-DAG:     %[[SIZE_CLAMPED_HIGH:.+]] = affine.min #[[$MAP2]](%[[SIZE_CLAMPED_LOW]])

//       CHECK:     %[[SLICED_COPY:.+]] = tensor.extract_slice %[[COPY]][0] [%[[SIZE_CLAMPED_HIGH]]] [1] : tensor<2xf32> to tensor<?xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[SLICED_COPY]] into %[[SLICED_BBARG]]
//  CHECK-SAME:         [0, %[[SLICE_IDX]]] [1, %[[SIZE_CLAMPED_HIGH]]] [1, 1] : tensor<?xf32> into tensor<4x?xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   return %[[FORALL_RESULT]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @fuse_extract_slice_into_multi_result_forall(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: index) -> (tensor<?xf32>, tensor<8xf32>) {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = tensor.empty() : tensor<8xf32>
    %2:2 = scf.forall (%arg3) in (4) shared_outs(%arg4 = %0, %arg5 = %1) -> (tensor<8xf32>, tensor<8xf32>) {
      %3 = affine.apply #map(%arg3)
      %extracted_slice_0 = tensor.extract_slice %arg0[%3] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%3] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %extracted_slice_2 = tensor.extract_slice %arg1[%3] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[%3] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %4 = linalg.copy ins(%extracted_slice_0 : tensor<2xf32>) outs(%extracted_slice_1 : tensor<2xf32>) -> tensor<2xf32>
      %5 = linalg.copy ins(%extracted_slice_2 : tensor<2xf32>) outs(%extracted_slice_3 : tensor<2xf32>) -> tensor<2xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg4[%3] [2] [1] : tensor<2xf32> into tensor<8xf32>
        tensor.parallel_insert_slice %5 into %arg5[%3] [2] [1] : tensor<2xf32> into tensor<8xf32>
      }
    } {mapping = [#gpu.thread<x>]}
    %extracted_slice = tensor.extract_slice %2#0[0] [%arg2] [1] : tensor<8xf32> to tensor<?xf32>
    return %extracted_slice, %2#1 : tensor<?xf32>, tensor<8xf32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_consumers_of_result %0[0] : (!transform.any_op) -> !transform.any_op
    %2 = transform.iree.fuse_extract_slice_into_forall %1 into %0 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @fuse_extract_slice_into_multi_result_forall

//       CHECK:   %[[FORALL_RESULT:.+]]:2 = scf.forall {{.*}} -> (tensor<?xf32>, tensor<8xf32>) {
//       CHECK:     scf.forall.in_parallel {
//   CHECK-DAG:       tensor.parallel_insert_slice {{.*}} : tensor<?xf32> into tensor<?xf32>
//   CHECK-DAG:       tensor.parallel_insert_slice {{.*}} : tensor<2xf32> into tensor<8xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   return %[[FORALL_RESULT]]#0, %[[FORALL_RESULT]]#1

// -----

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @no_fuse_extract_slice_with_offset(%arg0: tensor<8xf32>, %arg1: index) -> tensor<?xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %0) -> (tensor<8xf32>) {
      %2 = affine.apply #map(%arg2)
      %extracted_slice_0 = tensor.extract_slice %arg0[%2] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%2] [2] [1] : tensor<8xf32> to tensor<2xf32>
      %3 = linalg.copy ins(%extracted_slice_0 : tensor<2xf32>) outs(%extracted_slice_1 : tensor<2xf32>) -> tensor<2xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg3[%2] [2] [1] : tensor<2xf32> into tensor<8xf32>
      }
    } {mapping = [#gpu.thread<x>]}
    %extracted_slice = tensor.extract_slice %1[2] [%arg1] [1] : tensor<8xf32> to tensor<?xf32>
    return %extracted_slice : tensor<?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %producer = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer = transform.get_consumers_of_result %producer[0] : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{failed to fuse extract_slice op}}
    %2 = transform.iree.fuse_extract_slice_into_forall %consumer into %producer
     : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
