// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @flatten_forall_thread_mapping(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (4, 16) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %3 = affine.apply #map(%arg5)
      %4 = affine.apply #map1(%arg6)
      %extracted_slice = tensor.extract_slice %arg0[%3, %4] [32, 8] [1, 1] : tensor<128x128xf32> to tensor<32x8xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%3, %4] [32, 8] [1, 1] : tensor<128x128xf32> to tensor<32x8xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<32x8xf32>) outs(%extracted_slice_0 : tensor<32x8xf32>) -> tensor<32x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%3, %4] [32, 8] [1, 1] : tensor<32x8xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    return %2 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %new_loop = transform.iree.flatten_forall_mapping %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 16)>

// CHECK-LABEL: func @flatten_forall_thread_mapping
//       CHECK:   scf.forall (%[[FLAT_ID:.+]]) in (64)
//       CHECK:     %[[IDS:.+]]:2 = affine.delinearize_index %[[FLAT_ID]] into (%c4, %c16) : index, index
//   CHECK-DAG:     %[[IDX:.+]] = affine.apply #[[$MAP]](%[[IDS]]#0)
//   CHECK-DAG:     %[[IDY:.+]] = affine.apply #[[$MAP1]](%[[IDS]]#1)
//       CHECK:   } {mapping = [#gpu.thread<linear_dim_0>]}

// -----

#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @flatten_forall_warp_mapping(%arg0: tensor<128x128xf32>, %dimx: index, %dimy: index) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (%dimx, %dimy) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %3 = affine.apply #map(%arg5)
      %4 = affine.apply #map1(%arg6)
      %extracted_slice = tensor.extract_slice %arg0[%3, %4] [32, 8] [1, 1] : tensor<128x128xf32> to tensor<32x8xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%3, %4] [32, 8] [1, 1] : tensor<128x128xf32> to tensor<32x8xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<32x8xf32>) outs(%extracted_slice_0 : tensor<32x8xf32>) -> tensor<32x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%3, %4] [32, 8] [1, 1] : tensor<32x8xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]}
    return %2 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %new_loop = transform.iree.flatten_forall_mapping %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>

// CHECK-LABEL: func @flatten_forall_warp_mapping
//  CHECK-SAME:   %[[DIMX:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[DIMY:[A-Za-z0-9]+]]: index
//       CHECK:   %[[FLAT_UB:.+]] = affine.apply #[[$MAP2]]()[%[[DIMX]], %[[DIMY]]]
//       CHECK:   scf.forall (%[[FLAT_ID:.+]]) in (%[[FLAT_UB]])
//       CHECK:     %[[IDS:.+]]:2 = affine.delinearize_index %[[FLAT_ID]] into (%[[DIMX]], %[[DIMY]]) : index, index
//   CHECK-DAG:     %[[IDX:.+]] = affine.apply #[[$MAP]](%[[IDS]]#0)
//   CHECK-DAG:     %[[IDY:.+]] = affine.apply #[[$MAP1]](%[[IDS]]#1)
//       CHECK:   } {mapping = [#gpu.warp<linear_dim_0>]}
