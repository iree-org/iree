// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 8 + d2)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//   CHECK-DAG:     %[[OUTID0:.+]] = affine.apply #[[$MAP]](%[[IDX]])
//   CHECK-DAG:     %[[OUTID1:.+]] = affine.apply #[[$MAP]](%[[IDY]])

//       CHECK:     %[[LOOP:.+]] = scf.for %[[I:.+]] = %c0 to %c64{{.*}} step %c64{{.*}} iter_args(%[[ITER:.+]] = %[[ALLOC]]) -> (tensor<128x128xf32>)
//       CHECK:       %[[LINEARID:.+]] = affine.apply #[[$MAP2]](%[[I]], %[[IDX]], %[[IDY]])
//       CHECK:       %[[IDS:.+]]:2 = affine.delinearize_index %[[LINEARID]] into (%c1, %c64) : index, index
//       CHECK:       %[[INID0:.+]] = affine.apply #[[$MAP3]](%[[IDS]]#1)
//       CHECK:       %[[INSLICE0:.+]] = tensor.extract_slice %[[ARG0]][%[[INID0]], %[[IDS]]#0] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
//       CHECK:       %[[INSLICE1:.+]] = tensor.extract_slice %[[ITER]][%[[INID0]], %[[IDS]]#0] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[INSLICE0]] : tensor<2x128xf32>) outs(%[[INSLICE1]] : tensor<2x128xf32>) -> tensor<2x128xf32>
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[ITER]][%[[INID0]], %[[IDS]]#0] [2, 128] [1, 1]
//       CHECK:       scf.yield %[[INSERT]]

//       CHECK:     %[[SHUFFLE:.+]] = iree_gpu.barrier_region %[[LOOP]]
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<128x128xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][%[[OUTID0]], %[[OUTID1]]] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]]
//       CHECK:     } : tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:     %[[OUTSLICE:.+]] = tensor.extract_slice %[[INIT]][%[[OUTID0]], %[[OUTID1]]] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
//       CHECK:     %[[MM:.+]] = linalg.matmul ins(%[[SHUFFLE]], %[[SHUFFLE]] : tensor<16x16xf32>, tensor<16x16xf32>)
//  CHECK-SAME:       outs(%[[OUTSLICE]] : tensor<16x16xf32>) -> tensor<16x16xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[MM]] into %[[INIT]][%[[OUTID0]], %[[OUTID1]]] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//       CHECK:     %[[LOOP:.+]] = scf.for {{.*}} iter_args(%[[INIT:.+]] = %[[ALLOC]])
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %{{.*}} into %[[INIT]]
//       CHECK:     %[[SHUFFLE:.+]] = iree_gpu.barrier_region %[[LOOP]]
//       CHECK:       } : tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:   } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall_with_reshape(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    %expand = tensor.expand_shape %2 [[0, 1], [2]] output_shape [2, 64, 128] : tensor<128x128xf32> into tensor<2x64x128xf32>
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %expand[0, %6, %7] [1, 16, 16] [1, 1, 1] : tensor<2x64x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_forall_with_reshape
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//       CHECK:     %[[LOOP:.+]] = scf.for {{.*}} iter_args(%[[INIT:.+]] = %[[ALLOC]])
//       CHECK:       %[[INSERT:.+]] = tensor.insert_slice %{{.*}} into %[[INIT]]
//       CHECK:     %[[SHUFFLE:.+]] = iree_gpu.barrier_region %[[LOOP]]
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<128x128xf32>):
//       CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[INTERMEDIATE]] {{\[}}[0, 1], [2]{{\]}} output_shape [2, 64, 128]
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[EXPAND]][0, %{{.*}}, %{{.*}}] [1, 16, 16] [1, 1, 1] : tensor<2x64x128xf32> to tensor<16x16xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]]
//       CHECK:       } : tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:   } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d1 + d0 * 16)>
module {
  func.func @fuse_thread_forall_with_warp_and_lane(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %3 = scf.forall (%arg9, %arg10) in (2, 2) shared_outs(%arg8 = %empty) -> (tensor<128x128xf32>) {
      %extracted_slice_2 = tensor.extract_slice %arg8[%arg9, %arg10] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
      %9 = scf.forall (%arg5, %arg6) in (4, 4) shared_outs(%arg7 = %extracted_slice_2) -> (tensor<64x64xf32>) {
        %6 = affine.apply #map1(%arg5, %arg9)
        %7 = affine.apply #map1(%arg6, %arg10)
        %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
        %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
        %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
        }
      } {mapping = [#iree_gpu.lane_id<1>, #iree_gpu.lane_id<0>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %9 into %arg8[%arg9, %arg10] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %lane_consumer, %warp_consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %lane_consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 8 + d2 * 4)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d1 + d2 * 4 + d3 * 32 + d4 * 16)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_thread_forall_with_warp_and_lane
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[W_IDX:.+]], %[[W_IDY:.+]]) in (2, 2) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//       CHECK:     scf.forall (%[[L_IDX:.+]], %[[L_IDY:.+]]) in (4, 4) {{.*}} -> (tensor<64x64xf32>)

//       CHECK:       %[[LOOP:.+]] = scf.for %[[I:.+]] = %c0 to %c64{{.*}} step %c64{{.*}} iter_args(%[[ITER:.+]] = %[[ALLOC]]) -> (tensor<128x128xf32>)
//       CHECK:         %[[FLAT_ID:.+]] = affine.apply #[[$MAP4]](%[[I]], %[[L_IDY]], %[[L_IDX]], %[[W_IDX]], %[[W_IDY]])
//       CHECK:         %[[IDS:.+]]:2 = affine.delinearize_index %[[FLAT_ID]] into (%c1, %c64) : index, index
//       CHECK:         %[[IDX:.+]] = affine.apply #[[$MAP5]](%[[IDS]]#1)
//       CHECK:         %[[COPY:.+]] = linalg.copy
//       CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[COPY]] into %[[ITER]][%[[IDX]], %[[IDS]]#0] [2, 128]
//       CHECK:         scf.yield %[[INSERT]]

//       CHECK:       %[[SHUFFLE:.+]] = iree_gpu.barrier_region %[[LOOP]]
//       CHECK:       } : tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:     } {mapping = [#iree_gpu.lane_id<1>, #iree_gpu.lane_id<0>]}
//       CHECK:   } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall_different_thread_count(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5) in (32) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, 0] [4, 128] [1, 1] : tensor<128x128xf32> to tensor<4x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, 0] [4, 128] [1, 1] : tensor<128x128xf32> to tensor<4x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<4x128xf32>) outs(%extracted_slice_0 : tensor<4x128xf32>) -> tensor<4x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, 0] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<x>]}
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>

// CHECK-LABEL: func @fuse_forall_different_thread_count
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//       CHECK:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) {{.*}} -> (tensor<128x128xf32>) {
//       CHECK:     %[[LINEARID:.+]] = affine.apply #[[$MAP1]](%[[IDX]], %[[IDY]])
//       CHECK:     %[[LOOP:.+]] = scf.for %[[I:.+]] = %[[LINEARID]] to %c32{{.*}} step %c64{{.*}} iter_args(%[[ITER:.+]] = %[[ALLOC]])
//       CHECK:       %[[IDS:.+]] = affine.delinearize_index %[[I]] into (%c32) : index
//       CHECK:       scf.yield
//       CHECK:     iree_gpu.barrier_region %[[LOOP]]
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall_dynamic_thread_count(%arg0: tensor<128x128xf32>, %x: index, %y: index, %z: index) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6, %arg7) in (%x, %y, %z) shared_outs(%arg8 = %0) -> (tensor<128x128xf32>) {
      %slice = tensor.extract_slice %arg0[%arg5, %arg6] [4, 128] [1, 1] : tensor<128x128xf32> to tensor<4x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %arg8[%arg7, 0] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]}
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0, s1, s2] -> (s2 * (s0 * s1))>

// CHECK-LABEL: func @fuse_forall_dynamic_thread_count
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>
//  CHECK-SAME:   %[[X:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[Y:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[Z:[A-Za-z0-9]+]]: index

//       CHECK:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) {{.*}} -> (tensor<128x128xf32>) {
//   CHECK-DAG:     %[[LINEARID:.+]] = affine.apply #[[$MAP1]](%[[IDX]], %[[IDY]])
//   CHECK-DAG:     %[[PRODCOUNT:.+]] = affine.apply #[[$MAP3]]()[%[[X]], %[[Y]], %[[Z]]]
//       CHECK:     %[[LOOP:.+]] = scf.for %[[I:.+]] = %[[LINEARID]] to %[[PRODCOUNT]] step %c64{{.*}} iter_args(%[[ITER:.+]] = %[[ALLOC]])
//       CHECK:       %[[IDS:.+]] = affine.delinearize_index %[[I]] into (%[[Z]], %[[Y]], %[[X]]) : index
//       CHECK:       scf.yield
//       CHECK:     iree_gpu.barrier_region %[[LOOP]]
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
