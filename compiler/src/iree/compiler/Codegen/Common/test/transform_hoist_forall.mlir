// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @hoist_forall(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %4 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %0) -> (tensor<128x128xf32>) {
      %3 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %arg2) -> (tensor<128x128xf32>) {
        %4 = affine.apply #map(%arg5)
        %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
        %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
        %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.yield %3 : tensor<128x128xf32>
    }
    return %4 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.hoist_forall_from_for
    } : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func @hoist_forall

//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<128x128xf32>
//       CHECK:   %[[FORALL:.+]] = scf.forall (%[[ID0:.+]], %[[ID1:.+]]) in (64, 1) shared_outs(%[[OUTS:.+]] = %[[INIT]]) -> (tensor<128x128xf32>)
//  CHECK-NEXT:     %[[SLICE_ID:.+]] = affine.apply #[[$MAP]](%[[ID0]])
//  CHECK-NEXT:     %[[SLICE:.+]] = tensor.extract_slice %[[OUTS]][%[[SLICE_ID]], %[[ID1]]] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
//  CHECK-NEXT:     %[[FOR:.+]] = scf.for {{.*}} iter_args(%[[FOR_INIT:.+]] = %[[SLICE]]) -> (tensor<2x128xf32>)
//       CHECK:       %[[COPY:.+]] = linalg.copy {{.*}} outs(%[[FOR_INIT]] : tensor<2x128xf32>)
//       CHECK:       scf.yield %[[COPY]]
//       CHECK:     scf.forall.in_parallel
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[FOR]] into %[[OUTS]][%[[SLICE_ID]], %[[ID1]]] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
//       CHECK:   return %[[FORALL]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @no_hoist_loop_dependent(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %4 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %0) -> (tensor<128x128xf32>) {
      %3 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %arg2) -> (tensor<128x128xf32>) {
        %4 = affine.apply #map(%arg1)
        %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
        %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
        %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.yield %3 : tensor<128x128xf32>
    }
    return %4 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.hoist_forall_from_for
    } : !transform.any_op
    transform.yield
  }
}

// Verify we don't hoist when the slices are dependent on the loop.
// CHECK-LABEL: func @no_hoist_loop_dependent

//       CHECK:   %[[FOR:.+]] = scf.for
//       CHECK:     scf.forall
//       CHECK:   return %[[FOR]]
