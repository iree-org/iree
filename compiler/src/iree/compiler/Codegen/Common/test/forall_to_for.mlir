// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-forall-to-for))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @simple_forall
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>)
func.func @simple_forall(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  // CHECK: scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ITER0:.+]] = %[[ARG1]]) -> (tensor<?x?xf32>) {
  // CHECK:   scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ITER1:.+]] = %[[ITER0]]) -> (tensor<?x?xf32>) {
  %0 = scf.forall (%i, %j) = (0, 0) to (%dim0, %dim1) step (4, 4) shared_outs(%arg2 = %arg1) -> (tensor<?x?xf32>) {
    // CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[IV1]]] [4, 4] [1, 1]
    %extracted_slice = tensor.extract_slice %arg0[%i, %j] [4, 4] [1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
    // CHECK:     %[[INSERT:.+]] = tensor.insert_slice %[[SLICE]] into %[[ITER1]][%[[IV0]], %[[IV1]]] [4, 4] [1, 1]
    // CHECK:     scf.yield %[[INSERT]] : tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %extracted_slice into %arg2[%i, %j] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
    }
  }
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @forall_with_mapping_not_converted
func.func @forall_with_mapping_not_converted(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: scf.forall
  %0 = scf.forall (%i, %j) = (0, 0) to (32, 32) step (4, 4) shared_outs(%arg1 = %arg0) -> (tensor<32x32xf32>) {
    %slice = tensor.extract_slice %arg1[%i, %j] [4, 4] [1, 1] : tensor<32x32xf32> to tensor<4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %arg1[%i, %j] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<32x32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @nested_forall_inner_only
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?x16x1xf32>, %[[ARG1:.+]]: tensor<?x?x16x1xf32>, %[[ARG2:.+]]: tensor<?x16xf32>)
func.func @nested_forall_inner_only(%arg0: tensor<?x?x16x1xf32>, %arg1: tensor<?x?x16x1xf32>, %arg2: tensor<?x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x16x1xf32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x16x1xf32>
  %dim_1 = tensor.dim %arg1, %c0 : tensor<?x?x16x1xf32>
  %0 = tensor.empty(%dim, %dim_1) : tensor<?x?x16x16xf32>
  // CHECK: scf.forall
  %1 = scf.forall (%arg3, %arg4) in (%dim, %dim_1) shared_outs(%arg5 = %0) -> (tensor<?x?x16x16xf32>) {
    %extracted_slice = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : tensor<?x?x16x16xf32> to tensor<1x1x16x16xf32>
    // CHECK: %[[FOR:.+]] = scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ITER:.+]] = %{{.+}}) -> (tensor<1x1x16x16xf32>) {
    %2 = scf.forall (%arg6) = (0) to (16) step (2) shared_outs(%arg7 = %extracted_slice) -> (tensor<1x1x16x16xf32>) {
      %3 = tensor.empty() : tensor<1x1x2x16xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x1x2x16xf32>) -> tensor<1x1x2x16xf32>
      // CHECK: %[[INNER_FOR:.+]] = scf.for
      %5 = scf.for %arg8 = %c0 to %dim_0 step %c1 iter_args(%arg9 = %4) -> (tensor<1x1x2x16xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg0[%arg3, %arg8, %arg6, 0] [1, 1, 2, 1] [1, 1, 1, 1] : tensor<?x?x16x1xf32> to tensor<1x1x2x1xf32>
        %extracted_slice_3 = tensor.extract_slice %arg1[%arg4, %arg8, 0, 0] [1, 1, 16, 1] [1, 1, 1, 1] : tensor<?x?x16x1xf32> to tensor<1x1x16x1xf32>
        %6 = linalg.mmt4d ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x1x2x1xf32>, tensor<1x1x16x1xf32>) outs(%arg9 : tensor<1x1x2x16xf32>) -> tensor<1x1x2x16xf32>
        scf.yield %6 : tensor<1x1x2x16xf32>
      }
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg3, %arg6] [1, 2] [1, 1] : tensor<?x16xf32> to tensor<1x2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[0, 0, %arg6, 0] [1, 1, 2, 16] [1, 1, 1, 1] : tensor<1x1x16x16xf32> to tensor<1x1x2x16xf32>
      %6 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%5, %extracted_slice_0 : tensor<1x1x2x16xf32>, tensor<1x2xf32>) outs(%extracted_slice_1 : tensor<1x1x2x16xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %7 = arith.addf %in, %in_2 : f32
        %8 = arith.maximumf %7, %cst : f32
        linalg.yield %8 : f32
      } -> tensor<1x1x2x16xf32>
      // CHECK: %[[INSERT:.+]] = tensor.insert_slice %{{.+}} into %[[ITER]][%{{.+}}, %{{.+}}, %[[IV]], %{{.+}}] [1, 1, 2, 16] [1, 1, 1, 1]
      // CHECK: scf.yield %[[INSERT]] : tensor<1x1x16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg7[%c0, %c0, %arg6, %c0] [1, 1, 2, 16] [1, 1, 1, 1] : tensor<1x1x2x16xf32> into tensor<1x1x16x16xf32>
      }
    }
    // CHECK: scf.forall.in_parallel
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : tensor<1x1x16x16xf32> into tensor<?x?x16x16xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  return
}

// -----

// CHECK-LABEL: func.func @multiple_results
func.func @multiple_results(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %c4 = arith.constant 4 : index
  %cst = arith.constant 1.0 : f32
  // CHECK: %{{.+}}:2 = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ITER0:.+]] = %arg0, %[[ITER1:.+]] = %arg1) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  // CHECK:   %{{.+}}:2 = scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[INNER_ITER0:.+]] = %[[ITER0]], %[[INNER_ITER1:.+]] = %[[ITER1]]) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %0:2 = scf.forall (%i, %j) = (0, 0) to (32, 32) step (4, 4) shared_outs(%arg2 = %arg0, %arg3 = %arg1) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %slice0 = tensor.extract_slice %arg2[%i, %j] [4, 4] [1, 1] : tensor<32x32xf32> to tensor<4x4xf32>
    %slice1 = tensor.extract_slice %arg3[%i, %j] [4, 4] [1, 1] : tensor<32x32xf32> to tensor<4x4xf32>
    %filled = linalg.fill ins(%cst : f32) outs(%slice0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: %[[INSERT0:.+]] = tensor.insert_slice %{{.+}} into %[[INNER_ITER0]][%[[IV0]], %[[IV1]]] [4, 4] [1, 1]
    // CHECK: %[[INSERT1:.+]] = tensor.insert_slice %{{.+}} into %[[INNER_ITER1]][%[[IV0]], %[[IV1]]] [4, 4] [1, 1]
    // CHECK: scf.yield %[[INSERT0]], %[[INSERT1]] : tensor<32x32xf32>, tensor<32x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %filled into %arg2[%i, %j] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<32x32xf32>
      tensor.parallel_insert_slice %slice1 into %arg3[%i, %j] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<32x32xf32>
    }
  }
  return %0#0, %0#1 : tensor<32x32xf32>, tensor<32x32xf32>
}
