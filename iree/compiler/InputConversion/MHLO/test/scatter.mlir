// RUN: iree-opt -split-input-file -iree-mhlo-to-linalg-on-tensors -canonicalize %s | FileCheck %s

func @still_scatter(%arg0: tensor<1x2x64x12x64xf32>, %arg1: tensor<1x2x4xi32>, %arg2: tensor<1x2x2x1x2x64xf32>) -> tensor<1x2x64x12x64xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = "mhlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 2, 3, 4], inserted_window_dims = [1], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 2>, unique_indices = true} : (tensor<1x2x64x12x64xf32>, tensor<1x2x4xi32>, tensor<1x2x2x1x2x64xf32>) -> tensor<1x2x64x12x64xf32>
  return %0 : tensor<1x2x64x12x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK:       func @still_scatter
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[EXP:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0], [1], [2, 3], [4], [5], [6]]
// CHECK-DAG:     %[[CARG1:.+]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]]
// CHECK-DAG:     %[[CARG2:.+]] = tensor.collapse_shape %[[EXP]] {{\[\[}}0, 1, 2], [3], [4], [5], [6]]
// CHECK-DAG:     %[[SCF:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG0]])
// CHECK-DAG:       %[[MUL:.+]] = arith.muli %[[ARG3]], %[[C2]]
// CHECK-DAG:       %[[SLICE1:.+]] = tensor.extract_slice %[[CARG2]][%[[MUL]], 0, 0, 0, 0] [2, 1, 1, 2, 64] [1, 1, 1, 1, 1]
// CHECK-DAG:       %[[EX1:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C0]]]
// CHECK-DAG:       %[[CAST1:.+]] = arith.index_cast %[[EX1]]
// CHECK-DAG:       %[[EX2:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C1]]]
// CHECK-DAG:       %[[CAST2:.+]] = arith.index_cast %[[EX2]]
// CHECK-DAG:       %[[EX3:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C2]]]
// CHECK-DAG:       %[[CAST3:.+]] = arith.index_cast %[[EX3]]
// CHECK-DAG:       %[[EX4:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C3]]]
// CHECK-DAG:       %[[CAST4:.+]] = arith.index_cast %[[EX4]]
// CHECK-DAG:       %[[SLICE2:.+]] = tensor.extract_slice %arg0[%[[CAST1]], %[[CAST2]], %[[CAST3]], %[[CAST4]], 0] [2, 1, 1, 2, 64] [1, 1, 1, 1, 1]
// CHECK-DAG:       %[[GENERIC:.+]] = linalg.generic {
// CHECK-SAME        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:           ins(%[[SLICE1]] : tensor<2x1x1x2x64xf32>)
// CHECK-SAME:           outs(%[[SLICE2]] : tensor<2x1x1x2x64xf32>)
// CHECK-NEXT:        %[[ARG5:[a-zA-Z0-9_]*]]
// CHECK-SAME:        %[[ARG6:[a-zA-Z0-9_]*]]
// CHECK:             %[[ADD:.+]] = arith.addf %[[ARG5]], %[[ARG6]]
// CHECK:             linalg.yield %[[ADD]]
// CHECK:           %[[INSERT:.+]] = tensor.insert_slice %[[GENERIC]] into %[[ARG0]][%[[CAST1]], %[[CAST2]], %[[CAST3]], %[[CAST4]], 0] [2, 1, 1, 2, 64] [1, 1, 1, 1, 1]
// CHECK:           scf.yield %[[INSERT]]
// CHECK:         return %[[SCF]]

// -----

func @still_scatter_dynamic(%arg0: tensor<1x2x64x12x64xf32>, %arg1: tensor<?x1x4xi32>, %arg2: tensor<?x1x2x1x2x64xf32>) -> tensor<1x2x64x12x64xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = "mhlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 2, 3, 4], inserted_window_dims = [1], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 2>, unique_indices = true} : (tensor<1x2x64x12x64xf32>, tensor<?x1x4xi32>, tensor<?x1x2x1x2x64xf32>) -> tensor<1x2x64x12x64xf32>
  return %0 : tensor<1x2x64x12x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK:       func @still_scatter_dynamic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[EXP:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0], [1], [2, 3], [4], [5], [6]]
// CHECK:         %[[CARG1:.+]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]]
// CHECK:         %[[CARG2:.+]] = tensor.collapse_shape %[[EXP]] {{\[\[}}0, 1, 2], [3], [4], [5], [6]]
// CHECK:         %[[DIM:.+]] = tensor.dim %[[CARG1]], %[[C0]]
// CHECK:         %[[SCF:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[DIM]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG0]])
// CHECK:           %[[MUL:.+]] = arith.muli %[[ARG3]], %[[C2]]
// CHECK:           %[[SLICE1:.+]] = tensor.extract_slice %[[CARG2]][%[[MUL]], 0, 0, 0, 0] [2, 1, 1, 2, 64] [1, 1, 1, 1, 1]
// CHECK:           %[[EX1:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C0]]]
// CHECK:           %[[CAST1:.+]] = arith.index_cast %[[EX1]]
// CHECK:           %[[EX2:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C1]]]
// CHECK:           %[[CAST2:.+]] = arith.index_cast %[[EX2]]
// CHECK:           %[[EX3:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C2]]]
// CHECK:           %[[CAST3:.+]] = arith.index_cast %[[EX3]]
// CHECK:           %[[EX4:.+]] = tensor.extract %[[CARG1]][%[[ARG3]], %[[C3]]]
// CHECK:           %[[CAST4:.+]] = arith.index_cast %[[EX4]]
// CHECK:           %[[SLICE2:.+]] = tensor.extract_slice %arg0[%[[CAST1]], %[[CAST2]], %[[CAST3]], %[[CAST4]], 0] [2, 1, 1, 2, 64] [1, 1, 1, 1, 1]
// CHECK:           %[[GENERIC:.+]] = linalg.generic {
// CHECK-SAME        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:           ins(%[[SLICE1]] : tensor<2x1x1x2x64xf32>)
// CHECK-SAME:           outs(%[[SLICE2]] : tensor<2x1x1x2x64xf32>)
// CHECK-NEXT:        %[[ARG5:[a-zA-Z0-9_]*]]
// CHECK-SAME:        %[[ARG6:[a-zA-Z0-9_]*]]
// CHECK:             %[[ADD:.+]] = arith.addf %[[ARG5]], %[[ARG6]]
// CHECK:             linalg.yield %[[ADD]]
// CHECK:           %[[INSERT:.+]] = tensor.insert_slice %[[GENERIC]] into %[[ARG0]][%[[CAST1]], %[[CAST2]], %[[CAST3]], %[[CAST4]], 0] [2, 1, 1, 2, 64] [1, 1, 1, 1, 1]
// CHECK:           scf.yield %[[INSERT]]
// CHECK:         return %[[SCF]]
