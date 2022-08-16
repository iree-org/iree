// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors --canonicalize -cse %s | FileCheck %s

func.func @concatenate(%arg0: tensor<2x2xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x9xi32> {
  %cst = mhlo.constant dense<514> : tensor<2x3xi32>
  %0 = "mhlo.concatenate"(%arg0, %cst, %arg1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x4xi32>) -> tensor<2x9xi32>
  return %0 : tensor<2x9xi32>
}
// CHECK:       func.func @concatenate
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9$._-]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9$._-]+]]
// CHECK:         %[[CST:.+]] = arith.constant dense<514> : tensor<2x3xi32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [2, 9] : tensor<2x9xi32>
// CHECK:         %[[T0:.+]] = tensor.insert_slice %[[ARG0]] into %[[INIT]][0, 0] [2, 2] [1, 1]
// CHECK:         %[[T1:.+]] = tensor.insert_slice %[[CST]] into %[[T0]][0, 2] [2, 3] [1, 1]
// CHECK:         %[[T2:.+]] = tensor.insert_slice %[[ARG1]] into %[[T1]][0, 5] [2, 4] [1, 1]
// CHECK:         return %[[T2]]

// -----

func.func @scatter_update_scalar(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}
// CHECK:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, 0)>
// CHECK:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK:   #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:       func @scatter_update_scalar
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%[[ARG1]], %[[ARG2]] : tensor<1x1xi32>, tensor<1xi32>)
// CHECK-SAME:     outs(%[[ARG0]] : tensor<3xi32>) {
// CHECK:         ^bb0(%[[IDX_I32:.*]]: i32, %[[UPDATE:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK:           %[[CMP_IDX:.*]] = linalg.index 0 : index
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
// CHECK:           %[[PRED:.*]] = arith.cmpi eq, %[[CMP_IDX]], %[[IDX]] : index
// CHECK:           %[[SELECT:.*]] = arith.select %[[PRED]], %[[UPDATE]], %[[OUT]] : i32
// CHECK:           linalg.yield %[[SELECT]] : i32
// CHECK:         } -> tensor<3xi32>
// CHECK:         return %[[RES]] : tensor<3xi32>

// -----

func.func @scatter_update_slice(%arg0: tensor<6x3xi32>, %arg1: tensor<2x1xi32>,
                           %arg2: tensor<2x3xi32>) -> tensor<6x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  func.return %0 : tensor<6x3xi32>
}
// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d2, 0)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:   #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:       func @scatter_update_slice
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:      ins(%[[ARG1]], %[[ARG2]] : tensor<2x1xi32>, tensor<2x3xi32>)
// CHECK-SAME:     outs(%[[ARG0]] : tensor<6x3xi32>) {
// CHECK:         ^bb0(%[[IDX_I32:.*]]: i32, %[[UPDATE:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK:           %[[CMP_IDX:.*]] = linalg.index 0 : index
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
// CHECK:           %[[PRED:.*]] = arith.cmpi eq, %[[CMP_IDX]], %[[IDX]] : index
// CHECK:           %[[SELECT:.*]] = arith.select %[[PRED]], %[[UPDATE]], %[[OUT]] : i32
// CHECK:           linalg.yield %[[SELECT]] : i32
// CHECK:         } -> tensor<6x3xi32>
// CHECK:         return %[[RES]] : tensor<6x3xi32>

// -----

func.func @scatter_update_nontrivial_computation(%arg0: tensor<3xi32>,
  %arg1: tensor<6x1xi32>, %arg2: tensor<6xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<6x1xi32>, tensor<6xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d1, 0)>
// CHECK-DAG:   #[[MAP2:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK:       func @scatter_update_nontrivial_computation
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP0]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%[[ARG1]], %[[ARG2]] : tensor<6x1xi32>, tensor<6xi32>)
// CHECK-SAME:     outs(%[[ARG0]] : tensor<3xi32>) {
// CHECK:         ^bb0(%[[IDX_I32:.*]]: i32, %{{.*}}: i32, %{{.*}}: i32):
// CHECK:           %[[CMP_IDX:.*]] = linalg.index 0 : index
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
// CHECK:           %[[PRED:.*]] = arith.cmpi eq, %[[CMP_IDX]], %[[IDX]] : index
// CHECK:           %[[SELECT:.*]] =  arith.select %[[PRED]], %{{.*}}, %{{.*}}: i32
// CHECK:           linalg.yield %[[SELECT]] : i32
// CHECK:         } -> tensor<3xi32>
// CHECK:         return %[[RES]] : tensor<3xi32>
