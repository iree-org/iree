// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-tm-tensor-to-linalg-ext))" %s | FileCheck %s

func.func @sort(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xi64>) -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %0:2 = tm_tensor.sort dimension(1) outs(%arg0, %arg1 : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf ole, %arg2, %arg3 : f32
    tm_tensor.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}
// CHECK-LABEL: func.func @sort(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<3x4xf32>
// CHECK-SAME:    %[[INDICES:[a-zA-Z0-9]+]]: tensor<3x4xi64>
// CHECK:        %[[EMPTY_INPUT:.+]] = tensor.empty() : tensor<3x4xf32>
// CHECK:        %[[EMPTY_INDICES:.+]] = tensor.empty() : tensor<3x4xi64>
// CHECK:        %[[SORT:.+]]:2 = iree_linalg_ext.sort
// CHECK-SAME:     dimension(1)
// CHECK-SAME:     ins(%[[INPUT]], %[[INDICES]] : tensor<3x4xf32>, tensor<3x4xi64>)
// CHECK-SAME:     outs(%[[EMPTY_INPUT]], %[[EMPTY_INDICES]] : tensor<3x4xf32>, tensor<3x4xi64>) {
// CHECK:        ^bb0(%[[INP1:.+]]: f32, %[[INP2:.+]]: f32, %{{.*}}: i64, %{{.*}}: i64):
// CHECK:          %[[PREDICATE:.+]] = arith.cmpf ole, %[[INP1]], %[[INP2]] : f32
// CHECK:          iree_linalg_ext.yield %[[PREDICATE]] : i1
// CHECK:        } -> tensor<3x4xf32>, tensor<3x4xi64>
// CHECK:        return %[[SORT]]#0, %[[SORT]]#1 : tensor<3x4xf32>, tensor<3x4xi64>

// -----

// The second operand holds the positions along the sort dimension, as in the
// conversion of aten.sort. Elements that the comparator does not order are
// ordered by position, which makes the result the one of a stable sort.
func.func @sort_with_index_operand(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %empty = tensor.empty() : tensor<3x4xi64>
  %indices = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%empty : tensor<3x4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 1 : index
    %i64 = arith.index_cast %i : index to i64
    linalg.yield %i64 : i64
  } -> tensor<3x4xi64>
  %0:2 = tm_tensor.sort dimension(1) outs(%arg0, %indices : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf oge, %arg2, %arg3 : f32
    tm_tensor.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}
// CHECK-LABEL: func.func @sort_with_index_operand(
// CHECK:        %[[INDICES:.+]] = linalg.generic
// CHECK:        iree_linalg_ext.sort
// CHECK-SAME:     dimension(1)
// CHECK-SAME:     ins(%{{.+}}, %[[INDICES]] : tensor<3x4xf32>, tensor<3x4xi64>)
// CHECK:        ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32, %[[LHS_POS:.+]]: i64, %[[RHS_POS:.+]]: i64):
// CHECK:          %[[BEFORE:.+]] = arith.cmpf oge, %[[LHS]], %[[RHS]] : f32
// CHECK:          %[[AFTER:.+]] = arith.cmpf oge, %[[RHS]], %[[LHS]] : f32
// CHECK:          %[[ORDERED:.+]] = arith.xori %[[BEFORE]], %[[AFTER]] : i1
// CHECK:          %[[LOWER_POS:.+]] = arith.cmpi slt, %[[LHS_POS]], %[[RHS_POS]] : i64
// CHECK:          %[[RESULT:.+]] = arith.select %[[ORDERED]], %[[BEFORE]], %[[LOWER_POS]] : i1
// CHECK:          iree_linalg_ext.yield %[[RESULT]] : i1

// -----

// Positions along another dimension than the sorted one do not identify the
// elements of a sorted slice; the comparator is not changed.
func.func @sort_with_index_of_other_dimension(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %empty = tensor.empty() : tensor<3x4xi64>
  %indices = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%empty : tensor<3x4xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 0 : index
    %i64 = arith.index_cast %i : index to i64
    linalg.yield %i64 : i64
  } -> tensor<3x4xi64>
  %0:2 = tm_tensor.sort dimension(1) outs(%arg0, %indices : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf oge, %arg2, %arg3 : f32
    tm_tensor.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}
// CHECK-LABEL: func.func @sort_with_index_of_other_dimension(
// CHECK:        iree_linalg_ext.sort
// CHECK:        ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32, %{{.+}}: i64, %{{.+}}: i64):
// CHECK-NEXT:     %[[BEFORE:.+]] = arith.cmpf oge, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:     iree_linalg_ext.yield %[[BEFORE]] : i1

// -----
