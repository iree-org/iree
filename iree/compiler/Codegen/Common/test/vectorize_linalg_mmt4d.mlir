// RUN: iree-opt -split-input-file -iree-codegen-vectorize-linalg-mmt4d -canonicalize -cse %s | FileCheck %s

func @tiled_mmt4d_4x4x4_f32(%lhs: tensor<1x1x4x4xf32>, %rhs: tensor<1x1x4x4xf32>, %dst: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = linalg.mmt4d ins(%lhs, %rhs: tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) outs(%dst: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    return %0 : tensor<1x1x4x4xf32>
}

// CHECK:       #[[MAP0:.+]] = affine_map<([[D0:.*]], [[D1:.*]], [[D2:.*]]) -> ([[D0]], [[D2]])>
// CHECK:       #[[MAP1:.+]] = affine_map<([[D0]], [[D1]], [[D2]]) -> ([[D1]], [[D2]])>
// CHECK:       #[[MAP2:.+]] = affine_map<([[D0]], [[D1]], [[D2]]) -> ([[D0]], [[D1]])>
// CHECK:       func @tiled_mmt4d_4x4x4_f32(
// CHECK-SAME:      %[[LHS:[[:alnum:]]+]]: tensor<1x1x4x4xf32>
// CHECK-SAME:      %[[RHS:[[:alnum:]]+]]: tensor<1x1x4x4xf32>
// CHECK-SAME:      %[[ACC:[[:alnum:]]+]]: tensor<1x1x4x4xf32>
// CHECK-SAME:        -> tensor<1x1x4x4xf32> {
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C0F32:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[LHSV:.+]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[C0F32]]
// CHECK:         %[[RHSV:.+]] = vector.transfer_read %[[RHS]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[C0F32]]
// CHECK:         %[[ACCV:.+]] = vector.transfer_read %[[ACC]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[C0F32]]
// CHECK:         %[[CONTRACT:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHSV]], %[[RHSV]], %[[ACCV]] :
// CHECK-SAME:      vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[CONTRACT]], %[[ACC]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK:         return %[[WRITE]] : tensor<1x1x4x4xf32>
// CHECK:       }

// -----

func @tiled_mmt4d_8x2x4_i8(%lhs: tensor<1x1x8x2xi8>, %rhs: tensor<1x1x4x2xi8>, %dst: tensor<1x1x8x4xi32>) -> tensor<1x1x8x4xi32> {
    %0 = linalg.mmt4d ins(%lhs, %rhs: tensor<1x1x8x2xi8>, tensor<1x1x4x2xi8>) outs(%dst: tensor<1x1x8x4xi32>) -> tensor<1x1x8x4xi32>
    return %0 : tensor<1x1x8x4xi32>
}

// CHECK:       #[[MAP0:.+]] = affine_map<([[D0:.*]], [[D1:.*]], [[D2:.*]]) -> ([[D0]], [[D2]])>
// CHECK:       #[[MAP1:.+]] = affine_map<([[D0]], [[D1]], [[D2]]) -> ([[D1]], [[D2]])>
// CHECK:       #[[MAP2:.+]] = affine_map<([[D0]], [[D1]], [[D2]]) -> ([[D0]], [[D1]])>
// CHECK:       func @tiled_mmt4d_8x2x4_i8(
// CHECK-SAME:      %[[LHS:[[:alnum:]]+]]: tensor<1x1x8x2xi8>
// CHECK-SAME:      %[[RHS:[[:alnum:]]+]]: tensor<1x1x4x2xi8>
// CHECK-SAME:      %[[ACC:[[:alnum:]]+]]: tensor<1x1x8x4xi32>
// CHECK-SAME:        -> tensor<1x1x8x4xi32> {
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C0I8:.+]] = arith.constant 0 : i8
// CHECK-DAG:     %[[C0I32:.+]] = arith.constant 0 : i32
// CHECK:         %[[LHSV:.+]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[C0I8]]
// CHECK:         %[[RHSV:.+]] = vector.transfer_read %[[RHS]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[C0I8]]
// CHECK:         %[[ACCV:.+]] = vector.transfer_read %[[ACC]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[C0I32]]
// CHECK:         %[[LHSVI32:.+]] = arith.extsi %[[LHSV]] : vector<8x2xi8> to vector<8x2xi32>
// CHECK:         %[[RHSVI32:.+]] = arith.extsi %[[RHSV]] : vector<4x2xi8> to vector<4x2xi32>
// CHECK:         %[[CONTRACT:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHSVI32]], %[[RHSVI32]], %[[ACCV]] :
// CHECK-SAME:      vector<8x2xi32>, vector<4x2xi32> into vector<8x4xi32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[CONTRACT]], %[[ACC]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK:         return %[[WRITE]] : tensor<1x1x8x4xi32>
// CHECK:       }
