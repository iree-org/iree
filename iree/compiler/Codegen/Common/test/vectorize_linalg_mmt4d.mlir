// RUN: iree-opt -split-input-file -iree-codegen-vectorize-linalg-mmt4d -canonicalize -cse %s | IreeFileCheck %s

func @tiled_mmt4d(%lhs: memref<1x1x4x4xf32>, %rhs: memref<1x1x4x4xf32>, %dst: memref<1x1x4x4xf32>) {
    linalg.mmt4d ins(%lhs, %rhs: memref<1x1x4x4xf32>, memref<1x1x4x4xf32>) outs(%dst: memref<1x1x4x4xf32>)
    return
}

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: func @tiled_mmt4d(%[[LHS:.+]]: memref<1x1x4x4xf32>, %[[RHS:.+]]: memref<1x1x4x4xf32>, %[[DST:.+]]: memref<1x1x4x4xf32>
//      CHECK:   %[[LHS_4DVEC:.+]] = vector.transfer_read %[[LHS]]
//      CHECK:   %[[RHS_4DVEC:.+]] = vector.transfer_read %[[RHS]]
//      CHECK:   %[[DST_4DVEC:.+]] = vector.transfer_read %[[DST]]
//      CHECK:   %[[LHS_2DVEC:.+]] = vector.shape_cast %[[LHS_4DVEC]] : vector<1x1x4x4xf32> to vector<4x4xf32>
//      CHECK:   %[[RHS_2DVEC:.+]] = vector.shape_cast %[[RHS_4DVEC]] : vector<1x1x4x4xf32> to vector<4x4xf32>
//      CHECK:   %[[DST_2DVEC:.+]] = vector.shape_cast %[[DST_4DVEC]] : vector<1x1x4x4xf32> to vector<4x4xf32>
//      CHECK:   %[[RESULT_2D:.+]] = vector.contract
// CHECK-SAME:        indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
// CHECK-SAME:        %[[LHS_2DVEC]], %[[RHS_2DVEC]], %[[DST_2DVEC]] : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
//      CHECK:   %[[RESULT_4D:.+]] = vector.shape_cast %[[RESULT_2D]] : vector<4x4xf32> to vector<1x1x4x4xf32>

