// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#map = affine_map<(oi, oj, ok, ii, ij, ik) -> (oi, ok, ii, ik)>
#map1 = affine_map<(oi, oj, ok, ii, ij, ik) -> (ok, oj, ik, ij)>
#map2 = affine_map<(oi, oj, ok, ii, ij, ik) -> (oi, oj, ii, ij)>
func.func @convert_to_mfma_16x16x16(%lhs: tensor<2x2x16x16xf16>, %rhs: tensor<2x2x16x16xf16>, %acc: tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16>) outs(%acc : tensor<2x2x16x16xf32>) {
          ^bb0(%l: f16, %r: f16, %out: f32):
            %lext = arith.extf %l : f16 to f32
            %rext = arith.extf %r : f16 to f32
            %mul = arith.mulf %lext, %rext : f32
            %add = arith.addf %out, %mul : f32
            linalg.yield %add : f32
          } -> tensor<2x2x16x16xf32>
  return %0 : tensor<2x2x16x16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %1 = transform.iree.convert_to_multi_mma %0, kind(#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @convert_to_mfma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xf32>
//       CHECK:   iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:     iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16> into tensor<2x2x16x16xf32>

// -----

#map = affine_map<(ii, ij, ik) -> (ii, ik)>
#map1 = affine_map<(ii, ij, ik) -> (ik, ij)>
#map2 = affine_map<(ii, ij, ik) -> (ii, ij)>
func.func @convert_to_single_mfma_16x16x16(%lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>, %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) {
          ^bb0(%l: f16, %r: f16, %out: f32):
            %lext = arith.extf %l : f16 to f32
            %rext = arith.extf %r : f16 to f32
            %mul = arith.mulf %lext, %rext : f32
            %add = arith.addf %out, %mul : f32
            linalg.yield %add : f32
          } -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %1 = transform.iree.convert_to_multi_mma %0, kind(#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<() -> ()>

// CHECK-LABEL: func @convert_to_single_mfma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xf32>
//       CHECK:   iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]],
//  CHECK-SAME:     iterator_types = [],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : tensor<16x16xf16>, tensor<16x16xf16> into tensor<16x16xf32>

// -----

#map = affine_map<(ii, ij, ok, ik) -> (ok, ii, ik)>
#map1 = affine_map<(ii, ij, ok, ik) -> (ok, ij, ik)>
#map2 = affine_map<(ii, ij, ok, ik) -> (ii, ij)>
func.func @convert_to_mfma_16x16x16_transpose_b(%lhs: tensor<2x16x16xf16>, %rhs: tensor<2x16x16xf16>, %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%lhs, %rhs : tensor<2x16x16xf16>, tensor<2x16x16xf16>) outs(%acc : tensor<16x16xf32>) {
          ^bb0(%l: f16, %r: f16, %out: f32):
            %lext = arith.extf %l : f16 to f32
            %rext = arith.extf %r : f16 to f32
            %mul = arith.mulf %lext, %rext : f32
            %add = arith.addf %out, %mul : f32
            linalg.yield %add : f32
          } -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %1 = transform.iree.convert_to_multi_mma %0, kind(#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> ()>

// CHECK-LABEL: func @convert_to_mfma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<2x16x16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xf32>
//       CHECK:   iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP1]]],
//  CHECK-SAME:     iterator_types = [#iree_gpu.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
//  CHECK-SAME:     rhs_permutation = array<i64: 1, 0>
//  CHECK-SAME:     : tensor<2x16x16xf16>, tensor<2x16x16xf16> into tensor<16x16xf32>
