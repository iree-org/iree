// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_multi_mma(%lhs: tensor<2x3x4xf16>, %rhs: tensor<3x5x4xf16>, %acc: tensor<2x5x4xf32>) -> tensor<2x5x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<2x3x4xf16>, tensor<3x5x4xf16> into tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_iree_gpu
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @tensor_multi_mma

//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0, %c0, %c0], %[[CST]] {{.*}} : tensor<2x3x4xf16>, vector<2x3x4xf16>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg1[%c0, %c0, %c0], %[[CST]] {{.*}} : tensor<3x5x4xf16>, vector<3x5x4xf16>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg2[%c0, %c0, %c0], %[[CSTF32]] {{.*}} : tensor<2x5x4xf32>, vector<2x5x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
//  CHECK-SAME:     : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg2[%c0, %c0, %c0] {{.*}} : vector<2x5x4xf32>, tensor<2x5x4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @tensor_single_multi_mma(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>, %acc: tensor<4xf32>) -> tensor<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<4xf16>, tensor<4xf16> into tensor<4xf32>
  return %0 : tensor<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_iree_gpu
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @tensor_single_multi_mma

//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0], %[[CST]] {in_bounds = [true]} : tensor<4xf16>, vector<4xf16>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg1[%c0], %[[CST]] {in_bounds = [true]} : tensor<4xf16>, vector<4xf16>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg2[%c0], %[[CSTF32]] {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

#iterator_types = [
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<reduction>,
  #linalg.iterator_type<reduction>
]

func.func @scaled_tensor_multi_mma(%arg0: tensor<3x5x1x32xf4E2M1FN>, %arg1: tensor<3x5x1xf8E8M0FNU>,
  %arg2: tensor<5x1x7x32xf8E4M3FN>, %arg3: tensor<5x7x1xf8E8M0FNU>,
  %arg4: tensor<3x7x4xf32>) -> tensor<3x7x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4) {
    indexing_maps = #contraction_accesses,
    iterator_types = #iterator_types,
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>
    } : tensor<3x5x1x32xf4E2M1FN>, tensor<3x5x1xf8E8M0FNU>,
      tensor<5x1x7x32xf8E4M3FN>, tensor<5x7x1xf8E8M0FNU>
      into tensor<3x7x4xf32>
  return %0 : tensor<3x7x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_iree_gpu
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @scaled_tensor_multi_mma

//   CHECK-DAG:   %[[CSTFP4:.+]] = arith.constant 0.000000e+00 : f4E2M1FN
//   CHECK-DAG:   %[[CSTFP8:.+]] = arith.constant 0.000000e+00 : f8E4M3FN
//   CHECK-DAG:   %[[CSTSCALE:.+]] = arith.constant 5.877470e-39 : f8E8M0FNU
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %[[CSTFP4]] {{.*}} : tensor<3x5x1x32xf4E2M1FN>, vector<3x5x1x32xf4E2M1FN>
//   CHECK-DAG:   %[[LHS_SCALE:.+]] = vector.transfer_read %arg1[%c0, %c0, %c0], %[[CSTSCALE]] {{.*}} : tensor<3x5x1xf8E8M0FNU>, vector<3x5x1xf8E8M0FNU>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg2[%c0, %c0, %c0, %c0], %[[CSTFP8]] {{.*}} : tensor<5x1x7x32xf8E4M3FN>, vector<5x1x7x32xf8E4M3FN>
//   CHECK-DAG:   %[[RHS_SCALE:.+]] = vector.transfer_read %arg3[%c0, %c0, %c0], %[[CSTSCALE]] {{.*}} : tensor<5x7x1xf8E8M0FNU>, vector<5x7x1xf8E8M0FNU>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg4[%c0, %c0, %c0], %[[CSTF32]] {{.*}} : tensor<3x7x4xf32>, vector<3x7x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[LHS_SCALE]], %[[RHS]], %[[RHS_SCALE]]) outs(%[[ACC]])
//  CHECK-SAME: : vector<3x5x1x32xf4E2M1FN>, vector<3x5x1xf8E8M0FNU>, vector<5x1x7x32xf8E4M3FN>, vector<5x7x1xf8E8M0FNU> into vector<3x7x4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg4[%c0, %c0, %c0] {{.*}} : vector<3x7x4xf32>, tensor<3x7x4xf32>
