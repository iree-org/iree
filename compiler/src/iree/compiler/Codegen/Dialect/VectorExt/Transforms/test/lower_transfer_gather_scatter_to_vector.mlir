// RUN: iree-opt %s -pass-pipeline='builtin.module(func.func(iree-vector-ext-lower-transfer-gather-scatter-to-vector))' --split-input-file --mlir-print-local-scope | FileCheck %s

#map = affine_map<(d0)[s0] -> (0, 0, s0)>
#map1 = affine_map<(d0)[s0] -> (d0)>
module {
  func.func @lower_transfer_gather_to_vector_gather(%arg0: tensor<1x1x31xf32>, %arg1: tensor<1x1x1x1x16xf32>) -> tensor<1x1x1x1x16xf32> {
    %0 = ub.poison : vector<1x16xf32>
    %1 = ub.poison : vector<1x1x16xf32>
    %2 = ub.poison : vector<1x1x1x16xf32>
    %3 = ub.poison : vector<1x1x1x1x16xf32>
    %cst = arith.constant dense<2> : vector<16xindex>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %4 = vector.step : vector<16xindex>
    %5 = arith.muli %4, %cst : vector<16xindex>
    %6 = iree_vector_ext.transfer_gather %arg0[%c0, %c0, %c0] [%5 : vector<16xindex>], %cst_0 {indexing_maps = [#map, #map1]} : tensor<1x1x31xf32>, vector<16xf32>
    %7 = vector.insert_strided_slice %6, %0 {offsets = [0, 0], strides = [1]} : vector<16xf32> into vector<1x16xf32>
    %8 = vector.insert_strided_slice %7, %1 {offsets = [0, 0, 0], strides = [1, 1]} : vector<1x16xf32> into vector<1x1x16xf32>
    %9 = vector.insert_strided_slice %8, %2 {offsets = [0, 0, 0, 0], strides = [1, 1, 1]} : vector<1x1x16xf32> into vector<1x1x1x16xf32>
    %10 = vector.insert_strided_slice %9, %3 {offsets = [0, 0, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x1x16xf32> into vector<1x1x1x1x16xf32>
    %11 = vector.transfer_write %10, %arg1[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x16xf32>, tensor<1x1x1x1x16xf32>
    return %11 : tensor<1x1x1x1x16xf32>
  }
}

// CHECK-LABEL: func.func @lower_transfer_gather_to_vector_gather
// CHECK-SAME:  %[[SRC:.+]]: tensor<1x1x31xf32>
// CHECK-DAG:     %[[PASS_THRU:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:     %[[MASK:.+]] = arith.constant dense<true> : vector<16xi1>
// CHECK-DAG:     %[[STRIDE:.+]] = arith.constant dense<2> : vector<16xindex>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[STEP:.+]] = vector.step : vector<16xindex>
// CHECK:         %[[INDICES:.+]] = arith.muli %[[STEP]], %[[STRIDE]] : vector<16xindex>
// CHECK:         %[[GATHER:.+]] = vector.gather %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] [%[[INDICES]]], %[[MASK]], %[[PASS_THRU]]
// CHECK-SAME:      : tensor<1x1x31xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>

// -----

func.func @lower_gather_nontrivial_leading_dims(
    %src: tensor<4x16xf32>, %idx: vector<8xindex>) -> vector<8xf32> {
  %pad = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %out = iree_vector_ext.transfer_gather %src[%c2, %c0]
    [%idx : vector<8xindex>], %pad {
      indexing_maps = [affine_map<(d0)[s0] -> (0, s0)>,
                       affine_map<(d0)[s0] -> (d0)>]
    } : tensor<4x16xf32>, vector<8xf32>
  return %out : vector<8xf32>
}
// CHECK-LABEL: @lower_gather_nontrivial_leading_dims
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK:     vector.gather %{{.+}}[%[[C2]], %[[C0]]]

// -----

func.func @lower_masked_gather(
    %src: tensor<8x16xf16>, %idx: vector<16xindex>,
    %mask: vector<16xi1>) -> vector<16xf16> {
  %pad = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %src[%c0, %c0]
    [%idx : vector<16xindex>], %pad, %mask {
      indexing_maps = [affine_map<(d0)[s0] -> (0, s0)>,
                       affine_map<(d0)[s0] -> (d0)>,
                       affine_map<(d0)[s0] -> (d0)>]
    } : tensor<8x16xf16>, vector<16xf16>, vector<16xi1>
  return %out : vector<16xf16>
}
// CHECK-LABEL: @lower_masked_gather
// CHECK:       vector.gather
// CHECK-NOT:   iree_vector_ext.transfer_gather

// -----

func.func @negative_lower_gather_multiple_index_vecs(
    %src: tensor<8x16xf16>,
    %i0: vector<8xindex>, %i1: vector<16xindex>) -> vector<8x16xf16> {
  %pad = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %src[%c0, %c0]
    [%i0, %i1 : vector<8xindex>, vector<16xindex>], %pad {
      indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d0)>,
                       affine_map<(d0, d1)[s0, s1] -> (d1)>]
    } : tensor<8x16xf16>, vector<8x16xf16>
  return %out : vector<8x16xf16>
}
// CHECK-LABEL: @negative_lower_gather_multiple_index_vecs
// CHECK: iree_vector_ext.transfer_gather
// CHECK-NOT: vector.gather

// -----

func.func @negative_lower_gather_scalar_index(
    %src: memref<8x16xf16>, %idx: index) -> vector<8xf16> {
  %pad = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %src[%c0, %c0]
    [%idx : index], %pad {
      indexing_maps = [affine_map<(d0)[s0] -> (0, s0)>,
                       affine_map<(d0)[s0] -> ()>]
    } : memref<8x16xf16>, vector<8xf16>
  return %out : vector<8xf16>
}
// CHECK-LABEL: @negative_lower_gather_scalar_index
// CHECK: iree_vector_ext.transfer_gather
// CHECK-NOT: vector.gather

// -----

func.func @negative_lower_gather_symbol_in_leading_dim(
    %src: tensor<8x16xf16>, %idx: vector<8xindex>) -> vector<8x16xf16> {
  %pad = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %src[%c0, %c0]
    [%idx : vector<8xindex>], %pad {
      indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d0)>]
    } : tensor<8x16xf16>, vector<8x16xf16>
  return %out : vector<8x16xf16>
}
// CHECK-LABEL: @negative_lower_gather_symbol_in_leading_dim
// CHECK: iree_vector_ext.transfer_gather
// CHECK-NOT: vector.gather

// -----

func.func @negative_lower_gather_nonconstant_leading_dim(
    %src: tensor<16x16xf16>, %idx: vector<16xindex>) -> vector<16xf16> {
  %pad = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %src[%c0, %c0]
    [%idx : vector<16xindex>], %pad {
      indexing_maps = [affine_map<(d0)[s0] -> (d0, s0)>,
                       affine_map<(d0)[s0] -> (d0)>]
    } : tensor<16x16xf16>, vector<16xf16>
  return %out : vector<16xf16>
}
// CHECK-LABEL: @negative_lower_gather_nonconstant_leading_dim
// CHECK: iree_vector_ext.transfer_gather
// CHECK-NOT: vector.gather

// -----

func.func @lower_transfer_scatter_to_vector_scatter(
    %src: vector<16xf32>, %dest: tensor<1x1x32xf32>,
    %idx: vector<16xindex>) -> tensor<1x1x32xf32> {
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_scatter %src into %dest[%c0, %c0, %c0]
    [%idx : vector<16xindex>] {
      indexing_maps = [affine_map<(d0)[s0] -> (0, 0, s0)>,
                       affine_map<(d0)[s0] -> (d0)>]
    } : vector<16xf32>, tensor<1x1x32xf32> -> tensor<1x1x32xf32>
  return %out : tensor<1x1x32xf32>
}
// CHECK-LABEL: @lower_transfer_scatter_to_vector_scatter
// CHECK-DAG:     %[[MASK:.+]] = arith.constant dense<true> : vector<16xi1>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         vector.scatter %{{.+}}[%[[C0]], %[[C0]], %[[C0]]]
// CHECK-NOT:     iree_vector_ext.transfer_scatter

// -----

func.func @lower_scatter_nontrivial_leading_dims(
    %src: vector<8xf32>, %dest: tensor<4x16xf32>,
    %idx: vector<8xindex>) -> tensor<4x16xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %out = iree_vector_ext.transfer_scatter %src into %dest[%c2, %c0]
    [%idx : vector<8xindex>] {
      indexing_maps = [affine_map<(d0)[s0] -> (0, s0)>,
                       affine_map<(d0)[s0] -> (d0)>]
    } : vector<8xf32>, tensor<4x16xf32> -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}
// CHECK-LABEL: @lower_scatter_nontrivial_leading_dims
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK:     vector.scatter %{{.+}}[%[[C2]], %[[C0]]]

// -----

func.func @lower_masked_scatter(
    %src: vector<16xf16>, %dest: tensor<8x16xf16>,
    %idx: vector<16xindex>, %mask: vector<16xi1>) -> tensor<8x16xf16> {
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_scatter %src into %dest[%c0, %c0]
    [%idx : vector<16xindex>], %mask {
      indexing_maps = [affine_map<(d0)[s0] -> (0, s0)>,
                       affine_map<(d0)[s0] -> (d0)>,
                       affine_map<(d0)[s0] -> (d0)>]
    } : vector<16xf16>, tensor<8x16xf16>, vector<16xi1> -> tensor<8x16xf16>
  return %out : tensor<8x16xf16>
}
// CHECK-LABEL: @lower_masked_scatter
// CHECK:       vector.scatter
// CHECK-NOT:   iree_vector_ext.transfer_scatter

// -----

func.func @negative_lower_scatter_multiple_index_vecs(
    %src: vector<8x16xf16>, %dest: tensor<8x16xf16>,
    %i0: vector<8xindex>, %i1: vector<16xindex>) -> tensor<8x16xf16> {
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_scatter %src into %dest[%c0, %c0]
    [%i0, %i1 : vector<8xindex>, vector<16xindex>] {
      indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d0)>,
                       affine_map<(d0, d1)[s0, s1] -> (d1)>]
    } : vector<8x16xf16>, tensor<8x16xf16> -> tensor<8x16xf16>
  return %out : tensor<8x16xf16>
}
// CHECK-LABEL: @negative_lower_scatter_multiple_index_vecs
// CHECK: iree_vector_ext.transfer_scatter
// CHECK-NOT: vector.scatter

// -----

func.func @negative_lower_scatter_scalar_index(
    %src: vector<8xf16>, %dest: tensor<8x16xf16>, %idx: index) -> tensor<8x16xf16> {
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_scatter %src into %dest[%c0, %c0]
    [%idx : index] {
      indexing_maps = [affine_map<(d0)[s0] -> (0, s0)>,
                       affine_map<(d0)[s0] -> ()>]
    } : vector<8xf16>, tensor<8x16xf16> -> tensor<8x16xf16>
  return %out : tensor<8x16xf16>
}
// CHECK-LABEL: @negative_lower_scatter_scalar_index
// CHECK: iree_vector_ext.transfer_scatter
// CHECK-NOT: vector.scatter

// -----

func.func @negative_lower_scatter_symbol_in_leading_dim(
    %src: vector<8x16xf16>, %dest: tensor<8x16xf16>,
    %idx: vector<8xindex>) -> tensor<8x16xf16> {
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_scatter %src into %dest[%c0, %c0]
    [%idx : vector<8xindex>] {
      indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d0)>]
    } : vector<8x16xf16>, tensor<8x16xf16> -> tensor<8x16xf16>
  return %out : tensor<8x16xf16>
}
// CHECK-LABEL: @negative_lower_scatter_symbol_in_leading_dim
// CHECK: iree_vector_ext.transfer_scatter
// CHECK-NOT: vector.scatter

// -----

func.func @negative_lower_scatter_nonconstant_leading_dim(
    %src: vector<16xf16>, %dest: tensor<16x16xf16>,
    %idx: vector<16xindex>) -> tensor<16x16xf16> {
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_scatter %src into %dest[%c0, %c0]
    [%idx : vector<16xindex>] {
      indexing_maps = [affine_map<(d0)[s0] -> (d0, s0)>,
                       affine_map<(d0)[s0] -> (d0)>]
    } : vector<16xf16>, tensor<16x16xf16> -> tensor<16x16xf16>
  return %out : tensor<16x16xf16>
}
// CHECK-LABEL: @negative_lower_scatter_nonconstant_leading_dim
// CHECK: iree_vector_ext.transfer_scatter
// CHECK-NOT: vector.scatter
