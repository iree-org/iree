// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false}))" --split-input-file %s | FileCheck %s

// Tests for vectorization without masking enabled. Vector sizes are not
// computed by the pass; each op's VectorizableOpInterface implementation
// determines sizes from static tensor shapes.

func.func @matmul(%lhs: tensor<3x4xf16>, %rhs: tensor<4x5xf16>, %acc: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<3x4xf16>, tensor<4x5xf16>) outs(%acc: tensor<3x5xf32>) -> tensor<3x5xf32>
  return %result: tensor<3x5xf32>
}
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]
// CHECK:         %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[OUT_VEC:.+]] = vector.transfer_read %[[OUT]]
// CHECK:         %[[EXT_LHS:.+]] = arith.extf %[[LHS_VEC]]
// CHECK:         %[[EXT_RHS:.+]] = arith.extf %[[RHS_VEC]]
// CHECK:         %[[RES:.+]] = vector.contract {{.+}} %[[EXT_LHS]], %[[EXT_RHS]], %[[OUT_VEC]]

// -----

#map = affine_map<(d0) -> (-d0 + 13, 2)>
#map1 = affine_map<(d0) -> (-d0 + 51, 4)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0, d1) -> (d1 * -2 + 101, d0 * 2)>
#map4 = affine_map<(d0) -> (d0 * 16)>
#map5 = affine_map<(d0, d1) -> (d1 * -16 + 201, d0 * 16)>
func.func @single_static_pack_infer_vector_size(%arg0: tensor<101x201xi8>, %arg1: tensor<13x51x16x2xi8>) -> tensor<13x51x16x2xi8> {
  %c4 = arith.constant 4 : index
  %c51 = arith.constant 51 : index
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c13 = arith.constant 13 : index
  %c2 = arith.constant 2 : index
  %0 = scf.for %arg2 = %c0 to %c13 step %c2 iter_args(%arg3 = %arg1) -> (tensor<13x51x16x2xi8>) {
    %1 = scf.for %arg4 = %c0 to %c51 step %c4 iter_args(%arg5 = %arg3) -> (tensor<13x51x16x2xi8>) {
      %2 = affine.min #map(%arg2)
      %3 = affine.min #map1(%arg4)
      %4 = affine.apply #map2(%arg4)
      %5 = affine.min #map3(%3, %arg4)
      %6 = affine.apply #map4(%arg2)
      %7 = affine.min #map5(%2, %arg2)
      %extracted_slice = tensor.extract_slice %arg0[%4, %6] [%5, %7] [1, 1] : tensor<101x201xi8> to tensor<?x?xi8>
      %extracted_slice_0 = tensor.extract_slice %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<13x51x16x2xi8> to tensor<?x?x16x2xi8>
      %pack = linalg.pack %extracted_slice padding_value(%c0_i8 : i8) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %extracted_slice_0 : tensor<?x?xi8> -> tensor<?x?x16x2xi8>
      %inserted_slice = tensor.insert_slice %pack into %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<?x?x16x2xi8> into tensor<13x51x16x2xi8>
      scf.yield %inserted_slice : tensor<13x51x16x2xi8>
    }
    scf.yield %1 : tensor<13x51x16x2xi8>
  }
  return %0 : tensor<13x51x16x2xi8>
}
// Direct linalg.pack vectorization is only available with masking.
// TODO: Support non-masking path.
// CHECK-LABEL: func.func @single_static_pack_infer_vector_size
// CHECK:         linalg.pack

// -----

// CHECK-LABEL: func.func @vectorize_scan_add_inclusive
func.func @vectorize_scan_add_inclusive(
    %input: tensor<8xf32>,
    %output: tensor<8xf32>,
    %accum: tensor<f32>) -> (tensor<8xf32>, tensor<f32>) {
  %0:2 = iree_linalg_ext.scan
      dimension(0) inclusive(true)
      ins(%input : tensor<8xf32>)
      outs(%output, %accum : tensor<8xf32>, tensor<f32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      iree_linalg_ext.yield %sum : f32
  } -> tensor<8xf32>, tensor<f32>
  return %0#0, %0#1 : tensor<8xf32>, tensor<f32>
}
// CHECK: %[[READ:.+]] = vector.transfer_read
// CHECK: %[[INIT:.+]] = vector.transfer_read
// CHECK: %[[DEST:.+]], %{{.+}} = vector.scan <add>, %[[READ]], %[[INIT]]
// CHECK-SAME: inclusive = true
// CHECK: vector.transfer_write %[[DEST]]
// CHECK: vector.transfer_write

// -----

// CHECK-LABEL: func.func @vectorize_scan_mul_exclusive
func.func @vectorize_scan_mul_exclusive(
    %input: tensor<16xi32>,
    %output: tensor<16xi32>,
    %accum: tensor<i32>) -> (tensor<16xi32>, tensor<i32>) {
  %0:2 = iree_linalg_ext.scan
      dimension(0) inclusive(false)
      ins(%input : tensor<16xi32>)
      outs(%output, %accum : tensor<16xi32>, tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %prod = arith.muli %arg0, %arg1 : i32
      iree_linalg_ext.yield %prod : i32
  } -> tensor<16xi32>, tensor<i32>
  return %0#0, %0#1 : tensor<16xi32>, tensor<i32>
}
// CHECK: vector.scan <mul>
// CHECK-SAME: inclusive = false

// -----

// CHECK-LABEL: func.func @vectorize_scan_2d
func.func @vectorize_scan_2d(
    %input: tensor<4x8xf32>,
    %output: tensor<4x8xf32>,
    %accum: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4xf32>) {
  %0:2 = iree_linalg_ext.scan
      dimension(1) inclusive(true)
      ins(%input : tensor<4x8xf32>)
      outs(%output, %accum : tensor<4x8xf32>, tensor<4xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      iree_linalg_ext.yield %sum : f32
  } -> tensor<4x8xf32>, tensor<4xf32>
  return %0#0, %0#1 : tensor<4x8xf32>, tensor<4xf32>
}
// CHECK: vector.scan <add>
// CHECK-SAME: reduction_dim = 1

// -----

// CHECK-LABEL: func.func @vectorize_scan_maxsi
func.func @vectorize_scan_maxsi(
    %input: tensor<8xi32>,
    %output: tensor<8xi32>,
    %accum: tensor<i32>) -> (tensor<8xi32>, tensor<i32>) {
  %0:2 = iree_linalg_ext.scan
      dimension(0) inclusive(true)
      ins(%input : tensor<8xi32>)
      outs(%output, %accum : tensor<8xi32>, tensor<i32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %max = arith.maxsi %arg0, %arg1 : i32
      iree_linalg_ext.yield %max : i32
  } -> tensor<8xi32>, tensor<i32>
  return %0#0, %0#1 : tensor<8xi32>, tensor<i32>
}
// CHECK: vector.scan <maxsi>
