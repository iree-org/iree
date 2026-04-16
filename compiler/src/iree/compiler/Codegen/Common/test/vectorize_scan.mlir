// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false}))" --split-input-file %s | FileCheck %s

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
