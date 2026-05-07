// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-1d-vector-ops))" \
// RUN:   --split-input-file %s | FileCheck %s

// transfer_read/write on memref with in_bounds are lowered to vector.load/store.
// CHECK-LABEL: func @transfer_to_load_store(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:    vector.store %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_to_load_store(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// Scalar-sized vector (vector<1xf32>) on dynamic memref.
// CHECK-LABEL: func @transfer_scalar(
// CHECK-SAME:    %[[MEM:.*]]: memref<?x?xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<1xf32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<?x?xf32>, vector<1xf32>
// CHECK-NEXT:    return %[[RES]] : vector<1xf32>
func.func @transfer_scalar(%mem : memref<?x?xf32>, %idx : index) -> vector<1xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<?x?xf32>, vector<1xf32>
  return %res : vector<1xf32>
}

// -----

// Non-default but unit-stride layout still lowers.
// CHECK-LABEL: func @transfer_nondefault_layout(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32, #{{.*}}>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32, #{{.*}}>, vector<4xf32>
// CHECK-NEXT:    vector.store %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32, #{{.*}}>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>

#layout = affine_map<(d0, d1) -> (d0*16 + d1)>
func.func @transfer_nondefault_layout(%mem : memref<8x8xf32, #layout>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xf32, #layout>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32, #layout>
  return %res : vector<4xf32>
}

// -----

// Out-of-bounds (no in_bounds attr) should NOT be lowered.
// CHECK-LABEL: func @transfer_not_inbounds(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:    vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : vector<4xf32>, memref<8x8xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_not_inbounds(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// Non-identity permutation map should NOT be lowered.
// CHECK-LABEL: func @transfer_perm_map(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:    %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {in_bounds = [true], permutation_map = #{{.*}}} : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_perm_map(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// Masked rank-1 transfer_read/write lower to maskedload/maskedstore.
// CHECK-LABEL: func @transfer_masked(
// CHECK-SAME:    %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:    %[[IDX:.*]]: index,
// CHECK-SAME:    %[[MASK:.*]]: vector<4xi1>) -> vector<4xf32> {
// CHECK-NEXT:    %[[FILL:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-NEXT:    %[[RES:.*]] = vector.maskedload %[[MEM]][%[[IDX]], %[[IDX]]], %[[MASK]], %[[FILL]] : memref<8x8xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK-NEXT:    vector.maskedstore %[[MEM]][%[[IDX]], %[[IDX]]], %[[MASK]], %[[RES]] : memref<8x8xf32>, vector<4xi1>, vector<4xf32>
// CHECK-NEXT:    return %[[RES]] : vector<4xf32>
func.func @transfer_masked(%mem : memref<8x8xf32>, %idx : index, %mask : vector<4xi1>) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0, %mask {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx], %mask {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// Tensor source should NOT be lowered (only memref is supported).
// CHECK-LABEL: func @transfer_tensor(
// CHECK:         vector.transfer_read
func.func @transfer_tensor(%src : tensor<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %src[%idx, %idx], %cf0 {in_bounds = [true]} : tensor<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// 1-D vector.multi_reduction is lowered to vector.reduction.
// CHECK-LABEL: func @one_dim_reduction
// CHECK-SAME:    %[[INPUT:.+]]: vector<8xf32>, %[[ACC:.+]]: f32
func.func @one_dim_reduction(%arg0: vector<8xf32>, %acc: f32) -> f32 {
  // CHECK: %[[RESULT:.+]] = vector.reduction <add>, %[[INPUT]], %[[ACC]] : vector<8xf32> into f32
  %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
  // CHECK: return %[[RESULT]]
  return %0 : f32
}
