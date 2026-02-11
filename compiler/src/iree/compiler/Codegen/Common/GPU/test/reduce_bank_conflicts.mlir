// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-reduce-bank-conflicts{padding-bits=64}))" | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0 * 2048 + d1 * 64 + d2)>

// CHECK-LABEL: func.func @pad_alloc
// CHECK:         %[[A:.*]] = memref.alloc() : memref<4x32x66xf32, #gpu.address_space<workgroup>>
// CHECK:         %[[S1:.*]] = memref.subview %[[A]][0, 0, 0] [4, 32, 64] [1, 1, 1] :
// CHECK-SAME:      memref<4x32x66xf32, #gpu.address_space<workgroup>> to memref<4x32x64xf32, strided<[2112, 66, 1]>, #gpu.address_space<workgroup>>
// CHECK:         %[[S2:.*]] = memref.subview %[[S1]][0, 0, 0] [1, 32, 64] [1, 1, 1] :
// CHECK-SAME:      memref<4x32x64xf32, strided<[2112, 66, 1]>, #gpu.address_space<workgroup>> to memref<1x32x64xf32, strided<[2112, 66, 1]>, #gpu.address_space<workgroup>>
// CHECK:           vector.transfer_write %{{.*}}, %[[S2]][%{{.*}}, %{{.*}}, %{{.*}}] {in_bounds = [true]} :
// CHECK-SAME:      vector<4xf32>, memref<1x32x64xf32, strided<[2112, 66, 1]>, #gpu.address_space<workgroup>
func.func @pad_alloc(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %1 = memref.subview %0[0, 0, 0] [1, 32, 64] [1, 1, 1] :
    memref<4x32x64xf32, #gpu.address_space<workgroup>> to memref<1x32x64xf32, #map, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %2 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} :
    memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %2, %1[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<1x32x64xf32, #map, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @pad_alloc_expand_shape
// CHECK:         %[[A:.*]] = memref.alloc() : memref<4x32x66xf32, #gpu.address_space<workgroup>>
// CHECK:         %[[S1:.*]] = memref.subview %[[A]][0, 0, 0] [4, 32, 64] [1, 1, 1] :
// CHECK-SAME:      memref<4x32x66xf32, #gpu.address_space<workgroup>> to memref<4x32x64xf32, strided<[2112, 66, 1]>, #gpu.address_space<workgroup>>
// CHECK:         %[[E:.*]] = memref.expand_shape %[[S1]] {{\[}}[0], [1, 2], [3, 4]] output_shape [4, 2, 16, 8, 8]
// CHECK-SAME:      memref<4x32x64xf32, strided<[2112, 66, 1]>, #gpu.address_space<workgroup>> into
// CHECK-SAME:      memref<4x2x16x8x8xf32, strided<[2112, 1056, 66, 8, 1]>, #gpu.address_space<workgroup>>
// CHECK:           vector.transfer_write %{{.*}}, %[[E]][%{{.*}}, %{{.*}}, %{{.*}}] {in_bounds = [true]} :
// CHECK-SAME:      vector<4xf32>, memref<4x2x16x8x8xf32, strided<[2112, 1056, 66, 8, 1]>, #gpu.address_space<workgroup>
func.func @pad_alloc_expand_shape(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %1 = memref.expand_shape %0 [[0], [1, 2], [3, 4]] output_shape [4, 2, 16, 8, 8]
    : memref<4x32x64xf32, #gpu.address_space<workgroup>> into memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %3 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} :
    memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %3, %1[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
  return
}

// -----
// Verify that collapse_shape prevents padding.
// CHECK-LABEL: func.func @no_pad_alloc_collapse_shape
// CHECK:         memref.alloc() : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     memref.subview
func.func @no_pad_alloc_collapse_shape(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
  %1 = memref.collapse_shape %0 [[0], [1, 2], [3, 4]]
    : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>> into memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %3 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} :
    memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %3, %1[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<4x32x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @no_pad_alloc_collapse_shape_throughsubview
// CHECK:         %[[A:.*]] = memref.alloc() : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
// CHECK:         %[[S:.*]] = memref.subview %[[A]][0, 0, 0, 0, 0] [4, 2, 16, 8, 8] [1, 1, 1, 1, 1] :
// CHECK-SAME:      memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>> to
// CHECK-SAME:      memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
// CHECK:         %[[C:.*]] = memref.collapse_shape %[[S]] {{\[}}[0], [1, 2], [3, 4]]
// CHECK-SAME:      memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>> into
// CHECK-SAME:      memref<4x32x64xf32, #gpu.address_space<workgroup>>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[VEC_READ:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true]} :
// CHECK-SAME:      memref<1024x1024xf32>, vector<4xf32>
// CHECK:         vector.transfer_write %[[VEC_READ]], %[[C]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true]} :
// CHECK-SAME:      vector<4xf32>, memref<4x32x64xf32, #gpu.address_space<workgroup>>
func.func @no_pad_alloc_collapse_shape_throughsubview(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
  %subview = memref.subview %0[0, 0, 0, 0, 0] [4, 2, 16, 8, 8] [1, 1, 1, 1, 1]
   : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>> to memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
  %1 = memref.collapse_shape %subview [[0], [1, 2], [3, 4]]
    : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>> into memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %3 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} :
    memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %3, %1[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<4x32x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @pad_alloc_collapse_outer_shape
// CHECK-SAME:    %[[V:.*]]: vector<4xf32>
// CHECK:         %[[A:.*]] = memref.alloc() : memref<2x16x8x10xf32, #gpu.address_space<workgroup>>
// CHECK:         %[[S0:.*]] = memref.subview %[[A]][0, 0, 0, 0] [2, 16, 8, 8] [1, 1, 1, 1] :
// CHECK-SAME:      memref<2x16x8x10xf32, #gpu.address_space<workgroup>> to
// CHECK-SAME:      memref<2x16x8x8xf32, strided<[1280, 80, 10, 1]>, #gpu.address_space<workgroup>>
// CHECK:         %[[S1:.*]] = memref.subview %[[S0]][0, 0, 0, 0] [2, 16, 8, 8] [1, 1, 1, 1] :
// CHECK-SAME:      memref<2x16x8x8xf32, strided<[1280, 80, 10, 1]>, #gpu.address_space<workgroup>> to
// CHECK-SAME:      memref<2x16x8x8xf32, strided<[1280, 80, 10, 1]>, #gpu.address_space<workgroup>>
// CHECK:         %[[C:.*]] = memref.collapse_shape %[[S1]] {{\[}}[0, 1], [2], [3]]
// CHECK-SAME:      memref<2x16x8x8xf32, strided<[1280, 80, 10, 1]>, #gpu.address_space<workgroup>> into
// CHECK-SAME:      memref<32x8x8xf32, strided<[80, 10, 1]>, #gpu.address_space<workgroup>>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         vector.transfer_write %[[V]], %[[C]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true]} :
// CHECK-SAME:      vector<4xf32>, memref<32x8x8xf32, strided<[80, 10, 1]>, #gpu.address_space<workgroup>>
func.func @pad_alloc_collapse_outer_shape(%v: vector<4xf32>) {
  %0 = memref.alloc() : memref<2x16x8x8xf32, #gpu.address_space<workgroup>>
  %subview = memref.subview %0[0, 0, 0, 0] [2, 16, 8, 8] [1, 1, 1, 1]
   : memref<2x16x8x8xf32, #gpu.address_space<workgroup>> to memref<2x16x8x8xf32, #gpu.address_space<workgroup>>
  %1 = memref.collapse_shape %subview [[0, 1], [2], [3]]
    : memref<2x16x8x8xf32, #gpu.address_space<workgroup>> into memref<32x8x8xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  vector.transfer_write %v, %1[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<32x8x8xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @pad_alloc_negative
// CHECK:         memref.alloc(%{{.*}}) : memref<?x32x64xf32, #gpu.address_space<workgroup>
func.func @pad_alloc_negative(%a: memref<1024x1024xf32>, %i: index, %v: vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %0 = memref.alloc(%i) : memref<?x32x64xf32, #gpu.address_space<workgroup>>
  vector.transfer_write %v, %0[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<?x32x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @pad_alloc_rank_zero
// CHECK:         memref.alloc() : memref<f32, #gpu.address_space<workgroup>
func.func @pad_alloc_rank_zero() {
  %cst = arith.constant dense<-3.40282347E+38> : vector<f32>
  %0 = memref.alloc() : memref<f32, #gpu.address_space<workgroup>>
  vector.transfer_write %cst, %0[] : vector<f32>, memref<f32, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @no_padding_when_close_to_limit
// CHECK:         memref.alloc() : memref<4x32x127xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     memref.subview
func.func @no_padding_when_close_to_limit() {
  %0 = memref.alloc() : memref<4x32x127xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @no_padding_if_at_limit
// CHECK:         memref.alloc() : memref<4x32x128xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     memref.subview
func.func @no_padding_if_at_limit() {
  %0 = memref.alloc() : memref<4x32x128xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// CHECK-LABEL: func.func @pad_if_below_limit
// CHECK:         %[[A:.*]] = memref.alloc() : memref<4x32x128xf32, #gpu.address_space<workgroup>>
// CHECK:         memref.subview %[[A]][0, 0, 0] [4, 32, 126] [1, 1, 1] :
// CHECK-SAME:      memref<4x32x128xf32, #gpu.address_space<workgroup>> to
// CHECK-SAME:      memref<4x32x126xf32, strided<[4096, 128, 1]>, #gpu.address_space<workgroup>>
func.func @pad_if_below_limit() {
  %0 = memref.alloc() : memref<4x32x126xf32, #gpu.address_space<workgroup>>
  return
}

// -----
//
// Tests below exercise hint-based padding (BankConflictPaddingHintOp).
// Hints override the fallback padding-bits value for the hinted alloc.
//

// Hint with 32-bit padding (1 float): alloc padded to 65.
// CHECK-LABEL: func.func @pad_with_hint_32bit
// CHECK:         memref.alloc() : memref<4x32x65xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     iree_gpu.bank_conflict_padding_hint
func.func @pad_with_hint_32bit() {
  %0 = memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %hinted = iree_gpu.bank_conflict_padding_hint %0 [padding_bits = 32]
      : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// Hint with 128-bit padding (4 floats): alloc padded to 68.
// CHECK-LABEL: func.func @pad_with_hint_128bit
// CHECK:         memref.alloc() : memref<4x32x68xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     iree_gpu.bank_conflict_padding_hint
func.func @pad_with_hint_128bit() {
  %0 = memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %hinted = iree_gpu.bank_conflict_padding_hint %0 [padding_bits = 128]
      : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// Mixed: hinted alloc gets hint padding, unhinted alloc gets fallback.
// CHECK-LABEL: func.func @mixed_hinted_and_unhinted
// CHECK-DAG:     memref.alloc() : memref<2x16x65xf32, #gpu.address_space<workgroup>>
// CHECK-DAG:     memref.alloc() : memref<2x16x66xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     iree_gpu.bank_conflict_padding_hint
func.func @mixed_hinted_and_unhinted() {
  %0 = memref.alloc() : memref<2x16x64xf32, #gpu.address_space<workgroup>>
  %hinted = iree_gpu.bank_conflict_padding_hint %0 [padding_bits = 32]
      : memref<2x16x64xf32, #gpu.address_space<workgroup>>
  %1 = memref.alloc() : memref<2x16x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// Hint with zero padding results in no padding for that alloc.
// CHECK-LABEL: func.func @hint_zero_padding
// CHECK:         memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     memref.subview
// CHECK-NOT:     iree_gpu.bank_conflict_padding_hint
func.func @hint_zero_padding() {
  %0 = memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %hinted = iree_gpu.bank_conflict_padding_hint %0 [padding_bits = 0]
      : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// Multiple hints on the same alloc result in no padding.
// CHECK-LABEL: func.func @conflicting_hints_no_padding
// CHECK:         memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     memref.subview
// CHECK-NOT:     iree_gpu.bank_conflict_padding_hint
func.func @conflicting_hints_no_padding() {
  %0 = memref.alloc() : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %hint1 = iree_gpu.bank_conflict_padding_hint %0 [padding_bits = 32]
      : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %hint2 = iree_gpu.bank_conflict_padding_hint %0 [padding_bits = 64]
      : memref<4x32x64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// Collapse_shape prevents padding even with a hint.
// CHECK-LABEL: func.func @no_pad_collapse_shape_with_hint
// CHECK:         memref.alloc() : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
// CHECK-NOT:     memref.subview
func.func @no_pad_collapse_shape_with_hint(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
  %hinted = iree_gpu.bank_conflict_padding_hint %0 [padding_bits = 64]
      : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>>
  %1 = memref.collapse_shape %hinted [[0], [1, 2], [3, 4]]
    : memref<4x2x16x8x8xf32, #gpu.address_space<workgroup>> into memref<4x32x64xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %3 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} :
    memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %3, %1[%c0, %c0, %c0] {in_bounds = [true]} :
    vector<4xf32>, memref<4x32x64xf32, #gpu.address_space<workgroup>>
  return
}
