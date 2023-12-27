// RUN: iree-dialects-opt --split-input-file --test-vector-ext-iterators %s | FileCheck %s

// CHECK: VECTORY:0, BATCHX:0, VECTORX:0, BATCHY:0,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:0, BATCHY:0,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:0, BATCHY:0,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:0, BATCHY:0,
// CHECK: VECTORY:0, BATCHX:0, VECTORX:1, BATCHY:0,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:1, BATCHY:0,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:1, BATCHY:0,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:1, BATCHY:0,
// CHECK: VECTORY:0, BATCHX:0, VECTORX:0, BATCHY:1,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:0, BATCHY:1,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:0, BATCHY:1,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:0, BATCHY:1,
// CHECK: VECTORY:0, BATCHX:0, VECTORX:1, BATCHY:1,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:1, BATCHY:1,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:1, BATCHY:1,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:1, BATCHY:1,
#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [2, 1, 2]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [2, 1, 2]>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
func.func @iterator_test(%lhs: memref<4x4xf16>) -> vector<4x4xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true], __test_iterator_layout__ = #layout1} : memref<4x4xf16>, vector<4x4xf16>
  return %result : vector<4x4xf16>
}

// -----

// CHECK: VECTORY:0, BATCHX:0, VECTORX:0, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:0, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:0, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:0, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:0, BATCHX:0, VECTORX:1, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:1, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:1, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:1, BATCHY:0, VECTORZ:0,
// CHECK: VECTORY:0, BATCHX:0, VECTORX:0, BATCHY:1, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:0, BATCHY:1, VECTORZ:0,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:0, BATCHY:1, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:0, BATCHY:1, VECTORZ:0,
// CHECK: VECTORY:0, BATCHX:0, VECTORX:1, BATCHY:1, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:0, VECTORX:1, BATCHY:1, VECTORZ:0,
// CHECK: VECTORY:0, BATCHX:1, VECTORX:1, BATCHY:1, VECTORZ:0,
// CHECK: VECTORY:1, BATCHX:1, VECTORX:1, BATCHY:1, VECTORZ:0,
#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [2, 1, 2]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [2, 1, 2]>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
func.func @frozen_iterator_test(%lhs: memref<4x4xf16>) -> vector<4x4xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true], __test_frozen_iterator_layout__ = #layout1} : memref<4x4xf16>, vector<4x4xf16>
  return %result : vector<4x4xf16>
}
