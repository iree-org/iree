// RUN: iree-dialects-opt --split-input-file %s | FileCheck %s

#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [2, 4, 4]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout2 = #iree_vector_ext.layout<#col_layout1, #row_layout1>
func.func @specify_layout(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  %2 = iree_vector_ext.to_layout %result to #layout2 : vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// CHECK-DAG: #[[LAYOUT0:.+]] = #iree_vector_ext.layout<<[ BATCHY,  LANEY,  VECTORX], [4, 2, 4]>, <[ BATCHX,  LANEX,  VECTORY], [2, 4, 4]>>
// CHECK-LABEL: func.func @specify_layout
// CHECK:      iree_vector_ext.to_layout {{.*}} to #[[LAYOUT0]]

// -----

#nested_0 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [2, 4],
  outers_per_batch = [4, 1],
  threads_per_outer = [4, 2],
  elements_per_thread = [1, 4],

  subgroup_strides = [0, 0],
  thread_strides   = [1, 4]
>

#nested_1 = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [4, 2],
  outers_per_batch = [1, 4],
  threads_per_outer = [2, 4],
  elements_per_thread = [4, 1],

  subgroup_strides = [0, 0],
  thread_strides = [8, 2]
>

func.func @specify_nested(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {
    in_bounds = [true, true],
    layout0 = #nested_0,
    layout1 = #nested_1
  } : memref<32x32xf16>, vector<32x32xf16>
  return %result : vector<32x32xf16>
}

// CHECK: #[[LAYOUT0:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1],
// CHECK-SAME: batches_per_subgroup = [2, 4],
// CHECK-SAME: outers_per_batch = [4, 1],
// CHECK-SAME: threads_per_outer = [4, 2],
// CHECK-SAME: elements_per_thread = [1, 4],
// CHECK-SAME: subgroup_strides = [0, 0],
// CHECK-SAME: thread_strides = [1, 4]>

// CHECK: #[[LAYOUT1:.+]] = #iree_vector_ext.nested_layout<
// CHECK-SAME: subgroups_per_workgroup = [1, 1],
// CHECK-SAME: batches_per_subgroup = [4, 2],
// CHECK-SAME: outers_per_batch = [1, 4],
// CHECK-SAME: threads_per_outer = [2, 4],
// CHECK-SAME: elements_per_thread = [4, 1],
// CHECK-SAME: subgroup_strides = [0, 0],
// CHECK-SAME: thread_strides = [8, 2]>

// CHECK-LABEL: func.func @specify_nested
// CHECK:      vector.transfer_read
// CHECK-SAME:         layout0 = #[[LAYOUT0]]
// CHECK-SAME:         layout1 = #[[LAYOUT1]]

// -----

func.func @to_simd_op(%simt: vector<4x4x4xf16>) -> vector<64x64xf16> {
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf16> -> vector<64x64xf16>
  func.return %simd : vector<64x64xf16>
}
// CHECK-LABEL: func.func @to_simd_op
// CHECK:      iree_vector_ext.to_simd

// -----

func.func @to_simt_op(%simd: vector<64x64xf32>) -> vector<4x4x4xf32> {
  %simt = iree_vector_ext.to_simd %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  func.return %simt : vector<4x4x4xf32>
}
// CHECK-LABEL: func.func @to_simt_op
// CHECK:      iree_vector_ext.to_simd
