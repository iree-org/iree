// RUN: iree-dialects-opt --split-input-file %s | FileCheck %s

#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [2, 4, 4]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
#layout2 = #iree_vector_ext.layout<#col_layout1, #row_layout1>
func.func @specify_layout(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  %2 = iree_vector_ext.layout_conflict_resolution %result {sourceLayout = #layout1, desiredLayout = #layout2} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// CHECK-DAG: #[[LAYOUT0:.+]] = #iree_vector_ext.layout<<[ BATCHY,  LANEY,  VECTORX], [4, 2, 4]>, <[ BATCHX,  LANEX,  VECTORY], [2, 4, 4]>>
// CHECK-DAG: #[[LAYOUT1:.+]] = #iree_vector_ext.layout<<[ BATCHX,  LANEX,  VECTORY], [2, 4, 4]>, <[ BATCHY,  LANEY,  VECTORX], [4, 2, 4]>>
// CHECK-LABEL: func.func @specify_layout
// CHECK:      iree_vector_ext.layout_conflict_resolution
// CHECK-SAME:         desiredLayout = #[[LAYOUT0]]
// CHECK-SAME:         sourceLayout = #[[LAYOUT1]]

// -----

#nested_1 = #iree_vector_ext.nested<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [2, 4],
  duplicates_per_batch = [4, 1],
  threads_per_duplicate = [4, 2],
  elements_per_thread = [1, 4],

  subgroup_order = [0, 1],
  batch_order = [0, 1],
  duplicate_order = [1, 0],
  thread_order = [0, 1],
  element_order = [0, 1]
>

#nested_2 = #iree_vector_ext.nested<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [4, 2],
  duplicates_per_batch = [1, 4],
  threads_per_duplicate = [2, 4],
  elements_per_thread = [4, 1],

  subgroup_order = [1, 0],
  batch_order = [1, 0],
  duplicate_order = [0, 1],
  thread_order = [1, 0],
  element_order = [1, 0]
>

func.func @specify_nested(%lhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  %2 = iree_vector_ext.layout_conflict_resolution %result {sourceLayout = #nested_1, desiredLayout = #nested_2} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// CHECK-DAG: #[[LAYOUT0:.+]] = #iree_vector_ext.nested<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [4, 2], duplicates_per_batch = [1, 4], threads_per_duplicate = [2, 4], elements_per_thread = [4, 1], subgroup_order = [1, 0], batch_order = [1, 0], duplicate_order = [0, 1], thread_order = [1, 0], element_order = [1, 0]>
// CHECK-DAG: #[[LAYOUT1:.+]] = #iree_vector_ext.nested<subgroups_per_workgroup = [1, 1], batches_per_subgroup = [2, 4], duplicates_per_batch = [4, 1], threads_per_duplicate = [4, 2], elements_per_thread = [1, 4], subgroup_order = [0, 1], batch_order = [0, 1], duplicate_order = [1, 0], thread_order = [0, 1], element_order = [0, 1]>
// CHECK-LABEL: func.func @specify_nested
// CHECK:      iree_vector_ext.layout_conflict_resolution
// CHECK-SAME:         desiredLayout = #[[LAYOUT0]]
// CHECK-SAME:         sourceLayout = #[[LAYOUT1]]




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
