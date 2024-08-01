// RUN: iree-opt --split-input-file --verify-diagnostics %s

#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [1, 1, 1]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
func.func @invalid_layout(%lhs: memref<32x32xf16>, %rhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Vector shape: [32, 32] does not match the layout (layout<<[ BATCHX,  LANEX,  VECTORY], [1, 1, 1]>, <[ BATCHY,  LANEY,  VECTORX], [4, 2, 4]>>) at dim 0. Dimension expected by layout: 1 actual: 32}}
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  %2 = iree_vector_ext.to_layout %result to #layout1 : vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// -----

func.func @invalid_to_simd_vector_element_type(%simd : vector<2x2xf16>) -> vector<64xf32> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %simt = iree_vector_ext.to_simd %simd : vector<2x2xf16> -> vector<64xf32>
  func.return %simt : vector<64xf32>
}

// -----

func.func @invalid_to_simt_vector_element_type(%simt : vector<64xf32>) -> vector<2x2xf16> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %simd = iree_vector_ext.to_simt %simt : vector<64xf32> -> vector<2x2xf16>
  func.return %simd : vector<2x2xf16>
}

// -----

// expected-error @+1 {{all fields must have the same rank as the layout}}
#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1],
  batches_per_subgroup = [1],
  outers_per_batch = [1],
  threads_per_outer = [1],
  elements_per_thread = [1],

  subgroup_strides = [0, 0],
  thread_strides = [0]
>
