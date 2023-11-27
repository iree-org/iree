// RUN: iree-dialects-opt --split-input-file --verify-diagnostics %s

#row_layout1 = #iree_vector_ext.per_dim_layout<"BatchX"<"LaneX"<"VecY", 1>, 1>, 1>
#col_layout1 = #iree_vector_ext.per_dim_layout<"BatchY"<"LaneY"<"VecX", 4>, 2>, 4>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
#layout2 = #iree_vector_ext.layout<#col_layout1, #col_layout1>
func.func @invalid_desired_layout(%lhs: memref<32x32xf16>, %rhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  // expected-error @+1 {{The desired layout shape does not match the input shape. Expected shape to be 32, got 1}}
  %2 = iree_vector_ext.layout_conflict_resolution %result {desiredLayout = #layout1, sourceLayout = #layout2} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// -----

#row_layout1 = #iree_vector_ext.per_dim_layout<"BatchX"<"LaneX"<"VecY", 1>, 1>, 1>
#col_layout1 = #iree_vector_ext.per_dim_layout<"BatchY"<"LaneY"<"VecX", 4>, 2>, 4>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
#layout2 = #iree_vector_ext.layout<#col_layout1, #col_layout1>
func.func @invalid_source_layout(%lhs: memref<32x32xf16>, %rhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  // expected-error @+1 {{The source layout shape does not match the input shape. Expected shape to be 32, got 1}}
  %2 = iree_vector_ext.layout_conflict_resolution %result {desiredLayout = #layout2, sourceLayout = #layout1} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// -----
