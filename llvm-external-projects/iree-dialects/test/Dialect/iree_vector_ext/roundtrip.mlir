// RUN: iree-dialects-opt --split-input-file %s | FileCheck %s

#row_layout1 = #iree_vector_ext.per_dim_layout<"BatchX"<"LaneX"<"VecY", 2>, 4>, 4>
#col_layout1 = #iree_vector_ext.per_dim_layout<"BatchY"<"LaneY"<"VecX", 4>, 2>, 4>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
func.func @specify_layout(%lhs: memref<32x32xf16>, %rhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  %2 = iree_vector_ext.layout_conflict_resolution %result {desiredLayout = #layout1} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// CHECK-LABEL: func.func @specify_layout
// CHECK:         iree_vector_ext.layout_conflict_resolution
// CHECK-SAME:      {desiredLayout = #iree_vector_ext.layout<#iree_vector_ext.per_dim_layout<BatchX<LaneX<VecY, 2>, 4>, 4>,
// CHECK-SAME:                                              #iree_vector_ext.per_dim_layout<BatchY<LaneY<VecX, 4>, 2>, 4>>}

// -----
