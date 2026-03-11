// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// ============================================================================
// pack_map roundtrips (no coalescing — preserves structure exactly)
// ============================================================================

// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(8) : (1)>
// CHECK-LABEL: func @pack_map_rank1_contiguous
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_rank1_contiguous() attributes {layout = #iree_map.pack_map<(8) : (1)>} {
  return
}

// -----

// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(8) : (2)>
// CHECK-LABEL: func @pack_map_rank1_strided
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_rank1_strided() attributes {layout = #iree_map.pack_map<(8) : (2)>} {
  return
}

// -----

// Rank-2 column-major.
// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(4, 8) : (1, 4)>
// CHECK-LABEL: func @pack_map_rank2_col_major
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_rank2_col_major() attributes {layout = #iree_map.pack_map<(4, 8) : (1, 4)>} {
  return
}

// -----

// Rank-2 row-major.
// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(4, 8) : (8, 1)>
// CHECK-LABEL: func @pack_map_rank2_row_major
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_rank2_row_major() attributes {layout = #iree_map.pack_map<(4, 8) : (8, 1)>} {
  return
}

// -----

// Rank-3.
// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(2, 4, 8) : (1, 2, 8)>
// CHECK-LABEL: func @pack_map_rank3
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_rank3() attributes {layout = #iree_map.pack_map<(2, 4, 8) : (1, 2, 8)>} {
  return
}

// -----

// Hierarchical shape and stride (MMA-style).
// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<((2, 4), 8) : ((1, 16), 4)>
// CHECK-LABEL: func @pack_map_hierarchical
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_hierarchical() attributes {layout = #iree_map.pack_map<((2, 4), 8) : ((1, 16), 4)>} {
  return
}

// -----

// Deeply nested hierarchy — pack_map preserves nesting exactly.
// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(((2, 2), 4), 8) : (((1, 4), 16), 2)>
// CHECK-LABEL: func @pack_map_deeply_nested
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_deeply_nested() attributes {layout = #iree_map.pack_map<(((2, 2), 4), 8) : (((1, 4), 16), 2)>} {
  return
}

// -----

// Stride-0 (broadcast mode).
// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(4, 8) : (0, 1)>
// CHECK-LABEL: func @pack_map_broadcast
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_broadcast() attributes {layout = #iree_map.pack_map<(4, 8) : (0, 1)>} {
  return
}

// -----

// Shape-1 mode (unit extent).
// CHECK: #[[$LAYOUT:.+]] = #iree_map.pack_map<(1, 8) : (0, 1)>
// CHECK-LABEL: func @pack_map_unit_extent
// CHECK-SAME:    layout = #[[$LAYOUT]]
func.func @pack_map_unit_extent() attributes {layout = #iree_map.pack_map<(1, 8) : (0, 1)>} {
  return
}
