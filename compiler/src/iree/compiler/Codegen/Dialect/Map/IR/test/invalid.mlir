// RUN: iree-opt --split-input-file --verify-diagnostics %s

// expected-error @+1 {{shape leaf values must be positive, got 0}}
func.func @zero_shape() attributes {layout = #iree_map.pack_map<(0, 8) : (1, 4)>} {
  return
}

// -----

// expected-error @+1 {{shape leaf values must be positive, got -1}}
func.func @negative_shape() attributes {layout = #iree_map.pack_map<(-1, 8) : (1, 4)>} {
  return
}

// -----

// expected-error @+1 {{stride leaf values must be non-negative, got -1}}
func.func @negative_stride() attributes {layout = #iree_map.pack_map<(4, 8) : (-1, 4)>} {
  return
}

// -----

// expected-error @+1 {{shape and stride must be congruent}}
func.func @non_congruent_stride_nested() attributes {layout = #iree_map.pack_map<(4, 8) : (1, (2, 3))>} {
  return
}

// -----

// expected-error @+1 {{shape and stride must be congruent}}
func.func @non_congruent_shape_nested() attributes {layout = #iree_map.pack_map<((2, 4), 8) : (1, 4)>} {
  return
}

// -----

// PackLayoutAttr delegates validation to PackMapAttr.
// expected-error @+1 {{shape leaf values must be positive, got 0}}
func.func @pack_layout_zero_shape() attributes {layout = #iree_map.pack_layout<(0, 8) : (1, 4)>} {
  return
}
