// RUN: iree-dialects-opt %s -split-input-file -verify-diagnostics

iree_linalg_transform.sequence {
  %0 = match @match
  // expected-error@below {{result #0 has more than one use}}
  %1 = tile %0
  // expected-note@below {{used here as operand #0}}
  tile %1
  // expected-note@below {{used here as operand #0}}
  vectorize %1
}

// -----

iree_linalg_transform.sequence {
  %0 = match @match
  // expected-error@below {{"sizes" and "scalarize_dyn_dims" attributes are mutually exclusive}}
  tile %0 {sizes = [1,2,3], scalarize_dyn_dims = true}
}

// -----

iree_linalg_transform.sequence {
  %0 = match @match
  // expected-error@below {{expects iterator_interchange to be a permutation, found [1, 1]}}
  interchange %0 {iterator_interchange = [1, 1]}
}

// -----

iree_linalg_transform.sequence {
  %0 = match @match
  // expected-error@below {{expects interchange to be a permutation, found [1, 1]}}
  fuse %0 {tile_sizes=[0, 1], tile_interchange = [1, 1]}
}

// -----

iree_linalg_transform.sequence {
  %0 = match @match
  // expected-error@below {{expects pack_paddings to contain booleans (0/1), found [1, 7]}}
  pad %0 {pack_paddings=[1, 7]}
}

// -----

iree_linalg_transform.sequence {
  %0 = match @match
  // expected-error@below {{expects hoist_paddings to contain positive integers, found [1, -7]}}
  pad %0 {hoist_paddings=[1, -7]}
}

// -----

iree_linalg_transform.sequence {
  %0 = match @match
  // expected-error@below {{expects transpose_paddings to be a permutation, found [1, 1]}}
  pad %0 {transpose_paddings=[[1, 1]]}
}
