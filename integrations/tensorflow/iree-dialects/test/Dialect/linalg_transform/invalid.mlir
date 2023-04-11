// RUN: iree-dialects-opt %s --split-input-file -verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = pdl_match @match in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error@below {{expects iterator_interchange to be a permutation, found 1, 1}}
  transform.structured.interchange %0 iterator_interchange = [1, 1]
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = pdl_match @match in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error@below {{expected 'tile_sizes' attribute}}
  transform.structured.fuse %0
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = pdl_match @match in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error@below {{expects interchange to be a permutation, found [1, 1]}}
  transform.structured.fuse %0 {tile_sizes=[0, 1], tile_interchange = [1, 1]}
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = pdl_match @match in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error@below {{expects pack_paddings to contain booleans (0/1), found [1, 7]}}
  transform.structured.pad %0 {pack_paddings=[1, 7]}
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = pdl_match @match in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error@below {{expects transpose_paddings to be a permutation, found [1, 1]}}
  transform.structured.pad %0 {transpose_paddings=[[1, 1]]}
}
