// RUN: iree-dialects-opt -linalg-interp-transforms -split-input-file -verify-diagnostics -allow-unregistered-dialect %s

// This cannot be vectorized because of dynamic tensor shapes. We expect the
// pass fail and report an error at the vectorization operation below.
func public @non_vectorizable(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
    ins(%arg0: tensor<?xf32>) outs(%arg1: tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = arith.mulf %arg2, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @target_pattern
  // expected-error@below {{failed to apply}}
  vectorize %0
}

// -----

func public @no_loop(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
    ins(%arg0: tensor<?xf32>) outs(%arg1: tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = arith.mulf %arg2, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @target_pattern
  // expected-error@below {{the transformed op is enclosed by 0 loops, but 1 expected}}
  // expected-error@below {{failed to apply}}
  get_parent_loop %0
}

// -----

func private @prevent_dce()

pdl.pattern @something : benefit(1) {
  %0 = operands
  %2 = operation "scf.for"(%0 : !pdl.range<value>)
  rewrite %2 with "iree_linalg_transform.apply"
}

func public @loop(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    call @prevent_dce() : () -> ()
  }
  return
}

iree_linalg_transform.sequence {
  %0 = match @something
  // expected-error@below {{NYI: cannot target the result of pipelining}}
  // expected-error@below {{failed to apply}}
  %1 = pipeline_loop %0
  // expected-note@below {{use here}}
  get_parent_loop %1
}

// -----

func public @no_outlining() {
  "some.operation"() ({}, {}) : () -> ()
  return
}

pdl.pattern @some_operation : benefit(1) {
  %0 = operation "some.operation"
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @some_operation
  // Make sure we don't crash on wrong operation type.
  // expected-error@below {{failed to apply}}
  outline_loop %0 {func_name = "outlined"}
}

// -----

func @no_replacement(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // expected-error @below {{could not find replacement for tracked op}}
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@no_replacement](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  // expected-error @below {{failed to apply}}
  vectorize
  tile %0
}

// -----

func @repeated_match(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
  %arg2: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // expected-error @below {{operation tracked by two handles}}
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

pdl.pattern @pdl_target1 : benefit(1) {
  %args = operands
  %results = types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@repeated_match](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

// An exact copy of the above, but with a different name.
pdl.pattern @pdl_target2 : benefit(1) {
  %args = operands
  %results = types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@repeated_match](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  // expected-note @below {{handle}}
  %0 = match @pdl_target1
  // expected-error @below {{failed to apply}}
  // expected-note @below {{handle}}
  %1 = match @pdl_target2
  
  // Add references to handles produced by match so that they are not DCE'd.
  tile %0
  tile %1
}
