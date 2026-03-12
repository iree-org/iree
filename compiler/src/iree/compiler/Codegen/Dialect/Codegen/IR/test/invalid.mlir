// RUN: iree-opt --split-input-file --verify-diagnostics %s

// -----

module {
  func.func @export_config_invalid_type() attributes {
    // expected-error @+1 {{expected workgroup size to have atmost 3 entries}}
    export_config = #iree_codegen.export_config<workgroup_size = [4, 1, 1, 1]>
  } {
    return
  }
}

// -----

func.func @load_from_buffer_invalid_shape(%arg0: memref<5xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{buffer and tensor shapes must be compatible and element types must match}}
  %value = iree_codegen.load_from_buffer %arg0 : memref<5xf32> -> tensor<4xf32>
  return %value : tensor<4xf32>
}

// -----

func.func @load_from_buffer_invalid_element_type(%arg0: memref<4xf32>) -> tensor<4xf16> {
  // expected-error @+1 {{buffer and tensor shapes must be compatible and element types must match}}
  %value = iree_codegen.load_from_buffer %arg0 : memref<4xf32> -> tensor<4xf16>
  return %value : tensor<4xf16>
}

// -----

func.func @store_to_buffer_invalid_shape(%arg0: tensor<4xf32>, %arg1: memref<5xf32>) {
  // expected-error @+1 {{tensor and buffer shapes must be compatible and element types must match}}
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4xf32> into memref<5xf32>
  return
}

// -----

func.func @store_to_buffer_invalid_element_type(%arg0: tensor<4xf16>, %arg1: memref<4xf32>) {
  // expected-error @+1 {{tensor and buffer shapes must be compatible and element types must match}}
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4xf16> into memref<4xf32>
  return
}

// -----

// Constraints op: block arg wrong type.
func.func @constraints_block_arg_wrong_type(%arg0: index) {
  // expected-error @+1 {{'iree_codegen.smt.constraints' op block argument #0 must be !smt.int but got 'index'}}
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {}
   dims(%arg0) {
  ^bb0(%m: index):
  }
  return
}

// -----

// KnobOp outside of ConstraintsOp.
func.func @knob_outside_constraints() {
  // expected-error @+1 {{'iree_codegen.smt.knob' op expects parent op 'iree_codegen.smt.constraints'}}
  %x = iree_codegen.smt.knob "foo" : !smt.int
  return
}

// -----

// Constraints op: block arg count mismatch with problem_dims.
func.func @constraints_block_arg_mismatch(%arg0: index) {
  // expected-error @+1 {{'iree_codegen.smt.constraints' op expected 1 block arguments but got 2}}
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {}
   dims(%arg0) {
  ^bb0(%m: !smt.int, %extra: !smt.int):
  }
  return
}

// -----

// Knob op: duplicate knob name.
func.func @duplicate_knob_name(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">]}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    // expected-note @+1 {{first occurrence here}}
    %first = iree_codegen.smt.knob "wg_m" : !smt.int
    // expected-error @+1 {{'iree_codegen.smt.knob' op duplicate knob name 'wg_m'}}
    %second = iree_codegen.smt.knob "wg_m" : !smt.int
  }
  return
}

// -----

// Constraints op: too few block args for problem_dims.
func.func @constraints_block_arg_too_few(%arg0: index, %arg1: index) {
  // expected-error @+1 {{'iree_codegen.smt.constraints' op expected 2 block arguments but got 1}}
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {}
   dims(%arg0, %arg1) {
  ^bb0(%m: !smt.int):
  }
  return
}

// -----

// Knob op: knob name not found in knobs dict.
func.func @knob_name_not_found(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">]}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    // expected-error @+1 {{'iree_codegen.smt.knob' op knob name 'nonexistent' not found in knobs dict}}
    %bad = iree_codegen.smt.knob "nonexistent" : !smt.int
  }
  return
}

// -----

// Knob op: bare string in knobs dict does not satisfy knob lookup.
func.func @string_attr_not_a_knob(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {name = "wg_m"}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    // expected-error @+1 {{'iree_codegen.smt.knob' op knob name 'wg_m' not found in knobs dict}}
    %bad = iree_codegen.smt.knob "wg_m" : !smt.int
  }
  return
}

// -----

// Constraints op: pipeline attr must be DispatchLoweringPassPipelineAttr or
// PipelineAttrInterface — a plain string attr is neither.
func.func @constraints_invalid_pipeline(%arg0: index) {
  // expected-error @+1 {{'iree_codegen.smt.constraints' op attribute 'pipeline' failed to satisfy constraint}}
  iree_codegen.smt.constraints target = <set = 0>, pipeline = "not_a_pipeline",
   knobs = {}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
  }
  return
}

// -----

// LookupOp: keys and values size mismatch.
func.func @smt_lookup_size_mismatch(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {x = #iree_codegen.smt.int_knob<"x">}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %x = iree_codegen.smt.knob "x" : !smt.int
    // expected-error @+1 {{'iree_codegen.smt.lookup' op keys and values must have the same size, got 2 keys and 3 values}}
    %r = iree_codegen.smt.lookup %x [0, 1] -> [10, 20, 30] : !smt.int
  }
  return
}

// -----

// LookupOp: empty table.
func.func @smt_lookup_empty(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {x = #iree_codegen.smt.int_knob<"x">}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %x = iree_codegen.smt.knob "x" : !smt.int
    // expected-error @+1 {{'iree_codegen.smt.lookup' op lookup table must be non-empty}}
    %r = iree_codegen.smt.lookup %x [] -> [] : !smt.int
  }
  return
}

// -----

// LookupOp: duplicate keys.
func.func @smt_lookup_duplicate_keys(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {x = #iree_codegen.smt.int_knob<"x">}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %x = iree_codegen.smt.knob "x" : !smt.int
    // expected-error @+1 {{'iree_codegen.smt.lookup' op duplicate key 1 in lookup table}}
    %r = iree_codegen.smt.lookup %x [0, 1, 1] -> [10, 20, 30] : !smt.int
  }
  return
}

// -----

// LookupOp: must be inside ConstraintsOp.
func.func @smt_lookup_outside_constraints(%arg0: !smt.int) {
  // expected-error @+1 {{'iree_codegen.smt.lookup' op expects parent op 'iree_codegen.smt.constraints'}}
  %r = iree_codegen.smt.lookup %arg0 [0] -> [42] : !smt.int
  return
}

// -----

// OneOfKnobAttr: knob name not found in knobs dict.
func.func @one_of_knob_name_not_found(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = None,
   knobs = {mma = #iree_codegen.smt.one_of_knob<"mma_idx", ["a", "b"]>}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    // expected-error @+1 {{'iree_codegen.smt.knob' op knob name 'nonexistent' not found in knobs dict}}
    %bad = iree_codegen.smt.knob "nonexistent" : !smt.int
  }
  return
}
