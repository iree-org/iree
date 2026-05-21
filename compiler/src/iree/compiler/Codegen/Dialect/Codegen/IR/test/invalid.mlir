// RUN: iree-opt --split-input-file --verify-diagnostics %s

// -----

func.func @export_config_invalid_type() attributes {
  // expected-error @+1 {{expected workgroup size to have atmost 3 entries}}
  export_config = #iree_codegen.export_config<workgroup_size = [4, 1, 1, 1]>
} {
  return
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
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
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
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {}
   dims(%arg0) {
  ^bb0(%m: !smt.int, %extra: !smt.int):
  }
  return
}

// -----

// dispatch_config with no terminator. Uses generic format to bypass
// SingleBlockImplicitTerminator inserting a yield automatically.
"iree_codegen.dispatch_config"() <{function_ref = @no_terminator, workgroup_size = array<i64: 64, 1, 1>}> ({
^bb0(%w0: index):
  // expected-error@+1 {{block with no terminator}}
  %c1 = "arith.constant"() <{value = 1 : index}> : () -> index
}) : () -> ()

// -----

// expected-error @+1 {{expected terminator to yield exactly 3 operands (workgroup count x, y, z), got 2}}
iree_codegen.dispatch_config @bad_yield
    workgroup_size = [64, 1, 1] {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1 : index, index
}

// -----

// Knob op: duplicate knob name.
func.func @duplicate_knob_name(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
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
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {}
   dims(%arg0, %arg1) {
  ^bb0(%m: !smt.int):
  }
  return
}

// -----

// expected-error @+1 {{expected terminator to yield exactly 3 operands (workgroup count x, y, z), got 0}}
iree_codegen.dispatch_config @empty_yield
    workgroup_size = [64, 1, 1] {
  iree_codegen.yield
}

// -----

// Knob op: knob name not found in knobs dict.
func.func @knob_name_not_found(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
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
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {name = "wg_m"}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    // expected-error @+1 {{'iree_codegen.smt.knob' op knob name 'wg_m' not found in knobs dict}}
    %bad = iree_codegen.smt.knob "wg_m" : !smt.int
  }
  return
}

// -----

// expected-error @+1 {{workgroup_size must have 1 to 3 entries, got 0}}
iree_codegen.dispatch_config @empty_wg_size
    workgroup_size = [] {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}

// -----

// Constraints op: pipeline attr must implement PipelineAttrInterface -- a
// plain string attr does not.
func.func @constraints_invalid_pipeline(%arg0: index) {
  // expected-error @+1 {{custom op 'iree_codegen.smt.constraints' invalid kind of attribute specified}}
  iree_codegen.smt.constraints target = <set = 0>, pipeline = "not_a_pipeline",
   knobs = {}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
  }
  return
}

// -----

func.func @translation_info_invalid_pipeline() attributes {
  // expected-error @+1 {{pass pipeline must implement PipelineAttrInterface}}
  translation_info = #iree_codegen.translation_info<pipeline = "not_a_pipeline">
} {
  return
}

// -----

func.func @translation_info_spec_requires_transform_pipeline() attributes {
  // expected-error @+1 {{transform dialect codegen spec requires transform dialect codegen pipeline}}
  translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline codegen_spec = @foo>
} {
  return
}

// -----

// LookupOp: keys and values size mismatch.
func.func @smt_lookup_size_mismatch(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
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
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
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
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
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

// AssertOp: too few args for format string placeholders.
func.func @assert_too_few_args(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {x = #iree_codegen.smt.int_knob<"x">}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %x = iree_codegen.smt.knob "x" : !smt.int
    %zero = smt.int.constant 0
    %cmp = smt.int.cmp gt %x, %zero
    // expected-error @+1 {{'iree_codegen.smt.assert' op format string has 2 placeholder(s) but got 1 arg(s)}}
    iree_codegen.smt.assert %cmp, "x ({}) > {}", %x : !smt.bool, !smt.int
  }
  return
}

// -----

// AssertOp: too many args for format string placeholders.
func.func @assert_too_many_args(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {x = #iree_codegen.smt.int_knob<"x">}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %x = iree_codegen.smt.knob "x" : !smt.int
    %zero = smt.int.constant 0
    %cmp = smt.int.cmp gt %x, %zero
    // expected-error @+1 {{'iree_codegen.smt.assert' op format string has 1 placeholder(s) but got 2 arg(s)}}
    iree_codegen.smt.assert %cmp, "x ({})", %x, %zero : !smt.bool, !smt.int, !smt.int
  }
  return
}

// -----

// OneOfKnobAttr: knob name not found in knobs dict.
func.func @one_of_knob_name_not_found(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {mma = #iree_codegen.smt.one_of_knob<"mma_idx", ["a", "b"]>}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    // expected-error @+1 {{'iree_codegen.smt.knob' op knob name 'nonexistent' not found in knobs dict}}
    %bad = iree_codegen.smt.knob "nonexistent" : !smt.int
  }
  return
}

// -----

// expected-error @+1 {{workgroup_size must have 1 to 3 entries, got 4}}
iree_codegen.dispatch_config @big_wg_size
    workgroup_size = [64, 1, 1, 1] {
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c1, %c1, %c1 : index, index, index
}
