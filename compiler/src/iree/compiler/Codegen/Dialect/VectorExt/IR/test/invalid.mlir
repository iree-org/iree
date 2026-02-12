// RUN: iree-opt --split-input-file --verify-diagnostics %s

#layout1 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [1, 1],

  subgroup_strides = [0, 0],
  thread_strides = [0, 0]>

func.func @invalid_layout(%arg0: vector<32x32xf16>) -> vector<32x32xf16> {
  // expected-error @+1 {{Vector shape: [32, 32] does not match the layout (nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [1, 1], element_tile = [1, 1], subgroup_strides = [0, 0], thread_strides = [0, 0]>) at dim 0. Dimension expected by layout: 1 actual: 32}}
  %0 = iree_vector_ext.to_layout %arg0 to layout(#layout1) : vector<32x32xf16>
  return %0 : vector<32x32xf16>
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
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [1],
  element_tile = [1],

  subgroup_strides = [0, 0],
  thread_strides = [0]
>

// -----

func.func @indexing_map_mismatch(%indices: vector<128xindex>,
  %source: tensor<128xf16>)
  -> vector<128xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{op requires number of results for corressponding indexing map to match the rank of index vector at dim: 0}}
  %out = iree_vector_ext.transfer_gather %source[%c0]
  [%indices: vector<128xindex>], %cst0
  { indexed_maps = [affine_map<(d0, d1) -> (d0, d1)>]}
  : tensor<128xf16>, vector<128xf16>

  return %out : vector<128xf16>
}

// -----

func.func @indexing_map_invalid_index_vector_shape(%indices: vector<128x64xindex>,
  %source: tensor<128x64xf16>)
  -> vector<128x64xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{'iree_vector_ext.transfer_gather' op Invalid index vector shape at dim: 0, expected: 64, 128, got: 128, 64}}
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [None, %indices: vector<128x64xindex>], %cst0
  { indexed_maps = [affine_map<(d0, d1) -> (d1, d0)>]}
  : tensor<128x64xf16>, vector<128x64xf16>

  return %out : vector<128x64xf16>
}

// -----

func.func @arg_compare_dimension_out_of_bounds(%input: vector<4x128xf32>,
                                               %out_val: vector<4xf32>,
                                               %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  // expected-error @+1 {{dimension}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(2)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_negative_dimension(%input: vector<4x128xf32>,
                                          %out_val: vector<4xf32>,
                                          %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  // expected-error @+1 {{dimension}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(-1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_wrong_output_rank(%input: vector<4x128xf32>,
                                         %out_val: vector<4x8xf32>,
                                         %out_idx: vector<4xi32>)
    -> (vector<4x8xf32>, vector<4xi32>) {
  // expected-error @+1 {{init value rank (2) must be input rank - 1 (1)}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4x8xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4x8xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4x8xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_wrong_output_shape(%input: vector<4x128xf32>,
                                          %out_val: vector<8xf32>,
                                          %out_idx: vector<8xi32>)
    -> (vector<8xf32>, vector<8xi32>) {
  // expected-error @+1 {{init value shape must match input shape with reduction dimension removed}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<8xf32>, vector<8xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<8xf32>, vector<8xi32>
  return %result#0, %result#1 : vector<8xf32>, vector<8xi32>
}

// -----

func.func @arg_compare_wrong_output_element_type(%input: vector<4x128xf32>,
                                                  %out_val: vector<4xf16>,
                                                  %out_idx: vector<4xi32>)
    -> (vector<4xf16>, vector<4xi32>) {
  // expected-error @+1 {{failed to verify that all of {input_value, init_value} have same element type}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf16>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf16>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf16>, vector<4xi32>
}

// -----

func.func @arg_compare_explicit_index_shape_mismatch(%input_val: vector<4x128xf32>,
                                                     %input_idx: vector<8x128xi32>,
                                                     %out_val: vector<4xf32>,
                                                     %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  // expected-error @+1 {{explicit-index mode: value and index inputs must have the same shape. Value shape: [4, 128], index shape: [8, 128]}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input_val, %input_idx : vector<4x128xf32>, vector<8x128xi32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_explicit_index_element_type_mismatch(%input_val: vector<4x128xf32>,
                                                            %input_idx: vector<4x128xi64>,
                                                            %out_val: vector<4xf32>,
                                                            %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  // expected-error @+1 {{explicit-index mode: input and init index element types must match. Input index type: 'i64', init index type: 'i32'}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input_val, %input_idx : vector<4x128xf32>, vector<4x128xi64>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_wrong_comparator_args(%input: vector<4x128xf32>,
                                             %out_val: vector<4xf32>,
                                             %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  // expected-error @+1 {{comparator region must have exactly 2 arguments}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_output_index_not_integer(%input: vector<4x128xf32>,
                                                 %out_val: vector<4xf32>,
                                                 %out_idx: vector<4xf32>)
    -> (vector<4xf32>, vector<4xf32>) {
  // expected-error @+1 {{init index must have integer or index element type}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xf32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xf32>
}

// -----

func.func @arg_compare_explicit_index_not_integer(%input_val: vector<4x128xf32>,
                                                   %input_idx: vector<4x128xf32>,
                                                   %out_val: vector<4xf32>,
                                                   %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  // expected-error @+1 {{explicit-index mode: index input must have integer or index element type, but got 'f32'}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input_val, %input_idx : vector<4x128xf32>, vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_index_base_with_explicit_index(%input_val: vector<4x128xf32>,
                                                       %input_idx: vector<4x128xi32>,
                                                       %out_val: vector<4xf32>,
                                                       %out_idx: vector<4xi32>,
                                                       %base: index)
    -> (vector<4xf32>, vector<4xi32>) {
  // expected-error @+1 {{index_base must not be used with explicit indices}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input_val, %input_idx : vector<4x128xf32>, vector<4x128xi32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>)
      index_base(%base : index) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_result_value_type_mismatch(%input: vector<4x128xf32>,
                                                   %out_val: vector<4xf32>,
                                                   %out_idx: vector<4xi32>)
    -> (vector<4xf16>, vector<4xi32>) {
  // expected-error @+1 {{failed to verify that all of {init_value, result_value} have same type}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf16>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf16>, vector<4xi32>
}

// -----

func.func @arg_compare_result_index_type_mismatch(%input: vector<4x128xf32>,
                                                   %out_val: vector<4xf32>,
                                                   %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi64>) {
  // expected-error @+1 {{failed to verify that all of {init_index, result_index} have same type}}
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi64>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi64>
}

// -----

func.func @arg_compare_yield_wrong_operand_count(%input: vector<4x128xf32>,
                                                  %out_val: vector<4xf32>,
                                                  %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      // expected-error @+1 {{expected 1 yield operand, but got 2}}
      iree_vector_ext.yield %cmp, %cmp : i1, i1
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}

// -----

func.func @arg_compare_yield_wrong_operand_type(%input: vector<4x128xf32>,
                                                 %out_val: vector<4xf32>,
                                                 %out_idx: vector<4xi32>)
    -> (vector<4xf32>, vector<4xi32>) {
  %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input : vector<4x128xf32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      // expected-error @+1 {{expected yield operand to have type i1, but got 'f32'}}
      iree_vector_ext.yield %a : f32
  } -> vector<4xf32>, vector<4xi32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xi32>
}
