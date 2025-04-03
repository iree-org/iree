// RUN: iree-opt --split-input-file --verify-diagnostics %s

#layout1 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [1, 1],

  subgroup_strides = [0, 0],
  thread_strides = [0, 0]>

func.func @invalid_layout(%lhs: memref<32x32xf16>, %rhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  // expected-error @+1 {{Vector shape: [32, 32] does not match the layout (nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [1, 1], element_tile = [1, 1], subgroup_strides = [0, 0], thread_strides = [0, 0]>) at dim 0. Dimension expected by layout: 1 actual: 32}}
  %2 = iree_vector_ext.to_layout %result to layout(#layout1) : vector<32x32xf16>
  return %2 : vector<32x32xf16>
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

func.func @indexing_map_mismatch(%indices: vector<128x64xindex>,
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
