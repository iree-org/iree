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

func.func @gather_mismatch_dims(%indices: vector<128xindex>,
  %source: memref<4096x64xf16>)
  -> vector<128x64xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected all indexing maps to have number of dims equal to vector rank. expected: 2, got: 3 dims}}
  %out1 = iree_vector_ext.gather %source[%c0, %c0]
  [%indices : vector<128xindex>], %cst0
    { indexing_maps = [affine_map<(d0, d1, d2)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1, d2)[s0] -> (d0)>]}
  : memref<4096x64xf16>, vector<128x64xf16>

  return %out1 : vector<128x64xf16>
}

// -----

func.func @gather_mismatch_syms(%indices: vector<128xindex>,
  %source: memref<4096x64xf16>)
  -> vector<128x64xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected all indexing maps to have number of dims equal to number of index vecs. expected: 1, got: 2 syms}}
  %out1 = iree_vector_ext.gather %source[%c0, %c0]
  [%indices : vector<128xindex>], %cst0
    { indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d0)>]}
  : memref<4096x64xf16>, vector<128x64xf16>

  return %out1 : vector<128x64xf16>
}

// -----

func.func @gather_mismatch_maps(%indices: vector<128xindex>,
  %source: memref<4096x64xf16>)
  -> vector<128x64xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected 2 indexing maps. got: 3}}
  %out1 = iree_vector_ext.gather %source[%c0, %c0]
  [%indices : vector<128xindex>], %cst0
    { indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d0)>,
                       affine_map<(d0, d1)[s0] -> (d1)>]}
  : memref<4096x64xf16>, vector<128x64xf16>

  return %out1 : vector<128x64xf16>
}

// -----

func.func @gather_indirect_index_vec_load(%indices: vector<64xindex>,
  %source: memref<4096x64xf16>)
  -> vector<64x64xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected vector indexing maps to not have any symbols}}
  %out1 = iree_vector_ext.gather %source[%c0, %c0]
  [%indices, %indices : vector<64xindex>, vector<64xindex>], %cst0
    { indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d1)>,
                       affine_map<(d0, d1)[s0, s1] -> (s0)>]}
  : memref<4096x64xf16>, vector<64x64xf16>

  return %out1 : vector<64x64xf16>
}

// -----

func.func @scatter_mismatch_dims(%indices: vector<128xindex>,
  %vector: vector<128x64xf16>,
  %dest: memref<4096x64xf16>) {
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected all indexing maps to have number of dims equal to vector rank. expected: 2, got: 3 dims}}
  iree_vector_ext.scatter %vector, %dest[%c0, %c0]
  [%indices : vector<128xindex>]
    { indexing_maps = [affine_map<(d0, d1, d2)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1, d2)[s0] -> (d0)>]}
  : memref<4096x64xf16>, vector<128x64xf16>
  return
}

// -----

func.func @scatter_mismatch_syms(%indices: vector<128xindex>,
  %vector: vector<128x64xf16>,
  %dest: memref<4096x64xf16>) {
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected all indexing maps to have number of dims equal to number of index vecs. expected: 1, got: 2 syms}}
  iree_vector_ext.scatter %vector, %dest[%c0, %c0]
  [%indices : vector<128xindex>]
    { indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d0)>]}
  : memref<4096x64xf16>, vector<128x64xf16>
  return
}

// -----

func.func @scatter_mismatch_maps(%indices: vector<128xindex>,
  %vector: vector<128x64xf16>,
  %dest: memref<4096x64xf16>) {
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected 2 indexing maps. got: 3}}
  iree_vector_ext.scatter %vector, %dest[%c0, %c0]
  [%indices : vector<128xindex>]
    { indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d0)>,
                       affine_map<(d0, d1)[s0] -> (d1)>]}
  : memref<4096x64xf16>, vector<128x64xf16>
  return
}

// -----

func.func @scatter_indirect_index_vec_load(%indices: vector<64xindex>,
  %vector: vector<64x64xf16>,
  %dest: memref<4096x64xf16>) {
  %c0 = arith.constant 0 : index

  // expected-error @+1 {{expected vector indexing maps to not have any symbols}}
  iree_vector_ext.scatter %vector, %dest[%c0, %c0]
  [%indices, %indices : vector<64xindex>, vector<64xindex>]
    { indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                       affine_map<(d0, d1)[s0, s1] -> (d1)>,
                       affine_map<(d0, d1)[s0, s1] -> (s0)>]}
  : memref<4096x64xf16>, vector<64x64xf16>
  return
}
