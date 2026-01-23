// RUN: iree-opt --split-input-file --verify-diagnostics %s

// Test that innerDimsPos index out of bounds for tensor rank is rejected.
// The tensor is rank 2, but innerDimsPos references index 2 which is out of bounds.

// expected-error @+2 {{innerDimsPos index 2 is out of bounds for tensor rank 2}}
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 2], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>
func.func @invalid_dims_pos_out_of_bounds(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}

// -----

// Test that innerDimsPos with negative index is rejected.

// expected-error @+2 {{innerDimsPos index -1 is out of bounds for tensor rank 2}}
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [-1, 0], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>
func.func @invalid_negative_dims_pos(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}

// -----

// Test that mismatched innerDimsPos and innerTileSizes sizes is rejected.

// expected-error @+2 {{innerDimsPos size (2) does not match innerTileSizes size (3)}}
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16, 8], outerDimsPerm = [0, 1]}}>
func.func @invalid_mismatched_sizes(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}

// -----

// Test that innerDimsPos with duplicate indices is rejected.

// expected-error @+2 {{innerDimsPos contains duplicate index 0}}
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 0], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>
func.func @invalid_inner_dims_pos_duplicate(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}

// -----

// Test that outerDimsPerm with out of bounds index is rejected.

// expected-error @+2 {{outerDimsPerm index 2 is out of bounds for tensor rank 2}}
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 2]}}>
func.func @invalid_outer_dims_perm_out_of_bounds(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}

// -----

// Test that outerDimsPerm with duplicate indices is rejected.

// expected-error @+2 {{outerDimsPerm contains duplicate index 0}}
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 0]}}>
func.func @invalid_outer_dims_perm_duplicate(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}

// -----

// Test that swizzle with permutation size mismatch is rejected.
// The expandShape produces 2 total dimensions, but permutation has 3 elements.

// expected-error @+2 {{swizzle permutation size (3) does not match total expanded dimensions (2)}}
#encoding_swizzle = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 16]], [["Internal", 16]]], permutation = [0, 1, 2]}}}>
func.func @invalid_swizzle_permutation_size_mismatch(%arg0: tensor<32x64xf32, #encoding_swizzle>) -> tensor<32x64xf32, #encoding_swizzle> {
  return %arg0 : tensor<32x64xf32, #encoding_swizzle>
}

// -----

// Test that swizzle with permutation index out of bounds is rejected.
// The expandShape produces 2 dimensions (indices 0-1), but permutation contains index 2.

// expected-error @+2 {{swizzle permutation is not a valid permutation}}
#encoding_swizzle = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 16]], [["Internal", 16]]], permutation = [2, 0]}}}>
func.func @invalid_swizzle_permutation_out_of_bounds(%arg0: tensor<32x64xf32, #encoding_swizzle>) -> tensor<32x64xf32, #encoding_swizzle> {
  return %arg0 : tensor<32x64xf32, #encoding_swizzle>
}

// -----

// Test that swizzle with duplicate permutation indices is rejected.

// expected-error @+2 {{swizzle permutation is not a valid permutation}}
#encoding_swizzle = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 16]], [["Internal", 16]]], permutation = [0, 0]}}}>
func.func @invalid_swizzle_permutation_duplicate(%arg0: tensor<32x64xf32, #encoding_swizzle>) -> tensor<32x64xf32, #encoding_swizzle> {
  return %arg0 : tensor<32x64xf32, #encoding_swizzle>
}

// -----

// Test that swizzle expandShape size mismatch with innerTileSizes is rejected.
// innerTileSizes has 2 entries but expandShape has 3.

// expected-error @+2 {{swizzle expandShape size (3) does not match innerTileSizes size (2)}}
#encoding_swizzle = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 16]], [["Internal", 16]], [["Internal", 8]]], permutation = [0, 1, 2]}}}>
func.func @invalid_swizzle_expand_shape_size_mismatch(%arg0: tensor<32x64xf32, #encoding_swizzle>) -> tensor<32x64xf32, #encoding_swizzle> {
  return %arg0 : tensor<32x64xf32, #encoding_swizzle>
}

// -----

// Test that swizzle expandShape product mismatch with innerTileSizes is rejected.
// innerTileSizes[0] is 16, but expandShape[0] product is 4*8 = 32.

// expected-error @+2 {{swizzle expandShape[0] product (32) does not match innerTileSizes[0] (16)}}
#encoding_swizzle = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 4], ["Internal", 8]], [["Internal", 16]]], permutation = [0, 1, 2]}}}>
func.func @invalid_swizzle_expand_shape_product_mismatch(%arg0: tensor<32x64xf32, #encoding_swizzle>) -> tensor<32x64xf32, #encoding_swizzle> {
  return %arg0 : tensor<32x64xf32, #encoding_swizzle>
}

// -----

// Test that swizzle with negative permutation index is rejected.

// expected-error @+2 {{swizzle permutation is not a valid permutation}}
#encoding_swizzle = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 16]], [["Internal", 16]]], permutation = [-1, 0]}}}>
func.func @invalid_swizzle_permutation_negative(%arg0: tensor<32x64xf32, #encoding_swizzle>) -> tensor<32x64xf32, #encoding_swizzle> {
  return %arg0 : tensor<32x64xf32, #encoding_swizzle>
}

// -----

// CPU encoding resolver invalid test.
// Note: Most encoding verifier tests are done on gpu_encoding_resolver above.
// This test verifies that cpu_encoding_resolver also rejects invalid encodings.

// expected-error @+2 {{innerDimsPos index 2 is out of bounds for tensor rank 2}}
#cpu_encoding = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 2], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>
func.func @cpu_invalid_dims_pos_out_of_bounds(%arg0: tensor<32x64xf32, #cpu_encoding>) -> tensor<32x64xf32, #cpu_encoding> {
  return %arg0 : tensor<32x64xf32, #cpu_encoding>
}

// -----

// VMVX encoding resolver invalid test.
// Note: Most encoding verifier tests are done on gpu_encoding_resolver above.
// This test verifies that vmvx_encoding_resolver also rejects invalid encodings.

// expected-error @+2 {{innerDimsPos index 2 is out of bounds for tensor rank 2}}
#vmvx_encoding = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 2], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>
func.func @vmvx_invalid_dims_pos_out_of_bounds(%arg0: tensor<32x64xf32, #vmvx_encoding>) -> tensor<32x64xf32, #vmvx_encoding> {
  return %arg0 : tensor<32x64xf32, #vmvx_encoding>
}
