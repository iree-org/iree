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
