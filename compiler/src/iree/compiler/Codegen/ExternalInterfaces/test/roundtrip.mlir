// RUN: iree-opt --split-input-file %s | FileCheck %s

// Test that the GPU encoding resolver with valid encoding_info roundtrips correctly.
// The verifier checks that innerDimsPos, outerDimsPerm and swizzle are valid.

#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 16]], [["Internal", 16]]], permutation = [0, 1]}}}>
func.func @valid_2d_encoding(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}
// CHECK-LABEL: func.func @valid_2d_encoding
// CHECK-SAME:    tensor<32x64xf32, #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = {{\[}}{{\[}}{{\[}}"Internal", 16{{\]}}{{\]}}, {{\[}}{{\[}}"Internal", 16{{\]}}{{\]}}{{\]}}, permutation = [0, 1]}}}>

// -----

#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0], innerTileSizes = [32], outerDimsPerm = [0]}}>
func.func @valid_1d_encoding(%arg0: tensor<128xf32, #encoding>) -> tensor<128xf32, #encoding> {
  return %arg0 : tensor<128xf32, #encoding>
}
// CHECK-LABEL: func.func @valid_1d_encoding
// CHECK-SAME:    tensor<128xf32, #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0], innerTileSizes = [32], outerDimsPerm = [0]}}>>

// -----

#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1, 2], innerTileSizes = [8, 16, 4], outerDimsPerm = [0, 1, 2]}}>
func.func @valid_3d_encoding(%arg0: tensor<16x32x64xf32, #encoding>) -> tensor<16x32x64xf32, #encoding> {
  return %arg0 : tensor<16x32x64xf32, #encoding>
}
// CHECK-LABEL: func.func @valid_3d_encoding
// CHECK-SAME:    tensor<16x32x64xf32, #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1, 2], innerTileSizes = [8, 16, 4], outerDimsPerm = [0, 1, 2]}}>>

// -----

// Test with dynamic dimensions - verifier should still validate based on rank.
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>
func.func @valid_dynamic_encoding(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
// CHECK-LABEL: func.func @valid_dynamic_encoding
// CHECK-SAME:    tensor<?x?xf32, #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>>

// -----

// Test with partial innerDimsPos.
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [1], innerTileSizes = [16], outerDimsPerm = [0, 1]}}>
func.func @partial_tiling(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}
// CHECK-LABEL: func.func @partial_tiling
// CHECK-SAME:    tensor<32x64xf32, #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [1], innerTileSizes = [16], outerDimsPerm = [0, 1]}}>>

// -----

// Test encoding resolver without configuration - should always be valid as this is an optional parameter.
#encoding = #iree_gpu.gpu_encoding_resolver<>
func.func @encoding_without_config(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}
// CHECK-LABEL: func.func @encoding_without_config
// CHECK-SAME:    tensor<32x64xf32, #iree_gpu.gpu_encoding_resolver<>>

// -----

// Test encoding with valid swizzle - multi-dimensional expand shape with CrossThread and CrossIntrinsic kinds.
#encoding_swizzle_multi = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [128, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["CrossThread", 2], ["CrossIntrinsic", 4], ["CrossThread", 16]], [["CrossIntrinsic", 4], ["CrossThread", 4]]], permutation = [0, 1, 4, 2, 3]}}}>
func.func @valid_swizzle_multi_expand(%arg0: tensor<256x128xf32, #encoding_swizzle_multi>) -> tensor<256x128xf32, #encoding_swizzle_multi> {
  return %arg0 : tensor<256x128xf32, #encoding_swizzle_multi>
}
// CHECK-LABEL: func.func @valid_swizzle_multi_expand
// CHECK-SAME:    swizzle = {expandShape = {{\[}}{{\[}}{{\[}}"CrossThread", 2], ["CrossIntrinsic", 4], ["CrossThread", 16{{\]}}{{\]}}, {{\[}}{{\[}}"CrossIntrinsic", 4], ["CrossThread", 4{{\]}}{{\]}}{{\]}}, permutation = [0, 1, 4, 2, 3]}

// -----

// Test with non-trivial outerDimsPerm (transpose).
#encoding = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [1, 0]}}>
func.func @valid_transposed_outer_dims(%arg0: tensor<32x64xf32, #encoding>) -> tensor<32x64xf32, #encoding> {
  return %arg0 : tensor<32x64xf32, #encoding>
}
// CHECK-LABEL: func.func @valid_transposed_outer_dims
// CHECK-SAME:    tensor<32x64xf32, #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [1, 0]}}>>

// -----

// CPU encoding resolver tests.
// Note: Most encoding verifier tests are done on gpu_encoding_resolver above.
// This test just verifies that cpu_encoding_resolver roundtrips correctly.

#cpu_encoding = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>
func.func @cpu_valid_2d_encoding(%arg0: tensor<32x64xf32, #cpu_encoding>) -> tensor<32x64xf32, #cpu_encoding> {
  return %arg0 : tensor<32x64xf32, #cpu_encoding>
}
// CHECK-LABEL: func.func @cpu_valid_2d_encoding
// CHECK-SAME:    tensor<32x64xf32, #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>>

// -----

// VMVX encoding resolver tests.
// Note: Most encoding verifier tests are done on gpu_encoding_resolver above.
// This test just verifies that vmvx_encoding_resolver roundtrips correctly.

#vmvx_encoding = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [8, 8], outerDimsPerm = [0, 1]}}>
func.func @vmvx_valid_2d_encoding(%arg0: tensor<32x64xf32, #vmvx_encoding>) -> tensor<32x64xf32, #vmvx_encoding> {
  return %arg0 : tensor<32x64xf32, #vmvx_encoding>
}
// CHECK-LABEL: func.func @vmvx_valid_2d_encoding
// CHECK-SAME:    tensor<32x64xf32, #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [8, 8], outerDimsPerm = [0, 1]}}>>
