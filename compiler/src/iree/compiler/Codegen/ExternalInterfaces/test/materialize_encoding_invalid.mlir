/// Test that layout compatibility checks are correctly applied when encoding
/// materialization is run on unsupported linalg.generic + encoding combinations.
/// To avoid any doubt, the examples tested here are synthetic and not expected from
/// data tiling/fusion. Testing against these examples enforces cleaner diagnostics
/// from codegen passes.

// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file --verify-diagnostics %s

// COM: The second output uses a transpose indexing map but retains an untransposed
// physical layout. The first output materializes as tensor<1x2x4x16xf32>, while
// second output materializes as tensor<8x1x4x16xf32>. Since the packed shapes aren't
// permutations of one another, the packed outputs cannot share an iteration domain
// using only permutation indexing maps.
#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 16], outerDimsPerm = [0, 1]}}>
#encoding = #iree_encoding.layout<[#layout]>
#target = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
// expected-error @+1 {{'func.func' op materialization failed}}
func.func @incompatible_transposed_output_layout()
    -> (tensor<4x32xf32, #encoding>, tensor<32x4xf32, #encoding>)
    attributes {hal.executable.target = #target} {
  %c0 = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<4x32xf32, #encoding>
  %empty1 = tensor.empty() : tensor<32x4xf32, #encoding>
  // expected-error @+1 {{failed to legalize operation 'linalg.generic' that was explicitly marked illegal}}
  %result:2 = linalg.generic {
      indexing_maps = [#id, #transpose],
      iterator_types = ["parallel", "parallel"]
  } outs(%empty0, %empty1 : tensor<4x32xf32, #encoding>, tensor<32x4xf32, #encoding>) {
  ^bb0(%out0: f32, %out1: f32):
    linalg.yield %c0, %c0 : f32, f32
  } -> (tensor<4x32xf32, #encoding>, tensor<32x4xf32, #encoding>)
  return %result#0, %result#1 : tensor<4x32xf32, #encoding>, tensor<32x4xf32, #encoding>
}

// -----

// COM: No layout is encoded for the first output. The second output is packed and materialized
// to rank 4. A shared iteration domain cannot be preserved because of a rank mismatch between
// the outputs.
#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 16], outerDimsPerm = [0, 1]}}>
#encoding = #iree_encoding.layout<[#layout]>
#target = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
// expected-error @+1 {{'func.func' op materialization failed}}
func.func @mixed_identity_and_packed_output_layouts()
    -> (tensor<4x32xf32>, tensor<32x4xf32, #encoding>)
    attributes {hal.executable.target = #target} {
  %c0 = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<4x32xf32>
  %empty1 = tensor.empty() : tensor<32x4xf32, #encoding>
  // expected-error @+1 {{failed to legalize operation 'linalg.generic' that was explicitly marked illegal}}
  %result:2 = linalg.generic {
      indexing_maps = [#id, #transpose],
      iterator_types = ["parallel", "parallel"]
  } outs(%empty0, %empty1 : tensor<4x32xf32>, tensor<32x4xf32, #encoding>) {
  ^bb0(%out0: f32, %out1: f32):
    linalg.yield %c0, %c0 : f32, f32
  } -> (tensor<4x32xf32>, tensor<32x4xf32, #encoding>)
  return %result#0, %result#1 : tensor<4x32xf32>, tensor<32x4xf32, #encoding>
}

// -----

// The first output's swizzle expands its inner tiles into four dimensions,
// producing tensor<1x2x2x2x4x4xf32>. The second output has two unswizzled inner
// tile dimensions and materializes as tensor<2x1x16x4xf32>. Its inner
// coordinates require composing multiple packed iteration dimensions, so the
// outputs cannot share an iteration domain using only permutation indexing maps.
#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#swizzled_layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = [[["Internal", 2], ["Internal", 2]], [["Internal", 4], ["Internal", 4]]], permutation = [0, 1, 2, 3]}}}>
#output_layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 4], outerDimsPerm = [0, 1]}}>
#swizzled_encoding = #iree_encoding.layout<[#swizzled_layout]>
#output_encoding = #iree_encoding.layout<[#output_layout]>
#target = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
// expected-error @+1 {{'func.func' op materialization failed}}
func.func @incompatible_swizzle_expansion()
    -> (tensor<4x32xf32, #swizzled_encoding>, tensor<32x4xf32, #output_encoding>)
    attributes {hal.executable.target = #target} {
  %c0 = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<4x32xf32, #swizzled_encoding>
  %empty1 = tensor.empty() : tensor<32x4xf32, #output_encoding>
  // expected-error @+1 {{failed to legalize operation 'linalg.generic' that was explicitly marked illegal}}
  %result:2 = linalg.generic {
      indexing_maps = [#id, #transpose],
      iterator_types = ["parallel", "parallel"]
  } outs(%empty0, %empty1 : tensor<4x32xf32, #swizzled_encoding>, tensor<32x4xf32, #output_encoding>) {
  ^bb0(%out0: f32, %out1: f32):
    linalg.yield %c0, %c0 : f32, f32
  } -> (tensor<4x32xf32, #swizzled_encoding>, tensor<32x4xf32, #output_encoding>)
  return %result#0, %result#1 : tensor<4x32xf32, #swizzled_encoding>, tensor<32x4xf32, #output_encoding>
}
