// RUN: iree-opt --split-input-file --iree-gpu-test-target=rdna3@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=rdna4@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_256x1024x128_div_add(%arg0: tensor<256x1024xf16>, %arg1: tensor<256x1024xf16>, %arg2: tensor<256x128xf16>, %arg3: tensor<128x1024xf16>) -> tensor<256x1024xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256x1024xf16>
  %1 = tensor.empty() : tensor<256x1024xf16>
  %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %3 = linalg.matmul ins(%arg2, %arg3 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%2 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %arg0, %arg1 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>) outs(%0 : tensor<256x1024xf16>) {
  ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
    %5 = arith.divf %in, %in_0 : f16
    %6 = arith.addf %5, %in_1 : f16
    linalg.yield %6 : f16
  } -> tensor<256x1024xf16>
  return %4 : tensor<256x1024xf16>
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128], [32, 64], [0, 0, 32], [16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 1 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @matmul_256x1024x128_div_add(
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @batch_matmul_16x128x256x512_div(%arg0: tensor<16x128x512xf16>, %arg1: tensor<16x512x256xf16>, %arg2: tensor<16x128x256xf16>) -> tensor<16x128x256xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<16x128x256xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<16x128x512xf16>, tensor<16x512x256xf16>) outs(%1 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %arg2 : tensor<16x128x256xf16>, tensor<16x128x256xf16>) outs(%0 : tensor<16x128x256xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %4 = arith.divf %in, %in_0 : f16
    linalg.yield %4 : f16
  } -> tensor<16x128x256xf16>
  return %3 : tensor<16x128x256xf16>
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128], [1, 32, 64], [0, 0, 0, 32], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 1 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @batch_matmul_16x128x256x512_div(
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Linalg.generic that is a batch matmul.

#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @generic_batch_matmul_32x8x512x64(%arg0: tensor<128x32x64xf16>, %arg1: tensor<32x64x512xf16>) -> tensor<32x128x512xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<32x128x512xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<32x128x512xf16>) -> tensor<32x128x512xf16>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x32x64xf16>, tensor<32x64x512xf16>) outs(%1 : tensor<32x128x512xf16>) attrs =  {linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %3 = arith.mulf %in, %in_0 : f16
    %4 = arith.addf %out, %3 : f16
    linalg.yield %4 : f16
  } -> tensor<32x128x512xf16>
  return %2 : tensor<32x128x512xf16>
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128], [1, 32, 64], [0, 0, 0, 32], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 1 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @generic_batch_matmul_32x8x512x64(
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// K dim size not divisble by 32.

func.func @batch_matmul_16x1024x1024x80(%arg0: tensor<16x1024x80xf16>, %arg1: tensor<16x80x1024xf16>) -> tensor<16x1024x1024xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<16x1024x1024xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<16x1024x1024xf16>) -> tensor<16x1024x1024xf16>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<16x1024x80xf16>, tensor<16x80x1024xf16>) outs(%1 : tensor<16x1024x1024xf16>) -> tensor<16x1024x1024xf16>
  return %2 : tensor<16x1024x1024xf16>
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128], [1, 32, 64], [0, 0, 0, 16], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 0 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @batch_matmul_16x1024x1024x80(
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Small K - not supported by cooperative matrix.

func.func @matmul_256x1024x8(%arg0: tensor<256x8xf16>, %arg1: tensor<8x1024xf16>) -> tensor<256x1024xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256x1024xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %2 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%arg0, %arg1 : tensor<256x8xf16>, tensor<8x1024xf16>) outs(%1 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  return %2 : tensor<256x1024xf16>
}

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
// CHECK-LABEL: func.func @matmul_256x1024x8
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
