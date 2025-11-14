// RUN: iree-opt --split-input-file --iree-gpu-test-target=rdna2@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

func.func @batch_matmul_f32_16x4096x40x4096(%arg0: tensor<16x4096x4096xf32>, %arg1: tensor<16x4096x40xf32>, %arg2: tensor<16x4096x40xf32>) -> tensor<16x4096x40xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<16x4096x4096xf32>, tensor<16x4096x40xf32>) outs(%arg2 : tensor<16x4096x40xf32>) -> tensor<16x4096x40xf32>
  return %0 : tensor<16x4096x40xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 256, 8, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [2, 32, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @batch_matmul_f32_16x4096x40x4096(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

func.func @matmul_f16_64x640x320(%arg0: tensor<64x320xf16>, %arg1: tensor<320x640xf16>) -> tensor<64x640xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<64x640xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<64x640xf16>) -> tensor<64x640xf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<64x320xf16>, tensor<320x640xf16>) outs(%1 : tensor<64x640xf16>) -> tensor<64x640xf16>
  return %2 : tensor<64x640xf16>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 2 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @matmul_f16_64x640x320(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

func.func @batch_matmul_f32_16x4096x40x4096(%arg0: tensor<16x4096x4096xf32>, %arg1: tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x4096x48xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<16x4096x4096xf32>, tensor<16x4096x48xf32>) outs(%1 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  return %2 : tensor<16x4096x48xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 128, 16, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [4, 16, 1], {pipeline_depth = 2 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @batch_matmul_f32_16x4096x40x4096(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @batch_matmul_f16_1x4096x4096x512(%arg0: tensor<1x4096x512xf16>, %arg1: tensor<1x512x4096xf16>) -> tensor<1x4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1x4096x4096xf32>
  %1 = tensor.empty() : tensor<1x4096x4096xf16>
  %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
  %3 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x4096x512xf16>, tensor<1x512x4096xf16>) outs(%2 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<1x4096x4096xf16>) outs(%0 : tensor<1x4096x4096xf32>) {
  ^bb0(%in: f16, %out: f32):
    %5 = arith.extf %in : f16 to f32
    linalg.yield %5 : f32
  } -> tensor<1x4096x4096xf32>
  return %4 : tensor<1x4096x4096xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @batch_matmul_f16_1x4096x4096x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @matmul_multi_reduce_i4xf32xf32(%arg0: tensor<11008x32x128xi4>, %arg1: tensor<11008x32xf32>, %arg2: tensor<11008x32xf32>, %arg3: tensor<512x32x128xf32>) -> tensor<512x11008xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<512x11008xf32>
  %1 = tensor.empty() : tensor<11008x32x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x11008xf32>) -> tensor<512x11008xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<11008x32x128xi4>, tensor<11008x32xf32>, tensor<11008x32xf32>) outs(%1 : tensor<11008x32x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %5 = arith.extui %in : i4 to i32
    %6 = arith.uitofp %5 : i32 to f32
    %7 = arith.subf %6, %in_1 : f32
    %8 = arith.mulf %7, %in_0 : f32
    linalg.yield %8 : f32
  } -> tensor<11008x32x128xf32>
  %4 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg3, %3 : tensor<512x32x128xf32>, tensor<11008x32x128xf32>) outs(%2 : tensor<512x11008xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.mulf %in, %in_0 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<512x11008xf32>
  return %4 : tensor<512x11008xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 1, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @matmul_multi_reduce_i4xf32xf32(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]
