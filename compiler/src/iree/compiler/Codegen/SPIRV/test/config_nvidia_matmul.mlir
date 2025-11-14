// RUN: iree-opt --split-input-file --iree-gpu-test-target=pascal@vulkan --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-generalize-named-ops),iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

func.func @matmul_4x4096x9216(%4: tensor<4x9216xf32>, %5: tensor<9216x4096xf32>, %6: tensor<4x4096xf32>) -> tensor<4x4096xf32> {
  %7 = linalg.matmul ins(%4, %5 : tensor<4x9216xf32>, tensor<9216x4096xf32>) outs(%6 : tensor<4x4096xf32>) -> tensor<4x4096xf32>
  return %7 : tensor<4x4096xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [32, 4, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @matmul_4x4096x9216(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Matvec does not go down matmul pipelines.

func.func @matmul_1x4096x9216(%4: tensor<1x9216xf32>, %5: tensor<9216x4096xf32>, %6: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  %7 = linalg.matmul ins(%4, %5 : tensor<1x9216xf32>, tensor<9216x4096xf32>) outs(%6 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  return %7 : tensor<1x4096xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 1024]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [256, 1, 1]>
//      CHECK: func.func @matmul_1x4096x9216(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Multi-reduction-dimension transposed-B matmul.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @multi_reduction_transposed_b_matmul(%3: tensor<4096x86x128xf32>, %4: tensor<2048x86x128xf32>) -> tensor<4096x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<4096x2048xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4096x2048xf32>) -> tensor<4096x2048xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%3, %4 : tensor<4096x86x128xf32>, tensor<2048x86x128xf32>) outs(%6 : tensor<4096x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.mulf %in, %in_0 : f32
    %9 = arith.addf %out, %8 : f32
    linalg.yield %9 : f32
  } -> tensor<4096x2048xf32>
  return %7 : tensor<4096x2048xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 128, 1, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [32, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @multi_reduction_transposed_b_matmul(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]
