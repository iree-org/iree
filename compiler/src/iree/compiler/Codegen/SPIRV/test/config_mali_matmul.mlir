// RUN: iree-opt --split-input-file --iree-gpu-test-target=valhall1 --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Large matmul that can match the best tiling scheme.

func.func @matmul_1024x2048x512(%3: tensor<1024x512xf32>, %4: tensor<512x2048xf32>) -> tensor<1024x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1024x2048xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<1024x512xf32>, tensor<512x2048xf32>) outs(%6 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
  return %7 : tensor<1024x2048xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 32], [4, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 1]>
//      CHECK: func.func @matmul_1024x2048x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Small matmul N that can still tile to all threads in a workgroup.

func.func @matmul_3136x24x96(%3: tensor<3136x96xf32>, %4: tensor<96x24xf32>) -> tensor<3136x24xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<3136x24xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<3136x24xf32>) -> tensor<3136x24xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<3136x96xf32>, tensor<96x24xf32>) outs(%6 : tensor<3136x24xf32>) -> tensor<3136x24xf32>
  return %7 : tensor<3136x24xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 8], [4, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 8, 1]>
//      CHECK: func.func @matmul_3136x24x96(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Small matmul M that can still tile to all threads in a workgroup.

func.func @matmul_196x64x192(%3: tensor<196x192xf32>, %4: tensor<192x64xf32>) -> tensor<196x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<196x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<196x64xf32>) -> tensor<196x64xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<196x192xf32>, tensor<192x64xf32>) outs(%6 : tensor<196x64xf32>) -> tensor<196x64xf32>
  return %7 : tensor<196x64xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 32], [2, 4], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 1]>
//      CHECK: func.func @matmul_196x64x192(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:        lowering_config = #[[CONFIG]]

// -----

// Small matmul K that can still tile to all threads in a workgroup.

func.func @matmul_12544x96x16(%0: memref<12544x16xf32>, %1: memref<16x96xf32>, %2: memref<12544x96xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%2 : memref<12544x96xf32>)
  linalg.matmul ins(%0, %1 : memref<12544x16xf32>, memref<16x96xf32>) outs(%2 : memref<12544x96xf32>)
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 32], [4, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 1]>
//      CHECK: func.func @matmul_12544x96x16(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Odd matmul M and small N that cannot utilize all threads in a workgroup.

func.func @matmul_49x160x576(%3: tensor<49x576xf32>, %4: tensor<576x160xf32>) -> tensor<49x160xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<49x160xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<49x160xf32>) -> tensor<49x160xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<49x576xf32>, tensor<576x160xf32>) outs(%6 : tensor<49x160xf32>) -> tensor<49x160xf32>
  return %7 : tensor<49x160xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 32], [1, 4], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 1, 1]>
//      CHECK: func.func @matmul_49x160x576(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Small matmul M to "shift" parallelism to N.

func.func @matmul_2x1024x576(%4: tensor<2x576xf32>, %5: tensor<576x1024xf32>) -> tensor<2x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %7 = tensor.empty() : tensor<2x1024xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<2x1024xf32>) -> tensor<2x1024xf32>
  %9 = linalg.matmul ins(%4, %5 : tensor<2x576xf32>, tensor<576x1024xf32>) outs(%8 : tensor<2x1024xf32>) -> tensor<2x1024xf32>
  return %9 : tensor<2x1024xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[2, 128], [2, 4], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//      CHECK: func.func @matmul_2x1024x576(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Large matmul with i8 inputs.

func.func @matmul_1024x2048x512xi8(%3: tensor<1024x512xi8>, %4: tensor<512x2048xi8>) -> tensor<1024x2048xi32> {
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<1024x2048xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<1024x2048xi32>) -> tensor<1024x2048xi32>
  %7 = linalg.matmul ins(%3, %4 : tensor<1024x512xi8>, tensor<512x2048xi8>) outs(%6 : tensor<1024x2048xi32>) -> tensor<1024x2048xi32>
  return %7 : tensor<1024x2048xi32>
}

// -----

func.func @batch_matmul_4x384x384(%3: tensor<4x384x32xf32>, %4: tensor<4x32x384xf32>) -> tensor<4x384x384xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<4x384x384xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x384x384xf32>) -> tensor<4x384x384xf32>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<4x384x32xf32>, tensor<4x32x384xf32>) outs(%6 : tensor<4x384x384xf32>) -> tensor<4x384x384xf32>
  return %7 : tensor<4x384x384xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 12, 32], [1, 6, 4], [0, 0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 1]>
//      CHECK: func.func @batch_matmul_4x384x384(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Small batch matmul.

func.func @batch_matmul_4x2x8(%3: tensor<4x2x32xf32>, %4: tensor<4x32x8xf32>) -> tensor<4x2x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<4x2x8xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x2x8xf32>) -> tensor<4x2x8xf32>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<4x2x32xf32>, tensor<4x32x8xf32>) outs(%6 : tensor<4x2x8xf32>) -> tensor<4x2x8xf32>
  return %7 : tensor<4x2x8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 2, 8], [1, 1, 4], [0, 0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 2, 1]>
//      CHECK: func.func @batch_matmul_4x2x8(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Linalg.generic that is a batch matmul.

#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @generic_batch_matmul_32x2x512(%3: tensor<8x32x64xf32>, %4: tensor<32x64x512xf32>) -> tensor<32x8x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<32x8x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x8x512xf32>) -> tensor<32x8x512xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<8x32x64xf32>, tensor<32x64x512xf32>) outs(%6 : tensor<32x8x512xf32>) attrs =  {linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.mulf %in, %in_0 : f32
    %9 = arith.addf %out, %8 : f32
    linalg.yield %9 : f32
  } -> tensor<32x8x512xf32>
  return %7 : tensor<32x8x512xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 8, 32], [1, 4, 4], [0, 0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 1]>
//      CHECK: func.func @generic_batch_matmul_32x2x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Linalg.generic that is a batch matmul.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @generic_batch_matmul_8x2500x512x4608(%5: tensor<8x2500x4608xf32>, %6: tensor<4608x512xf32>, %7: tensor<8x2500x512xf32>, %8: tensor<8x2500x512xf32>) -> tensor<8x2500x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %9 = tensor.empty() : tensor<8x2500x512xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<8x2500x512xf32>) -> tensor<8x2500x512xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %6 : tensor<8x2500x4608xf32>, tensor<4608x512xf32>) outs(%10 : tensor<8x2500x512xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %13 = arith.mulf %in, %in_0 : f32
    %14 = arith.addf %13, %out : f32
    linalg.yield %14 : f32
  } -> tensor<8x2500x512xf32>
  %12 = linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %7, %8 : tensor<8x2500x512xf32>, tensor<8x2500x512xf32>, tensor<8x2500x512xf32>) outs(%9 : tensor<8x2500x512xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %13 = arith.addf %in, %in_0 : f32
    %14 = arith.subf %13, %in_1 : f32
    linalg.yield %14 : f32
  } -> tensor<8x2500x512xf32>
  return %12 : tensor<8x2500x512xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 10, 32], [1, 5, 4], [0, 0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 1]>
//      CHECK: func.func @generic_batch_matmul_8x2500x512x4608(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]
