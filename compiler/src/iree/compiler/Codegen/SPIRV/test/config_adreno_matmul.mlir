// RUN: iree-opt --split-input-file --iree-gpu-test-target=adreno --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Large matmul that can match the best tiling scheme.

func.func @matmul_1024x2048x512(%3: tensor<1024x512xf32>, %4: tensor<512x2048xf32>) -> tensor<1024x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1024x2048xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<1024x512xf32>, tensor<512x2048xf32>) outs(%6 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
  return %7 : tensor<1024x2048xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 128], [16, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 2, 1]>
//      CHECK: func.func @matmul_1024x2048x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Small matmul N that can still tile to all threads in a workgroup.

func.func @matmul_3136x24x96(%3: tensor<3136x96xf32>, %4: tensor<96x24xf32>) -> tensor<3136x24xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<3136x24xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<3136x24xf32>) -> tensor<3136x24xf32>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<3136x96xf32>, tensor<96x24xf32>) outs(%6 : tensor<3136x24xf32>) -> tensor<3136x24xf32>
  return %7 : tensor<3136x24xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[448, 8], [14, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 32, 1]>
//      CHECK: func.func @matmul_3136x24x96(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Small matmul M that can still tile to all threads in a workgroup.

func.func @matmul_196x64x192(%3: tensor<196x192xf32>, %4: tensor<192x64xf32>) -> tensor<196x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<196x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<196x64xf32>) -> tensor<196x64xf32>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<196x192xf32>, tensor<192x64xf32>) outs(%6 : tensor<196x64xf32>) -> tensor<196x64xf32>
  return %7 : tensor<196x64xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[28, 64], [7, 4], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [16, 4, 1]>
//      CHECK: func.func @matmul_196x64x192(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:      lowering_config = #[[CONFIG]]

// -----

// Small matmul K that can still tile to all threads in a workgroup.

func.func @matmul_12544x96x16(%0: memref<12544x16xf32>, %1: memref<16x96xf32>, %2: memref<12544x96xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%2 : memref<12544x96xf32>)
  linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%0, %1 : memref<12544x16xf32>, memref<16x96xf32>) outs(%2 : memref<12544x96xf32>)
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 32], [16, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 8, 1]>
//      CHECK: func.func @matmul_12544x96x16(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Odd matmul M and small N that cannot utilize all threads in a workgroup.

func.func @matmul_49x160x576(%3: tensor<49x576xf32>, %4: tensor<576x160xf32>) -> tensor<49x160xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<49x160xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<49x160xf32>) -> tensor<49x160xf32>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<49x576xf32>, tensor<576x160xf32>) outs(%6 : tensor<49x160xf32>) -> tensor<49x160xf32>
  return %7 : tensor<49x160xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[7, 32], [7, 4], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 1, 1]>
//      CHECK: func.func @matmul_49x160x576(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Large batch matmul.

func.func @batch_matmul_4x384x384(%3: tensor<4x384x32xf32>, %4: tensor<4x32x384xf32>) -> tensor<4x384x384xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<4x384x384xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x384x384xf32>) -> tensor<4x384x384xf32>
  %7 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<4x384x32xf32>, tensor<4x32x384xf32>) outs(%6 : tensor<4x384x384xf32>) -> tensor<4x384x384xf32>
  return %7 : tensor<4x384x384xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 32, 128], [1, 16, 4], [0, 0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 2, 1]>
//      CHECK: func.func @batch_matmul_4x384x384(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Small batch matmul.

func.func @batch_matmul_4x8x8(%3: tensor<4x8x32xf32>, %4: tensor<4x32x8xf32>) -> tensor<4x8x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<4x8x8xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x8x8xf32>) -> tensor<4x8x8xf32>
  %7 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%6 : tensor<4x8x8xf32>) -> tensor<4x8x8xf32>
  return %7 : tensor<4x8x8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 8, 8], [1, 1, 4], [0, 0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 8, 1]>
//      CHECK: func.func @batch_matmul_4x8x8(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
