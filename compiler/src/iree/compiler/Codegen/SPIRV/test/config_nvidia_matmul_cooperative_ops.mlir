// RUN: iree-opt --split-input-file --iree-gpu-test-target=volta@vulkan \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s |  \
// RUN:   FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_256x1024x128_div_add() {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<readonly:tensor<256x128xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) : !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
  %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
  %7 = tensor.empty() : tensor<256x1024xf16>
  %8 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xf16>> -> tensor<256x128xf16>
  %9 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>> -> tensor<128x1024xf16>
  %10 = tensor.empty() : tensor<256x1024xf16>
  %11 = linalg.fill ins(%cst : f16) outs(%10 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %12 = linalg.matmul ins(%8, %9 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%11 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %13 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %5, %6 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>) outs(%7 : tensor<256x1024xf16>) {
  ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
    %14 = arith.divf %in, %in_0 : f16
    %15 = arith.addf %14, %in_1 : f16
    linalg.yield %15 : f16
  } -> tensor<256x1024xf16>
  flow.dispatch.tensor.store %13, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
  return
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [32, 32], [0, 0, 32], [16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 1 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @matmul_256x1024x128_div_add()
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @batch_matmul_16x128x256x512_div() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 128, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>> -> tensor<16x128x512xf16>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 512, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>> -> tensor<16x512x256xf16>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>> -> tensor<16x128x256xf16>
  %7 = tensor.empty() : tensor<16x128x256xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
  %9 = linalg.batch_matmul ins(%4, %5 : tensor<16x128x512xf16>, tensor<16x512x256xf16>) outs(%8 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
  %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<16x128x256xf16>, tensor<16x128x256xf16>) outs(%7 : tensor<16x128x256xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %11 = arith.divf %in, %in_0 : f16
    linalg.yield %11 : f16
  } -> tensor<16x128x256xf16>
  flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : tensor<16x128x256xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
  return
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 64], [1, 32, 32], [0, 0, 0, 32], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 1 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @batch_matmul_16x128x256x512_div()
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Linalg.generic that is a batch matmul.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @generic_batch_matmul_32x8x512x64() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x32x64xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x64x512xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 32, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x64xf16>> -> tensor<128x32x64xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [32, 64, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x64x512xf16>> -> tensor<32x64x512xf16>
  %5 = tensor.empty() : tensor<32x128x512xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32x128x512xf16>) -> tensor<32x128x512xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<128x32x64xf16>, tensor<32x64x512xf16>) outs(%6 : tensor<32x128x512xf16>) attrs =  {linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %out, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<32x128x512xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [32, 128, 512], strides = [1, 1, 1] : tensor<32x128x512xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
  return
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 64], [1, 32, 32], [0, 0, 0, 32], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 1 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @generic_batch_matmul_32x8x512x64()
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// K dim size not divisble by 32.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @batch_matmul_16x1024x1024x80() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x1024x80xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x80x1024xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x1024x1024xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 1024, 80], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024x80xf16>> -> tensor<16x1024x80xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 80, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x80x1024xf16>> -> tensor<16x80x1024xf16>
  %5 = tensor.empty() : tensor<16x1024x1024xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<16x1024x1024xf16>) -> tensor<16x1024x1024xf16>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<16x1024x80xf16>, tensor<16x80x1024xf16>) outs(%6 : tensor<16x1024x1024xf16>) -> tensor<16x1024x1024xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [16, 1024, 1024], strides = [1, 1, 1] : tensor<16x1024x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x1024xf16>>
  return
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 64], [1, 32, 32], [0, 0, 0, 16], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 0 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @batch_matmul_16x1024x1024x80()
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Small K - not supported by cooperative matrix.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_256x1024x8() {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<256x8xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<8x1024xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x8xf16>> -> tensor<256x8xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x1024xf16>> -> tensor<8x1024xf16>
  %5 = tensor.empty() : tensor<256x1024xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<256x8xf16>, tensor<8x1024xf16>) outs(%6 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
  return
}

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [32, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//       CHECK: func.func @matmul_256x1024x8
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
