// RUN: iree-opt --split-input-file --iree-gpu-test-target=rdna2@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @batch_matmul_f32_16x4096x40x4096() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x40xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x4096x40xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x4096xf32>> -> tensor<16x4096x4096xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x40xf32>> -> tensor<16x4096x40xf32>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x4096x40xf32>> -> tensor<16x4096x40xf32>
  %6 = linalg.batch_matmul ins(%3, %4 : tensor<16x4096x4096xf32>, tensor<16x4096x40xf32>) outs(%5 : tensor<16x4096x40xf32>) -> tensor<16x4096x40xf32>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : tensor<16x4096x40xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x4096x40xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 256, 8, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [2, 32, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @batch_matmul_f32_16x4096x40x4096()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_f16_64x640x320() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x320xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x640xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x640xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x320xf16>> -> tensor<64x320xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [320, 640], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x640xf16>> -> tensor<320x640xf16>
  %5 = tensor.empty() : tensor<64x640xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x640xf16>) -> tensor<64x640xf16>
  %7 = linalg.matmul ins(%3, %4 : tensor<64x320xf16>, tensor<320x640xf16>) outs(%6 : tensor<64x640xf16>) -> tensor<64x640xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 640], strides = [1, 1] : tensor<64x640xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x640xf16>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 2 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @matmul_f16_64x640x320()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @batch_matmul_f32_16x4096x40x4096() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x48xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x4096x48xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x4096xf32>> -> tensor<16x4096x4096xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 4096, 48], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096x48xf32>> -> tensor<16x4096x48xf32>
  %5 = tensor.empty() : tensor<16x4096x48xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<16x4096x4096xf32>, tensor<16x4096x48xf32>) outs(%6 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [16, 4096, 48], strides = [1, 1, 1] : tensor<16x4096x48xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x4096x48xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 128, 16, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [4, 16, 1], {pipeline_depth = 2 : i64, store_stage = 0 : i64}>
//      CHECK: func.func @batch_matmul_f32_16x4096x40x4096()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @batch_matmul_f16_1x4096x4096x512() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096x512xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x512x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096x4096xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 4096, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096x512xf16>> -> tensor<1x4096x512xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 512, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x512x4096xf16>> -> tensor<1x512x4096xf16>
  %5 = tensor.empty() : tensor<1x4096x4096xf32>
  %6 = tensor.empty() : tensor<1x4096x4096xf16>
  %7 = linalg.fill ins(%cst : f16) outs(%6 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
  %8 = linalg.batch_matmul ins(%3, %4 : tensor<1x4096x512xf16>, tensor<1x512x4096xf16>) outs(%7 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
  %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x4096x4096xf16>) outs(%5 : tensor<1x4096x4096xf32>) {
  ^bb0(%in: f16, %out: f32):
    %10 = arith.extf %in : f16 to f32
    linalg.yield %10 : f32
  } -> tensor<1x4096x4096xf32>
  iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [0, 0, 0], sizes = [1, 4096, 4096], strides = [1, 1, 1] : tensor<1x4096x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096x4096xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @batch_matmul_f16_1x4096x4096x512()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @matmul_multi_reduce_i4xf32xf32() {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
  %5 = arith.index_castui %0 : i32 to index
  %6 = arith.index_castui %1 : i32 to index
  %7 = arith.index_castui %2 : i32 to index
  %8 = arith.index_castui %3 : i32 to index
  %9 = arith.index_castui %4 : i32 to index
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%5) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11008x32x128xi4>>
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%6) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11008x32xf32>>
  %12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%7) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11008x32xf32>>
  %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%8) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x32x128xf32>>
  %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%9) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x11008xf32>>
  %15 = iree_tensor_ext.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [11008, 32, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11008x32x128xi4>> -> tensor<11008x32x128xi4>
  %16 = iree_tensor_ext.dispatch.tensor.load %11, offsets = [0, 0], sizes = [11008, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11008x32xf32>> -> tensor<11008x32xf32>
  %17 = iree_tensor_ext.dispatch.tensor.load %12, offsets = [0, 0], sizes = [11008, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11008x32xf32>> -> tensor<11008x32xf32>
  %18 = iree_tensor_ext.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [512, 32, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x32x128xf32>> -> tensor<512x32x128xf32>
  %19 = tensor.empty() : tensor<512x11008xf32>
  %20 = tensor.empty() : tensor<11008x32x128xf32>
  %21 = linalg.fill ins(%cst : f32) outs(%19 : tensor<512x11008xf32>) -> tensor<512x11008xf32>
  %22 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %16, %17 : tensor<11008x32x128xi4>, tensor<11008x32xf32>, tensor<11008x32xf32>) outs(%20 : tensor<11008x32x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %24 = arith.extui %in : i4 to i32
    %25 = arith.uitofp %24 : i32 to f32
    %26 = arith.subf %25, %in_1 : f32
    %27 = arith.mulf %26, %in_0 : f32
    linalg.yield %27 : f32
  } -> tensor<11008x32x128xf32>
  %23 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%18, %22 : tensor<512x32x128xf32>, tensor<11008x32x128xf32>) outs(%21 : tensor<512x11008xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %24 = arith.mulf %in, %in_0 : f32
    %25 = arith.addf %24, %out : f32
    linalg.yield %25 : f32
  } -> tensor<512x11008xf32>
  iree_tensor_ext.dispatch.tensor.store %23, %14, offsets = [0, 0], sizes = [512, 11008], strides = [1, 1] : tensor<512x11008xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x11008xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 1, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
//      CHECK: func.func @matmul_multi_reduce_i4xf32xf32()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]
