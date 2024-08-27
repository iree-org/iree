// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Odd K that forbids vectorization.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @batch_matmul_1x3x32() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x3x3xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<1x3x32xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<1x3x32xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 3, 3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x3x3xf32>> -> tensor<1x3x3xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 3, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x3x32xf32>> -> tensor<1x3x32xf32>
  %5 = tensor.empty() : tensor<1x3x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x3x32xf32>) -> tensor<1x3x32xf32>
  %7 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<1x3x3xf32>, tensor<1x3x32xf32>) outs(%6 : tensor<1x3x32xf32>) -> tensor<1x3x32xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [1, 3, 32], strides = [1, 1, 1] : tensor<1x3x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x3x32xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 32], [0, 1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [32, 1, 1]>
//       CHECK: func.func @batch_matmul_1x3x32()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// 8-bit integers can be vectorized.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @matmul_64x16xi8() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<64x32xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<32x16xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<64x16xi32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32xi8>> -> tensor<64x32xi8>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x16xi8>> -> tensor<32x16xi8>
  %5 = tensor.empty() : tensor<64x16xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<64x16xi32>) -> tensor<64x16xi32>
  %7 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%3, %4 : tensor<64x32xi8>, tensor<32x16xi8>) outs(%6 : tensor<64x16xi32>) -> tensor<64x16xi32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 16], strides = [1, 1] : tensor<64x16xi32> -> !flow.dispatch.tensor<writeonly:tensor<64x16xi32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 16], [2, 8], [0, 0, 8]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [2, 32, 1]>
//       CHECK: func.func @matmul_64x16xi8()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Vectorize non-32 bit types.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int64|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @matmul_64x16xi64() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<64x32xi64>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<32x16xi64>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<64x16xi64>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32xi64>> -> tensor<64x32xi64>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x16xi64>> -> tensor<32x16xi64>
  %5 = tensor.empty() : tensor<64x16xi64>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<64x16xi64>) -> tensor<64x16xi64>
  %7 = linalg.matmul ins(%3, %4 : tensor<64x32xi64>, tensor<32x16xi64>) outs(%6 : tensor<64x16xi64>) -> tensor<64x16xi64>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 16], strides = [1, 1] : tensor<64x16xi64> -> !flow.dispatch.tensor<writeonly:tensor<64x16xi64>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 16], [1, 4], [0, 0, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [4, 16, 1]>
//       CHECK: func.func @matmul_64x16xi64()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Odd N that forbids vectorization.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_400x273() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c11775744 = arith.constant 11775744 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c400 = arith.constant 400 : index
  %c273 = arith.constant 273 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c11775744) : !flow.dispatch.tensor<readonly:tensor<273xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<400x576xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<576x273xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<400x273xf32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [273], strides = [1] : !flow.dispatch.tensor<readonly:tensor<273xf32>> -> tensor<273xf32>
  %5 = tensor.empty() : tensor<400x273xf32>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [400, 576], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<400x576xf32>> -> tensor<400x576xf32>
  %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [576, 273], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<576x273xf32>> -> tensor<576x273xf32>
  %8 = tensor.empty() : tensor<400x273xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<400x273xf32>) -> tensor<400x273xf32>
  %10 = linalg.matmul ins(%6, %7 : tensor<400x576xf32>, tensor<576x273xf32>) outs(%9 : tensor<400x273xf32>) -> tensor<400x273xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %10 : tensor<273xf32>, tensor<400x273xf32>) outs(%5 : tensor<400x273xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.addf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<400x273xf32>
  flow.dispatch.tensor.store %11, %3, offsets = [0, 0], sizes = [400, 273], strides = [1, 1] : tensor<400x273xf32> -> !flow.dispatch.tensor<writeonly:tensor<400x273xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[2, 32], [1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [32, 2, 1]>
//       CHECK: func.func @matmul_400x273()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Odd M and non-4-multiplier N

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_25x546() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c15842560 = arith.constant 15842560 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c25 = arith.constant 25 : index
  %c546 = arith.constant 546 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c15842560) : !flow.dispatch.tensor<readonly:tensor<546xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<25x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<512x546xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<25x546xf32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [546], strides = [1] : !flow.dispatch.tensor<readonly:tensor<546xf32>> -> tensor<546xf32>
  %5 = tensor.empty() : tensor<25x546xf32>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [25, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<25x512xf32>> -> tensor<25x512xf32>
  %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [512, 546], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x546xf32>> -> tensor<512x546xf32>
  %8 = tensor.empty() : tensor<25x546xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<25x546xf32>) -> tensor<25x546xf32>
  %10 = linalg.matmul ins(%6, %7 : tensor<25x512xf32>, tensor<512x546xf32>) outs(%9 : tensor<25x546xf32>) -> tensor<25x546xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %10 : tensor<546xf32>, tensor<25x546xf32>) outs(%5 : tensor<25x546xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.addf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<25x546xf32>
  flow.dispatch.tensor.store %11, %3, offsets = [0, 0], sizes = [25, 546], strides = [1, 1] : tensor<25x546xf32> -> !flow.dispatch.tensor<writeonly:tensor<25x546xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 2], [1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [2, 32, 1]>
//       CHECK: func.func @matmul_25x546()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Matmul with consumer pointwise ops

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_pointwise_256x1024() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
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
    %15 = arith.subf %14, %in_1 : f16
    linalg.yield %15 : f16
  } -> tensor<256x1024xf16>
  flow.dispatch.tensor.store %13, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 256], [8, 8], [0, 0, 8]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 2, 1]>
//       CHECK: func.func @matmul_pointwise_256x1024()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
