// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [16], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @copy_as_generic(%2: memref<?x?xi32>, %3: memref<?x?xi32>) attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<?x?xi32>) outs(%3 : memref<?x?xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 16], [1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [16, 1, 1]>
//      CHECK: func.func @copy_as_generic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @copy(%0: memref<1x224x224x3xf32>, %1: memref<1x224x224x3xf32>) attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : memref<1x224x224x3xf32>) outs(%1 : memref<1x224x224x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 2, 32, 1], [0, 1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [1, 32, 2]>
//      CHECK: func.func @copy(
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Average pooling op with nice tilable input.

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @avg_pool(%3: tensor<1x24x24x8xf32>) -> tensor<1x2x2x8xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<12x12xf32>
  %4 = tensor.empty() : tensor<1x2x2x8xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  %6 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<12> : vector<2xi64>} ins(%3, %2 : tensor<1x24x24x8xf32>, tensor<12x12xf32>) outs(%5 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  return %6 : tensor<1x2x2x8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 2, 2, 8], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 2, 2]>
//      CHECK: func.func @avg_pool(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [4], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @avg_pool(%2: tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 4.900000e+01 : f32
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = tensor.empty() : tensor<1x1x1x1280xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %6 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%2, %3 : tensor<1x7x7x1280xf32>, tensor<7x7xf32>) outs(%5 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x1x1x1280xf32>) outs(%4 : tensor<1x1x1x1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.divf %in, %cst_0 : f32
    linalg.yield %8 : f32
  } -> tensor<1x1x1x1280xf32>
  return %7 : tensor<1x1x1x1280xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 1, 128], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//      CHECK: func.func @avg_pool(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Max pooling op with odd size-1 dimension sizes.

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @max_pool(%3: tensor<1x76x1x1xf32>) -> tensor<1x38x1x1xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0xFF800000 : f32
  %2 = tensor.empty() : tensor<2x1xf32>
  %4 = tensor.empty() : tensor<1x38x1x1xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x38x1x1xf32>) -> tensor<1x38x1x1xf32>
  %6 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<[2, 1]> : vector<2xi64>} ins(%3, %2 : tensor<1x76x1x1xf32>, tensor<2x1xf32>) outs(%5 : tensor<1x38x1x1xf32>) -> tensor<1x38x1x1xf32>
  return %6 : tensor<1x38x1x1xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 32], [0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [32, 1, 1]>
//      CHECK: func.func @max_pool(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pooling_nhwc_max
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Element wise op with mismatched input and output rank.

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func.func @elementwise(%3: tensor<1x10xf32>, %4: tensor<10xf32>) -> tensor<10xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %5 = tensor.empty() : tensor<10xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<1x10xf32>, tensor<10xf32>) outs(%5 : tensor<10xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<10xf32>
  return %6 : tensor<10xf32>
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [32, 1, 1]>
//      CHECK: func.func @elementwise(
// CHECK-SAME:   translation_info = #[[TRANSLATION]]

// -----

// Fused depthwise convolution and element wise ops: don't vectorize with partially active subgroups.

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
func.func @dwconv_elementwise(%3: tensor<1x21x20x1xf32>) -> tensor<1x19x18x1x4xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant dense_resource<__elided__> : tensor<3x3x1x4xf32>
  %cst_0 = arith.constant 1.001000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<1x19x18x1x4xf32>
  %4 = tensor.empty() : tensor<1x19x18x1x4xf32>
  %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<1x19x18x1x4xf32>) -> tensor<1x19x18x1x4xf32>
  %6 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %cst : tensor<1x21x20x1xf32>, tensor<3x3x1x4xf32>) outs(%5 : tensor<1x19x18x1x4xf32>) -> tensor<1x19x18x1x4xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x19x18x1x4xf32>) outs(%2 : tensor<1x19x18x1x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = math.sqrt %cst_0 : f32
    %9 = arith.addf %in, %cst_1 : f32
    linalg.yield %9 : f32
  } -> tensor<1x19x18x1x4xf32>
  return %7 : tensor<1x19x18x1x4xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 2, 0, 4], [0, 1, 1, 0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [4, 2, 4]>
//      CHECK: func.func @dwconv_elementwise(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @outermost_reduction(%2: tensor<4x2048x512xf32>) -> tensor<2048x512xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<2048x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2048x512xf32>) -> tensor<2048x512xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<4x2048x512xf32>) outs(%4 : tensor<2048x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<2048x512xf32>
  return %5 : tensor<2048x512xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 128], [1, 4],  [0, 0, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//      CHECK: func.func @outermost_reduction(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @innermost_reduction(%9: tensor<128x384xf32>, %10: tensor<128xf32>) -> tensor<128xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant -0.000000e+00 : f32
  %11 = tensor.empty() : tensor<128xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<128xf32>) -> tensor<128xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%9, %10 : tensor<128x384xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %14 = arith.subf %in, %in_0 : f32
    %15 = arith.mulf %14, %14 : f32
    %16 = arith.addf %15, %out : f32
    linalg.yield %16 : f32
  } -> tensor<128xf32>
  return %13 : tensor<128xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32], [1],  [0, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//      CHECK: func.func @innermost_reduction(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [16], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @four_dim_elementwise(%2: tensor<128x8x256x4xf32>) -> tensor<128x256x4x8xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %3 = tensor.empty() : tensor<128x256x4x8xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<128x8x256x4xf32>) outs(%3 : tensor<128x256x4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<128x256x4x8xf32>
  return %4 : tensor<128x256x4x8xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 2, 4, 8], [0, 1, 1, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 4, 2]>
//      CHECK: func.func @four_dim_elementwise(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @odd_reduction_dimension_size_501(%2: tensor<512x501xf32>) -> tensor<512x501xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0xFF800000 : f32
  %3 = tensor.empty() : tensor<512x501xf32>
  %4 = tensor.empty() : tensor<512xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<512xf32>) -> tensor<512xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<512x501xf32>) outs(%5 : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.maximumf %out, %in : f32
    linalg.yield %8 : f32
  } -> tensor<512xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %6 : tensor<512x501xf32>, tensor<512xf32>) outs(%3 : tensor<512x501xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.subf %in, %in_0 : f32
    %9 = math.exp %8 : f32
    linalg.yield %9 : f32
  } -> tensor<512x501xf32>
  return %7 : tensor<512x501xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128], [4],  [0, 3]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//      CHECK: func.func @odd_reduction_dimension_size_501(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @odd_reduction_dimension_size_2809(%2: tensor<512x2809xf32>) -> tensor<512x2809xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0xFF800000 : f32
  %3 = tensor.empty() : tensor<512x2809xf32>
  %4 = tensor.empty() : tensor<512xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<512xf32>) -> tensor<512xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<512x2809xf32>) outs(%5 : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.maximumf %out, %in : f32
    linalg.yield %8 : f32
  } -> tensor<512xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %6 : tensor<512x2809xf32>, tensor<512xf32>) outs(%3 : tensor<512x2809xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %8 = arith.subf %in, %in_0 : f32
    %9 = math.exp %8 : f32
    linalg.yield %9 : f32
  } -> tensor<512x2809xf32>
  return %7 : tensor<512x2809xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128], [4],  [0, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//      CHECK: func.func @odd_reduction_dimension_size_2809(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @broadcast(%2: tensor<f32>) -> tensor<2048x1x1x1xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 1.000000e-10 : f32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %3 = tensor.empty() : tensor<2048x1x1x1xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<f32>) outs(%3 : tensor<2048x1x1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.maximumf %in, %cst : f32
    %6 = arith.divf %cst_0, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<2048x1x1x1xf32>
  return %4 : tensor<2048x1x1x1xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128], [4],  [0, 1, 1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//      CHECK: func.func @broadcast(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
