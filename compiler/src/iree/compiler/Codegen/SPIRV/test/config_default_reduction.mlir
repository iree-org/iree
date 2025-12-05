// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [16], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @subgroup_reduce_f32(%2: tensor<2x512xf32>) -> tensor<2xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<2xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %out, %in : f32
    linalg.yield %6 : f32
  } -> tensor<2xf32>
  return %5 : tensor<2xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [128, 1, 1]>
//      CHECK: func.func @subgroup_reduce_f32(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @subgroup_reduce_f16(%2: tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f16
  %3 = tensor.empty() : tensor<16x4096x4096xf16>
  %4 = tensor.empty() : tensor<16x4096xf16>
  %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<16x4096xf16>) -> tensor<16x4096xf16>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<16x4096x4096xf16>) outs(%5 : tensor<16x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %8 = arith.addf %in, %out : f16
    linalg.yield %8 : f16
  } -> tensor<16x4096xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %6 : tensor<16x4096x4096xf16>, tensor<16x4096xf16>) outs(%3 : tensor<16x4096x4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.divf %in, %in_0 : f16
    linalg.yield %8 : f16
  } -> tensor<16x4096x4096xf16>
  return %7 : tensor<16x4096x4096xf16>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//      CHECK: func.func @subgroup_reduce_f16(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 64]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @subgroup_reduce_dynamic(%10: tensor<8x?xf32>) -> tensor<8xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %11 = tensor.empty() : tensor<8xf32>
  %12 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%11 : tensor<8xf32>) -> tensor<8xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<8x?xf32>) outs(%12 : tensor<8xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %14 = math.powf %in, %cst_0 : f32
    %15 = arith.addf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<8xf32>
  return %13 : tensor<8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//      CHECK: func.func @subgroup_reduce_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// The first two functions verify workgroup_size is limited to subgroup_size
// when the consumer's broadcast dimensions can't be distributed.
// The third function verifies workgroup_size is not limited to subgroup_size
// when the consumer's broadcast dimensions can be distributed.
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [32], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> ()>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map8 = affine_map<(d0, d1) -> ()>

func.func @reduction_with_elementwise_consumer(
    %input: tensor<6144xf32>,
    %other: tensor<64x3x32xf32>,
    %filled: tensor<f32>,
    %empty_out: tensor<64x3x32xf32>
) -> tensor<64x3x32xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %reduction = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction"]
  } ins(%input : tensor<6144xf32>) outs(%filled : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<f32>
  %epilogue = linalg.generic {
    indexing_maps = [#map2, #map3, #map2],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%other, %reduction : tensor<64x3x32xf32>, tensor<f32>)
    outs(%empty_out : tensor<64x3x32xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.addf %in, %in_reduction : f32
    linalg.yield %0 : f32
  } -> tensor<64x3x32xf32>
  return %epilogue : tensor<64x3x32xf32>
}

func.func @batch_reduction_and_elementwise_consumer(
    %input: tensor<128x6144xf32>,
    %other: tensor<128x64x3x32xf32>,
    %filled: tensor<128xf32>,
    %empty_out: tensor<128x64x3x32xf32>
) -> tensor<128x64x3x32xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %reduction = linalg.generic {
    indexing_maps = [#map4, #map5],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<128x6144xf32>) outs(%filled : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<128xf32>
  %epilogue = linalg.generic {
    indexing_maps = [#map6, #map7, #map6],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%other, %reduction : tensor<128x64x3x32xf32>, tensor<128xf32>)
    outs(%empty_out : tensor<128x64x3x32xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.addf %in, %in_reduction : f32
    linalg.yield %0 : f32
  } -> tensor<128x64x3x32xf32>
  return %epilogue : tensor<128x64x3x32xf32>
}

func.func @reduction_with_distributable_elementwise_consumer(
    %input: tensor<6144xf32>,
    %other: tensor<512x12xf32>,
    %filled: tensor<f32>,
    %empty_out: tensor<512x12xf32>
) -> tensor<512x12xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %reduction = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction"]
  } ins(%input : tensor<6144xf32>) outs(%filled : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<f32>
  %epilogue = linalg.generic {
    indexing_maps = [#map4, #map8, #map4],
    iterator_types = ["parallel", "parallel"]
  } ins(%other, %reduction : tensor<512x12xf32>, tensor<f32>)
    outs(%empty_out : tensor<512x12xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.addf %in, %in_reduction : f32
    linalg.yield %0 : f32
  } -> tensor<512x12xf32>
  return %epilogue : tensor<512x12xf32>
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [128]{{\]}}>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 128]{{\]}}>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [2048]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION1:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [32, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION2:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [512, 1, 1]>
//      CHECK: func.func @reduction_with_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK: func.func @batch_reduction_and_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK: func.func @reduction_with_distributable_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3]]
