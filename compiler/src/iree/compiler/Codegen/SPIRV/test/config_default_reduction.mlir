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
func.func @subgroup_reduce_f32(%arg0: tensor<2x512xf32>) -> tensor<2xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<2x512xf32>) outs(%1 : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %out, %in : f32
    linalg.yield %3 : f32
  } -> tensor<2xf32>
  return %2 : tensor<2xf32>
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
func.func @subgroup_reduce_f16(%arg0: tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<16x4096x4096xf16>
  %1 = tensor.empty() : tensor<16x4096xf16>
  %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<16x4096xf16>) -> tensor<16x4096xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<16x4096x4096xf16>) outs(%2 : tensor<16x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.addf %in, %out : f16
    linalg.yield %5 : f16
  } -> tensor<16x4096xf16>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %3 : tensor<16x4096x4096xf16>, tensor<16x4096xf16>) outs(%0 : tensor<16x4096x4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %5 = arith.divf %in, %in_0 : f16
    linalg.yield %5 : f16
  } -> tensor<16x4096x4096xf16>
  return %4 : tensor<16x4096x4096xf16>
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
func.func @subgroup_reduce_dynamic(%arg0: tensor<8x?xf32>) -> tensor<8xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %0 = tensor.empty() : tensor<8xf32>
  %1 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<8x?xf32>) outs(%1 : tensor<8xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %3 = math.powf %in, %cst_0 : f32
    %4 = arith.addf %3, %out : f32
    linalg.yield %4 : f32
  } -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//      CHECK: func.func @subgroup_reduce_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]
