// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle, dot = none, mma = [],
    subgroup_size_choices = [16], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @subgroup_reduce_f32() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
  %3 = tensor.empty() : tensor<2xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %out, %in : f32
    linalg.yield %6 : f32
  } -> tensor<2xf32>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [128, 1, 1]>
//      CHECK: func.func @subgroup_reduce_f32()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle, dot = none, mma = [],
    subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @subgroup_reduce_f16() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x4096x4096xf16>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf16>> -> tensor<16x4096x4096xf16>
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
  flow.dispatch.tensor.store %7, %1, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : tensor<16x4096x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x4096x4096xf16>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//      CHECK: func.func @subgroup_reduce_f16()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 64]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle, dot = none, mma = [],
    subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @subgroup_reduce_dynamic() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.extui %0 : i32 to i64
  %3 = arith.extui %1 : i32 to i64
  %4 = arith.shli %3, %c32_i64 : i64
  %5 = arith.ori %2, %4 : i64
  %6 = arith.index_castui %5 : i64 to index
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
  %8 = flow.dispatch.workload.ordinal %6, 0 : index
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x?xf32>>{%8}
  %10 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [8, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x?xf32>>{%8} -> tensor<8x?xf32>
  %11 = tensor.empty() : tensor<8xf32>
  %12 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%11 : tensor<8xf32>) -> tensor<8xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<8x?xf32>) outs(%12 : tensor<8xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %14 = math.powf %in, %cst_0 : f32
    %15 = arith.addf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<8xf32>
  flow.dispatch.tensor.store %13, %7, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//      CHECK: func.func @subgroup_reduce_dynamic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]
