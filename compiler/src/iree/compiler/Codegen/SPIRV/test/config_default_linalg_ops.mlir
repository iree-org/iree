// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [16], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @copy_as_generic() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : memref<?x?xi32>{%0, %1}
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : memref<?x?xi32>{%0, %1}
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<?x?xi32>) outs(%3 : memref<?x?xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 16], [1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [16, 1, 1] subgroup_size = 16>
//      CHECK: func.func @copy_as_generic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [64], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @copy() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c224 = arith.constant 224 : index
  %c3 = arith.constant 3 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : memref<1x224x224x3xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : memref<1x224x224x3xf32>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : memref<1x224x224x3xf32>) outs(%1 : memref<1x224x224x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 2, 32, 1], [0, 1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [1, 32, 2] subgroup_size = 64>
//      CHECK: func.func @copy()
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Average pooling op with nice tilable input.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
func.func @avg_pool() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x24x24x8xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<writeonly:tensor<1x2x2x8xf32>>
  %2 = tensor.empty() : tensor<12x12xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 24, 24, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x24x24x8xf32>> -> tensor<1x24x24x8xf32>
  %4 = tensor.empty() : tensor<1x2x2x8xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  %6 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<12> : vector<2xi64>} ins(%3, %2 : tensor<1x24x24x8xf32>, tensor<12x12xf32>) outs(%5 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0, 0], sizes = [1, 2, 2, 8], strides = [1, 1, 1, 1] : tensor<1x2x2x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x2x2x8xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 2, 2, 8], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [2, 2, 2] subgroup_size = 32>
//      CHECK: func.func @avg_pool()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [4], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @avg_pool() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 4.900000e+01 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x7x7x1280xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1x1x1280xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 7, 7, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x7x7x1280xf32>> -> tensor<1x7x7x1280xf32>
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = tensor.empty() : tensor<1x1x1x1280xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %6 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%2, %3 : tensor<1x7x7x1280xf32>, tensor<7x7xf32>) outs(%5 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x1x1x1280xf32>) outs(%4 : tensor<1x1x1x1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.divf %in, %cst_0 : f32
    linalg.yield %8 : f32
  } -> tensor<1x1x1x1280xf32>
  flow.dispatch.tensor.store %7, %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 1280], strides = [1, 1, 1, 1] : tensor<1x1x1x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1x1x1280xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 1, 128], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CHECK: func.func @avg_pool()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Max pooling op with odd size-1 dimension sizes.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
func.func @max_pool() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0xFF800000 : f32
  %c38 = arith.constant 38 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c320 = arith.constant 320 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x76x1x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<writeonly:tensor<1x38x1x1xf32>>
  %2 = tensor.empty() : tensor<2x1xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 76, 1, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x76x1x1xf32>> -> tensor<1x76x1x1xf32>
  %4 = tensor.empty() : tensor<1x38x1x1xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x38x1x1xf32>) -> tensor<1x38x1x1xf32>
  %6 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<[2, 1]> : vector<2xi64>} ins(%3, %2 : tensor<1x76x1x1xf32>, tensor<2x1xf32>) outs(%5 : tensor<1x38x1x1xf32>) -> tensor<1x38x1x1xf32>
  flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0, 0], sizes = [1, 38, 1, 1], strides = [1, 1, 1, 1] : tensor<1x38x1x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x38x1x1xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 32], [0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CHECK: func.func @max_pool()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pooling_nhwc_max
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Element wise op with mismatched input and output rank.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func.func @elementwise() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x10xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<readonly:tensor<10xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) : !flow.dispatch.tensor<writeonly:tensor<10xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x10xf32>> -> tensor<1x10xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [10], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10xf32>> -> tensor<10xf32>
  %5 = tensor.empty() : tensor<10xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<1x10xf32>, tensor<10xf32>) outs(%5 : tensor<10xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<10xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<writeonly:tensor<10xf32>>
  return
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CHECK: func.func @elementwise()
// CHECK-SAME:   translation_info = #[[TRANSLATION]]

// -----

// Fused depthwise convolution and element wise ops: don't vectorize with partially active subgroups.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
func.func @dwconv_elementwise() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant dense_resource<__elided__> : tensor<3x3x1x4xf32>
  %cst_0 = arith.constant 1.001000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c18 = arith.constant 18 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4576 = arith.constant 4576 : index
  %c6272 = arith.constant 6272 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x21x20x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<writeonly:tensor<1x19x18x1x4xf32>>
  %2 = tensor.empty() : tensor<1x19x18x1x4xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 21, 20, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x21x20x1xf32>> -> tensor<1x21x20x1xf32>
  %4 = tensor.empty() : tensor<1x19x18x1x4xf32>
  %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<1x19x18x1x4xf32>) -> tensor<1x19x18x1x4xf32>
  %6 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %cst : tensor<1x21x20x1xf32>, tensor<3x3x1x4xf32>) outs(%5 : tensor<1x19x18x1x4xf32>) -> tensor<1x19x18x1x4xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x19x18x1x4xf32>) outs(%2 : tensor<1x19x18x1x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = math.sqrt %cst_0 : f32
    %9 = arith.addf %in, %cst_1 : f32
    linalg.yield %9 : f32
  } -> tensor<1x19x18x1x4xf32>
  flow.dispatch.tensor.store %7, %1, offsets = [0, 0, 0, 0, 0], sizes = [1, 19, 18, 1, 4], strides = [1, 1, 1, 1, 1] : tensor<1x19x18x1x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x19x18x1x4xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 2, 0, 4], [0, 1, 1, 0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [4, 2, 4] subgroup_size = 32>
//      CHECK: func.func @dwconv_elementwise()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @outermost_reduction() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x2048x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 2048, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x2048x512xf32>> -> tensor<4x2048x512xf32>
  %3 = tensor.empty() : tensor<2048x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2048x512xf32>) -> tensor<2048x512xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<4x2048x512xf32>) outs(%4 : tensor<2048x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<2048x512xf32>
  flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : tensor<2048x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 128], [1, 4],  [0, 0, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1] subgroup_size = 32>
//       CHECK: func.func @outermost_reduction()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @innermost_reduction() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant -0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = arith.index_cast %0 {stream.alignment = 512 : index, stream.values = [0 : index, 394752 : index, 984064 : index]} : i32 to index
  %4 = arith.index_cast %1 {stream.alignment = 512 : index, stream.values = [0 : index, 196608 : index, 197120 : index]} : i32 to index
  %5 = arith.index_cast %2 {stream.alignment = 512 : index, stream.values = [512 : index, 197120 : index, 197632 : index]} : i32 to index
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%3) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%4) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<128xf32>>
  %9 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<128x384xf32>
  %10 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
  %11 = tensor.empty() : tensor<128xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<128xf32>) -> tensor<128xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%9, %10 : tensor<128x384xf32>, tensor<128xf32>) outs(%12 : tensor<128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %14 = arith.subf %in, %in_0 : f32
    %15 = arith.mulf %14, %14 : f32
    %16 = arith.addf %15, %out : f32
    linalg.yield %16 : f32
  } -> tensor<128xf32>
  flow.dispatch.tensor.store %13, %8, offsets = [0], sizes = [128], strides = [1] : tensor<128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32], [1],  [0, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1] subgroup_size = 32>
//       CHECK: func.func @innermost_reduction()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [16], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @four_dim_elementwise() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x8x256x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x256x4x8xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 8, 256, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x8x256x4xf32>> -> tensor<128x8x256x4xf32>
  %3 = tensor.empty() : tensor<128x256x4x8xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<128x8x256x4xf32>) outs(%3 : tensor<128x256x4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<128x256x4x8xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [128, 256, 4, 8], strides = [1, 1, 1, 1] : tensor<128x256x4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x256x4x8xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 2, 4, 8], [0, 1, 1, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [2, 4, 2] subgroup_size = 16>
//       CHECK: func.func @four_dim_elementwise()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @odd_reduction_dimension_size_501() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0xFF800000 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x501xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<512x501xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 501], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x501xf32>> -> tensor<512x501xf32>
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
  flow.dispatch.tensor.store %7, %1, offsets = [0, 0], sizes = [512, 501], strides = [1, 1] : tensor<512x501xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x501xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128], [4],  [0, 3]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1] subgroup_size = 32>
//       CHECK: func.func @odd_reduction_dimension_size_501()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @odd_reduction_dimension_size_2809() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0xFF800000 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x2809xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<512x2809xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 2809], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x2809xf32>> -> tensor<512x2809xf32>
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
  flow.dispatch.tensor.store %7, %1, offsets = [0, 0], sizes = [512, 2809], strides = [1, 1] : tensor<512x2809xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x2809xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128], [4],  [0, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1] subgroup_size = 32>
//       CHECK: func.func @odd_reduction_dimension_size_2809()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384>>
}>
#map = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @broadcast() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e-10 : f32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<f32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x1x1x1xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
  %3 = tensor.empty() : tensor<2048x1x1x1xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<f32>) outs(%3 : tensor<2048x1x1x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.maximumf %in, %cst : f32
    %6 = arith.divf %cst_0, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<2048x1x1x1xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2048, 1, 1, 1], strides = [1, 1, 1, 1] : tensor<2048x1x1x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x1x1x1xf32>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128], [4],  [0, 1, 1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1] subgroup_size = 32>
//       CHECK: func.func @broadcast()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
