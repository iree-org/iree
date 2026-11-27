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
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce> workgroup_size = [128, 1, 1]>
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
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce> workgroup_size = [64, 1, 1]>
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
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce> workgroup_size = [64, 1, 1]>
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

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [256]{{\]}}>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 256]{{\]}}>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [2048]{{\]}}>
//  CHECK-DAG: #[[CONFIG1_FUSED:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [256, 0, 0]{{\]}}>
//  CHECK-DAG: #[[CONFIG2_FUSED:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [0, 256, 0, 0]{{\]}}>
//  CHECK-DAG: #[[CONFIG3_FUSED:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [2048, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION1:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce> workgroup_size = [64, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION2:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce> workgroup_size = [512, 1, 1]>
// In each of these the reduction op keeps its workgroup level while the fused
// consumer gets the reduction level only, so the consumer cannot outrank the
// reduction during workgroup root selection.
//      CHECK: func.func @reduction_with_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1_FUSED]]
//      CHECK: func.func @batch_reduction_and_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2_FUSED]]
//      CHECK: func.func @reduction_with_distributable_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3_FUSED]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [32], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map9 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map12 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map13 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @fail_reduction_with_nondistributable_consumer(
    %input: tensor<16x64x74xf32>,
    %filled: tensor<16x74xf32>,
    %empty_out: tensor<16x74x64xf32>
) -> tensor<16x74x64xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst_eps = arith.constant 9.99999974E-5 : f32
  %cst_scale = arith.constant 8.000000e+00 : f32

  %reduction = linalg.generic {
    indexing_maps = [#map9, #map13],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%input : tensor<16x64x74xf32>) outs(%filled : tensor<16x74xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<16x74xf32>

  %epilogue = linalg.generic {
    indexing_maps = [#map10, #map11, #map12],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input, %reduction : tensor<16x64x74xf32>, tensor<16x74xf32>)
    outs(%empty_out : tensor<16x74x64xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.maximumf %in, %in_reduction : f32
    %1 = math.sqrt %0 : f32
    %2 = arith.mulf %in, %cst_scale : f32
    %3 = arith.divf %2, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<16x74x64xf32>
  return %epilogue : tensor<16x74x64xf32>
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> {{.*}}>
//      CHECK: func.func @fail_reduction_with_nondistributable_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-NOT: pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce>
//      CHECK: return
// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle|arithmetic,
    subgroup_size_choices = [16], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 32768,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map14 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map15 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map16 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map17 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map18 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_with_permuted_consumer_and_extra_parallel_dim(
    %input: tensor<16x4x64xf32>,
    %other: tensor<4x16x64xf32>,
    %filled: tensor<4x64xf32>,
    %empty_out: tensor<16x4x64xf32>
) -> tensor<16x4x64xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %reduction = linalg.generic {
    indexing_maps = [#map14, #map15],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%input : tensor<16x4x64xf32>) outs(%filled : tensor<4x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.maximumf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<4x64xf32>

  %epilogue = linalg.generic {
    indexing_maps = [#map16, #map17, #map18],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%other, %reduction : tensor<4x16x64xf32>, tensor<4x64xf32>)
    outs(%empty_out : tensor<16x4x64xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.subf %in, %in_reduction : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<16x4x64xf32>
  return %epilogue : tensor<16x4x64xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[FUSED:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce> workgroup_size = [16, 1, 1]>
//      CHECK: func.func @reduction_with_permuted_consumer_and_extra_parallel_dim(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// The reduction op owns the workgroup level...
//      CHECK:   linalg.generic
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:       lowering_config = #[[CONFIG]]
//      CHECK:   linalg.generic
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:       lowering_config = #[[FUSED]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [16], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>

// A fused producer writes an elementwise result as well as a row reduction.
// It cannot form independent partial reductions for every result.
// CHECK-DAG: #[[$FALLBACK:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute>
// CHECK-LABEL: func.func @mixed_parallel_reduction_outputs(
// CHECK-SAME: translation_info = #[[$FALLBACK]]
func.func @mixed_parallel_reduction_outputs(%x: tensor<4x?xf32>) -> tensor<4xf32>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %n = tensor.dim %x, %c1 : tensor<4x?xf32>
  %empty = tensor.empty(%n) : tensor<4x?xf32>
  %row = tensor.empty() : tensor<4xf32>
  %zeros = linalg.fill ins(%zero : f32) outs(%row : tensor<4xf32>) -> tensor<4xf32>
  %positive, %sum = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<4x?xf32>) outs(%empty, %zeros : tensor<4x?xf32>, tensor<4xf32>) {
    ^bb0(%value: f32, %unused: f32, %acc: f32):
      %p = arith.maximumf %value, %zero : f32
      %s = arith.addf %value, %acc : f32
      linalg.yield %p, %s : f32, f32
  } -> (tensor<4x?xf32>, tensor<4xf32>)
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m)>, affine_map<(m, n) -> (m)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%positive, %sum : tensor<4x?xf32>, tensor<4xf32>) outs(%zeros : tensor<4xf32>) {
    ^bb0(%value: f32, %scale: f32, %acc: f32):
      %scaled = arith.mulf %value, %scale : f32
      %s = arith.addf %scaled, %acc : f32
      linalg.yield %s : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [16], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>

// Propagated tile sizes use loop positions, so reductions with distinct
// iteration spaces must use the general pipeline even with simple combiners.
// CHECK-DAG: #[[$AXIS_FALLBACK:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute>
// CHECK-LABEL: func.func @different_reduction_iteration_spaces(
// CHECK-SAME: translation_info = #[[$AXIS_FALLBACK]]
func.func @different_reduction_iteration_spaces(%x: tensor<4x?x32xf32>) -> tensor<4xf32>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %n = tensor.dim %x, %c1 : tensor<4x?x32xf32>
  %empty = tensor.empty(%n) : tensor<4x?xf32>
  %zeros = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x?xf32>) -> tensor<4x?xf32>
  %inner = linalg.generic {
      indexing_maps = [affine_map<(m, n, k) -> (m, n, k)>, affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x : tensor<4x?x32xf32>) outs(%zeros : tensor<4x?xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<4x?xf32>
  %row = tensor.empty() : tensor<4xf32>
  %row_zeros = linalg.fill ins(%zero : f32) outs(%row : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%inner : tensor<4x?xf32>) outs(%row_zeros : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}
