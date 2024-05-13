// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 64>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @i4_dequant_matvec_f32() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<86x128xf32>>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %8 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x128xf32>> -> tensor<86x128xf32>
    %9 = tensor.empty() : tensor<4096xf32>
    %10 = tensor.empty() : tensor<4096x86x128xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%9 : tensor<4096xf32>) -> tensor<4096xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%10 : tensor<4096x86x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %14 = arith.extui %in : i4 to i32
      %15 = arith.uitofp %14 : i32 to f32
      %16 = arith.subf %15, %in_1 : f32
      %17 = arith.mulf %16, %in_0 : f32
      linalg.yield %17 : f32
    } -> tensor<4096x86x128xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map, #map3], iterator_types = ["parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%11 : tensor<4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %14, %out : f32
      linalg.yield %15 : f32
    } -> tensor<4096xf32>
    flow.dispatch.tensor.store %13, %4, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f32()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @i4_dequant_matvec_f32() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x1x32x128xf32>>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf32>>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>> -> tensor<4096x32x128xi4>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4096, 32, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>> -> tensor<4096x32x1xf32>
    %7 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [4096, 32, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>> -> tensor<4096x32x1xf32>
    %8 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [1, 1, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x32x128xf32>> -> tensor<1x1x32x128xf32>
    %9 = tensor.empty() : tensor<1x1x4096xf32>
    %10 = tensor.empty() : tensor<4096x32x128xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x32x128xi4>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%10 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %14 = arith.extui %in : i4 to i32
      %15 = arith.uitofp %14 : i32 to f32
      %16 = arith.subf %15, %in_1 : f32
      %17 = arith.mulf %16, %in_0 : f32
      linalg.yield %17 : f32
    } -> tensor<4096x32x128xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%11 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %14, %out : f32
      linalg.yield %15 : f32
    } -> tensor<1x1x4096xf32>
    flow.dispatch.tensor.store %13, %4, offsets = [0, 0, 0], sizes = [1, 1, 4096], strides = [1, 1, 1] : tensor<1x1x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf32>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 4, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [128, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f32()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @i4_dequant_matvec_f32() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = hal.interface.constant.load[2] : i32
    %3 = hal.interface.constant.load[3] : i32
    %4 = hal.interface.constant.load[4] : i32
    %5 = hal.interface.constant.load[5] : i32
    %6 = hal.interface.constant.load[6] : i32
    %7 = hal.interface.constant.load[7] : i32
    %8 = hal.interface.constant.load[8] : i32
    %9 = arith.index_castui %0 : i32 to index
    %10 = arith.index_castui %1 : i32 to index
    %11 = arith.index_castui %2 : i32 to index
    %12 = arith.extui %3 : i32 to i64
    %13 = arith.extui %4 : i32 to i64
    %14 = arith.shli %13, %c32_i64 : i64
    %15 = arith.ori %12, %14 : i64
    %16 = arith.index_castui %15 : i64 to index
    %17 = arith.extui %5 : i32 to i64
    %18 = arith.extui %6 : i32 to i64
    %19 = arith.shli %18, %c32_i64 : i64
    %20 = arith.ori %17, %19 : i64
    %21 = arith.index_castui %20 : i64 to index
    %22 = arith.extui %7 : i32 to i64
    %23 = arith.extui %8 : i32 to i64
    %24 = arith.shli %23, %c32_i64 : i64
    %25 = arith.ori %22, %24 : i64
    %26 = arith.index_castui %25 : i64 to index
    %27 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
    %28 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %29 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %30 = flow.dispatch.workload.ordinal %26, 0 : index
    %31 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%16) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x86x128xf32>>{%30}
    %32 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%21) : !flow.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%30}
    %33 = flow.dispatch.tensor.load %27, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
    %34 = flow.dispatch.tensor.load %28, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %35 = flow.dispatch.tensor.load %29, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %36 = flow.dispatch.tensor.load %31, offsets = [0, 0, 0], sizes = [%30, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x86x128xf32>>{%30} -> tensor<?x86x128xf32>
    %37 = tensor.empty(%30) : tensor<?x4096xf32>
    %38 = tensor.empty() : tensor<4096x86x128xf32>
    %39 = linalg.fill ins(%cst : f32) outs(%37 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    %40 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %34, %35 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%38 : tensor<4096x86x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %42 = arith.extui %in : i4 to i32
      %43 = arith.uitofp %42 : i32 to f32
      %44 = arith.subf %43, %in_1 : f32
      %45 = arith.mulf %44, %in_0 : f32
      linalg.yield %45 : f32
    } -> tensor<4096x86x128xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%36, %40 : tensor<?x86x128xf32>, tensor<4096x86x128xf32>) outs(%39 : tensor<?x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %42 = arith.mulf %in, %in_0 : f32
      %43 = arith.addf %42, %out : f32
      linalg.yield %43 : f32
    } -> tensor<?x4096xf32>
    flow.dispatch.tensor.store %41, %32, offsets = [0, 0], sizes = [%30, 4096], strides = [1, 1] : tensor<?x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%30}
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f32()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, Float16, StorageBuffer16BitAccess, GroupNonUniform, GroupNonUniformShuffle], [SPV_KHR_16bit_storage]>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 64]>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @i4_dequant_matvec_f16() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x86x128xf16>>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf16>>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4096, 86, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>> -> tensor<4096x86x1xf16>
    %7 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [4096, 86, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>> -> tensor<4096x86x1xf16>
    %8 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [1, 1, 86, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x86x128xf16>> -> tensor<1x1x86x128xf16>
    %9 = tensor.empty() : tensor<1x1x4096xf16>
    %10 = tensor.empty() : tensor<4096x86x128xf16>
    %11 = linalg.fill ins(%cst : f16) outs(%9 : tensor<1x1x4096xf16>) -> tensor<1x1x4096xf16>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x86x128xi4>, tensor<4096x86x1xf16>, tensor<4096x86x1xf16>) outs(%10 : tensor<4096x86x128xf16>) {
    ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
      %14 = arith.extui %in : i4 to i32
      %15 = arith.uitofp %14 : i32 to f16
      %16 = arith.subf %15, %in_1 : f16
      %17 = arith.mulf %16, %in_0 : f16
      linalg.yield %17 : f16
    } -> tensor<4096x86x128xf16>
    %13 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<1x1x86x128xf16>, tensor<4096x86x128xf16>) outs(%11 : tensor<1x1x4096xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %14 = arith.mulf %in, %in_0 : f16
      %15 = arith.addf %14, %out : f16
      linalg.yield %15 : f16
    } -> tensor<1x1x4096xf16>
    flow.dispatch.tensor.store %13, %4, offsets = [0, 0, 0], sizes = [1, 1, 4096], strides = [1, 1, 1] : tensor<1x1x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf16>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [32, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f16()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @i4_dequant_matvec() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = hal.interface.constant.load[2] : i32
    %3 = hal.interface.constant.load[3] : i32
    %4 = hal.interface.constant.load[4] : i32
    %5 = hal.interface.constant.load[5] : i32
    %6 = hal.interface.constant.load[6] : i32
    %7 = hal.interface.constant.load[7] : i32
    %8 = hal.interface.constant.load[8] : i32
    %9 = arith.index_castui %0 : i32 to index
    %10 = arith.index_castui %1 : i32 to index
    %11 = arith.index_castui %2 : i32 to index
    %12 = arith.extui %3 : i32 to i64
    %13 = arith.extui %4 : i32 to i64
    %14 = arith.shli %13, %c32_i64 : i64
    %15 = arith.ori %12, %14 : i64
    %16 = arith.index_castui %15 : i64 to index
    %17 = arith.extui %5 : i32 to i64
    %18 = arith.extui %6 : i32 to i64
    %19 = arith.shli %18, %c32_i64 : i64
    %20 = arith.ori %17, %19 : i64
    %21 = arith.index_castui %20 : i64 to index
    %22 = arith.extui %7 : i32 to i64
    %23 = arith.extui %8 : i32 to i64
    %24 = arith.shli %23, %c32_i64 : i64
    %25 = arith.ori %22, %24 : i64
    %26 = arith.index_castui %25 : i64 to index
    %27 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
    %28 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %29 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %30 = flow.dispatch.workload.ordinal %26, 0 : index
    %31 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%16) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x86x128xf32>>{%30}
    %32 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%21) : !flow.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%30}
    %33 = flow.dispatch.tensor.load %27, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
    %34 = flow.dispatch.tensor.load %28, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %35 = flow.dispatch.tensor.load %29, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %36 = flow.dispatch.tensor.load %31, offsets = [0, 0, 0], sizes = [%30, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x86x128xf32>>{%30} -> tensor<?x86x128xf32>
    %37 = tensor.empty(%30) : tensor<?x4096xf32>
    %38 = tensor.empty() : tensor<4096x86x128xf32>
    %39 = linalg.fill ins(%cst : f32) outs(%37 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    %40 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %34, %35 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%38 : tensor<4096x86x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %42 = arith.extui %in : i4 to i32
      %43 = arith.uitofp %42 : i32 to f32
      %44 = arith.subf %43, %in_1 : f32
      %45 = arith.mulf %44, %in_0 : f32
      linalg.yield %45 : f32
    } -> tensor<4096x86x128xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%36, %40 : tensor<?x86x128xf32>, tensor<4096x86x128xf32>) outs(%39 : tensor<?x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %42 = arith.mulf %in, %in_0 : f32
      %43 = arith.addf %42, %out : f32
      linalg.yield %43 : f32
    } -> tensor<?x4096xf32>
    flow.dispatch.tensor.store %41, %32, offsets = [0, 0], sizes = [%30, 4096], strides = [1, 1] : tensor<?x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%30}
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @i4_dequant_matvec() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = hal.interface.constant.load[2] : i32
    %3 = hal.interface.constant.load[3] : i32
    %4 = hal.interface.constant.load[4] : i32
    %5 = hal.interface.constant.load[5] : i32
    %6 = hal.interface.constant.load[6] : i32
    %7 = arith.index_castui %0 : i32 to index
    %8 = arith.index_castui %1 : i32 to index
    %9 = arith.index_castui %2 : i32 to index
    %10 = arith.extui %3 : i32 to i64
    %11 = arith.extui %4 : i32 to i64
    %12 = arith.shli %11, %c32_i64 : i64
    %13 = arith.ori %10, %12 : i64
    %14 = arith.index_castui %13 : i64 to index
    %15 = arith.extui %5 : i32 to i64
    %16 = arith.extui %6 : i32 to i64
    %17 = arith.shli %16, %c32_i64 : i64
    %18 = arith.ori %15, %17 : i64
    %19 = arith.index_castui %18 : i64 to index
    %20 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<11008x32x128xi4>>
    %21 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<11008x32xf16>>
    %22 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<11008x32xf16>>
    %23 = flow.dispatch.workload.ordinal %19, 0 : index
    %24 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x32x128xf16>>{%23}
    %25 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%14) : !flow.dispatch.tensor<writeonly:tensor<?x11008xf16>>{%23}
    %26 = flow.dispatch.tensor.load %20, offsets = [0, 0, 0], sizes = [11008, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<11008x32x128xi4>> -> tensor<11008x32x128xi4>
    %27 = flow.dispatch.tensor.load %21, offsets = [0, 0], sizes = [11008, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<11008x32xf16>> -> tensor<11008x32xf16>
    %28 = flow.dispatch.tensor.load %22, offsets = [0, 0], sizes = [11008, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<11008x32xf16>> -> tensor<11008x32xf16>
    %29 = flow.dispatch.tensor.load %24, offsets = [0, 0, 0], sizes = [%23, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x32x128xf16>>{%23} -> tensor<?x32x128xf16>
    %30 = tensor.empty() : tensor<11008x32x128xf16>
    %31 = tensor.empty(%23) : tensor<?x11008xf16>
    %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x11008xf16>) -> tensor<?x11008xf16>
    %33 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %27, %28 : tensor<11008x32x128xi4>, tensor<11008x32xf16>, tensor<11008x32xf16>) outs(%30 : tensor<11008x32x128xf16>) {
    ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
      %35 = arith.extui %in : i4 to i32
      %36 = arith.uitofp %35 : i32 to f16
      %37 = arith.subf %36, %in_1 : f16
      %38 = arith.mulf %37, %in_0 : f16
      linalg.yield %38 : f16
    } -> tensor<11008x32x128xf16>
    %34 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%29, %33 : tensor<?x32x128xf16>, tensor<11008x32x128xf16>) outs(%32 : tensor<?x11008xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %35 = arith.mulf %in, %in_0 : f16
      %36 = arith.addf %35, %out : f16
      linalg.yield %36 : f16
    } -> tensor<?x11008xf16>
    flow.dispatch.tensor.store %34, %25, offsets = [0, 0], sizes = [%23, 11008], strides = [1, 1] : tensor<?x11008xf16> -> !flow.dispatch.tensor<writeonly:tensor<?x11008xf16>>{%23}
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 4, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64>>}>
module {
  func.func @dynamic_batch_matvec() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = hal.interface.constant.load[2] : i32
    %3 = hal.interface.constant.load[3] : i32
    %4 = hal.interface.constant.load[4] : i32
    %5 = arith.index_castui %0 : i32 to index
    %6 = arith.index_castui %1 : i32 to index
    %7 = arith.index_castui %2 : i32 to index
    %8 = arith.index_castui %3 : i32 to index
    %9 = arith.index_castui %4 : i32 to index
    %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
    %11 = flow.dispatch.workload.ordinal %8, 0 : index
    %12 = flow.dispatch.workload.ordinal %9, 1 : index
    %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%11}
    %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%12}
    %15 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [32, 1, %11], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%11} -> tensor<32x1x?xf16>
    %16 = flow.dispatch.tensor.load %14, offsets = [0, 0, 0], sizes = [32, %12, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%12} -> tensor<32x?x128xf16>
    %17 = tensor.empty() : tensor<32x1x128xf16>
    %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
    %19 = linalg.batch_matmul ins(%15, %16 : tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%18 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
    flow.dispatch.tensor.store %19, %10, offsets = [0, 0, 0], sizes = [32, 1, 128], strides = [1, 1, 1] : tensor<32x1x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 64]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @dynamic_batch_matvec()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
