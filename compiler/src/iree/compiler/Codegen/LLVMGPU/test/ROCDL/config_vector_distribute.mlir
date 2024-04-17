// RUN: iree-opt --split-input-file --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 64, 64, 128]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 1, subgroup_n_count = 4,
// CHECK-SAME:   subgroup_m_tile_count = 4, subgroup_n_tile_count = 1, subgroup_k_tile_count = 8

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>], target_arch = "gfx940"}>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  func.func @expanded_matmul_transpose_b() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 64, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>> -> tensor<2x64x2048xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 64, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>> -> tensor<10x64x2048xf16>
    %5 = tensor.empty() : tensor<2x10x64x64xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf16>) -> tensor<2x10x64x64xf16>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %8 = arith.mulf %in, %in_0 : f16
      %9 = arith.addf %8, %out : f16
      linalg.yield %9 : f16
    } -> tensor<2x10x64x64xf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1] : tensor<2x10x64x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
    return
  }
}

// CHECK-LABEL: func.func @expanded_matmul_transpose_b()
// CHECK: linalg.generic {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 64, 128, 1, 1, 32]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2,
// CHECK-SAME:   subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>], target_arch = "gfx940"}>
module {
  func.func @conv_nhwc() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 258, 514, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>> -> tensor<2x258x514x768xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 768, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>> -> tensor<3x3x768x256xf16>
    %5 = tensor.empty() : tensor<2x256x512x256xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x258x514x768xf16>, tensor<3x3x768x256xf16>) outs(%6 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 256, 512, 256], strides = [1, 1, 1, 1] : tensor<2x256x512x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
    return
  }
}

// CHECK-LABEL: func.func @conv_nhwc()
// CHECK: linalg.conv_2d_nhwc_hwcf {{.*}} lowering_config = #[[$TILE_SIZES]]

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [], target_arch = "gfx940"}>
module {
  func.func @matmul_256x256x256() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
    %5 = tensor.empty() : tensor<256x256xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
    return
  }
}

// Check that we do not use the distribute pipeline if there are no supported
// intrinsics.
//       CHECK-NOT: iree_codegen.translation_info<LLVMGPUVectorDistribute

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[64, 128, 64]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2,
// CHECK-SAME:   subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 4

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>], target_arch = "gfx940"}>
module {
  func.func @mfma_matmul_1024x1024x1024() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
    %5 = tensor.empty() : tensor<1024x1024xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
    return
  }
}

// CHECK-LABEL: func.func @mfma_matmul_1024x1024x1024()
// CHECK: linalg.matmul {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 1, 32, 32, 1, 1, 1, 32]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2,
// CHECK-SAME:   subgroup_m_tile_count = 1, subgroup_n_tile_count = 1, subgroup_k_tile_count = 2

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 32, 0, 1, 1, 1, 0]]>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>], target_arch = "gfx940"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d4, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  func.func @conv_nchwc() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x20x34x34x64xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x20x3x3x160x64xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x8x32x32x160xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0], sizes = [2, 20, 34, 34, 64], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x20x34x34x64xf16>> -> tensor<2x20x34x34x64xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 20, 3, 3, 160, 64], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x20x3x3x160x64xf16>> -> tensor<8x20x3x3x160x64xf16>
    %5 = tensor.empty() : tensor<2x8x32x32x160xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x8x32x32x160xf32>) -> tensor<2x8x32x32x160xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%3, %4 : tensor<2x20x34x34x64xf16>, tensor<8x20x3x3x160x64xf16>) outs(%6 : tensor<2x8x32x32x160xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %10 = arith.extf %in : f16 to f32
      %11 = arith.extf %in_0 : f16 to f32
      %12 = arith.mulf %10, %11 : f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<2x8x32x32x160xf32>
    %8 = tensor.empty() : tensor<2x8x32x32x160xf16>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<2x8x32x32x160xf32>) outs(%8 : tensor<2x8x32x32x160xf16>) {
    ^bb0(%in: f32, %out: f16):
      %10 = arith.truncf %in : f32 to f16
      linalg.yield %10 : f16
    } -> tensor<2x8x32x32x160xf16>
    flow.dispatch.tensor.store %9, %2, offsets = [0, 0, 0, 0, 0], sizes = [2, 8, 32, 32, 160], strides = [1, 1, 1, 1, 1] : tensor<2x8x32x32x160xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x8x32x32x160xf16>>
    return
  }
}

// CHECK-LABEL: func.func @conv_nchwc()
// CHECK: linalg.generic {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[64, 128, 64]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2,
// CHECK-SAME:   subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 4

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>], target_arch = "gfx1100"}>
module {
  func.func @wmma_matmul_1024x1024x1024() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
    %5 = tensor.empty() : tensor<1024x1024xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
    return
  }
}

// CHECK-LABEL: func.func @wmma_matmul_1024x1024x1024()
// CHECK: linalg.matmul {{.*}}lowering_config = #[[$TILE_SIZES]]
