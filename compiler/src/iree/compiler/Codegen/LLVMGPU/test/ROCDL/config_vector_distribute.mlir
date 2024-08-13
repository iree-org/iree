// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=WMMA

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 64, 64, 128]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:   subgroup_m_count = 1, subgroup_n_count = 4

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @expanded_matmul_transpose_b() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
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

// CHECK-LABEL: func.func @expanded_matmul_transpose_b()
// CHECK: linalg.generic {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 64, 128, 1, 1, 32]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @conv_nhwc() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 258, 514, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>> -> tensor<2x258x514x768xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 768, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>> -> tensor<3x3x768x256xf16>
  %5 = tensor.empty() : tensor<2x256x512x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x258x514x768xf16>, tensor<3x3x768x256xf16>) outs(%6 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 256, 512, 256], strides = [1, 1, 1, 1] : tensor<2x256x512x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
  return
}

// CHECK-LABEL: func.func @conv_nhwc()
// CHECK: linalg.conv_2d_nhwc_hwcf {{.*}} lowering_config = #[[$TILE_SIZES]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#target = #iree_gpu.target<arch = "gfx940", features = "", wgp = <
  compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8,
  subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
  mma = [],
  subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #target}>
func.func @matmul_256x256x256() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// Check that we do not use the distribute pipeline if there are no supported
// intrinsics.
//       CHECK-NOT: iree_codegen.translation_info<LLVMGPUVectorDistribute

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[64, 128, 64]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @mfma_matmul_1024x1024x1024() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
  %5 = tensor.empty() : tensor<1024x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  return
}

// CHECK-LABEL: func.func @mfma_matmul_1024x1024x1024()
// CHECK: linalg.matmul {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 1, 1, 32, 32, 1, 1, 1, 32]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:   subgroup_m_count = 2, subgroup_n_count = 2

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 32, 0, 1, 1, 1, 0]]>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d4, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
func.func @conv_nchwc() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x20x34x34x64xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x20x3x3x160x64xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x8x32x32x160xf16>>
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

// CHECK-LABEL: func.func @conv_nchwc()
// CHECK: linalg.generic {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// WMMA:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[64, 128, 64]{{\]}}
// WMMA:      #iree_codegen.translation_info<LLVMGPUVectorDistribute
// WMMA-SAME: mma_schedule = #iree_gpu.mma_schedule
// WMMA-SAME:   intrinsic = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>
// WMMA-SAME:   subgroup_m_count = 2, subgroup_n_count = 2

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @wmma_matmul_1024x1024x1024() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf16>> -> tensor<1024x1024xf16>
  %5 = tensor.empty() : tensor<1024x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  return
}

// WMMA-LABEL: func.func @wmma_matmul_1024x1024x1024()
// WMMA: linalg.matmul {{.*}}lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 16, 16, 16]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic =  #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:   subgroup_m_count = 1, subgroup_n_count = 1

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @unaligned_mk_batch_matmul() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 968, 1281], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>> -> tensor<64x968x1281xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 1281, 1281], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>> -> tensor<64x1281x1281xf16>
  %5 = tensor.empty() : tensor<64x968x1281xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x968x1281xf16>) -> tensor<64x968x1281xf16>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<64x968x1281xf16>, tensor<64x1281x1281xf16>) outs(%6 : tensor<64x968x1281xf16>) -> tensor<64x968x1281xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [64, 968, 1281], strides = [1, 1, 1] : tensor<64x968x1281xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
  return
}
// CHECK-LABEL: func.func @unaligned_mk_batch_matmul()
// CHECK:         linalg.batch_matmul
// CHECK-SAME:      lowering_config = #[[$TILE_SIZES]]

// -----

// CHECK:      #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 16, 128, 128]{{\]}}
// CHECK:      #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
// CHECK-SAME: mma_schedule = #iree_gpu.mma_schedule
// CHECK-SAME:   intrinsic =  #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:   subgroup_m_count = 1, subgroup_n_count = 4

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @unaligned_m_batch_matmul_64x72x1280x1280() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x72x1280xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x1280xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x72x1280xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 72, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x72x1280xf16>> -> tensor<64x72x1280xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 1280, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x1280xf16>> -> tensor<64x1280x1280xf16>
  %5 = tensor.empty() : tensor<64x72x1280xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x72x1280xf16>) -> tensor<64x72x1280xf16>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<64x72x1280xf16>, tensor<64x1280x1280xf16>) outs(%6 : tensor<64x72x1280xf16>) -> tensor<64x72x1280xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [64, 72, 1280], strides = [1, 1, 1] : tensor<64x72x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x72x1280xf16>>
  return
}
// CHECK-LABEL: func.func @unaligned_m_batch_matmul_64x72x1280x1280()
// CHECK:         linalg.batch_matmul
// CHECK-SAME:      lowering_config = #[[$TILE_SIZES]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @narrow_n_batch_matmul_64x968x4x320_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x320xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x320x4xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x4xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 968, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x968x320xf16>> -> tensor<64x968x320xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 320, 4], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x320x4xf16>> -> tensor<64x320x4xf16>
  %5 = tensor.empty() : tensor<64x968x4xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x968x4xf16>) -> tensor<64x968x4xf16>
  %7 = linalg.batch_matmul ins(%3, %4 : tensor<64x968x320xf16>, tensor<64x320x4xf16>) outs(%6 : tensor<64x968x4xf16>) -> tensor<64x968x4xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [64, 968, 4], strides = [1, 1, 1] : tensor<64x968x4xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x968x4xf16>>
  return
}
// Check that we don't support LLVMGPUPadAndVectorDistribute for narrow N/M atm.
// CHECK-NOT:      #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
// CHECK-LABEL: func.func @narrow_n_batch_matmul_64x968x4x320_f16()

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @matmul_dynamic_dim() {
  %c0 = arith.constant 0 : index
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.extui %0 : i32 to i64
  %3 = arith.extui %1 : i32 to i64
  %4 = arith.shli %3, %c32_i64 : i64
  %5 = arith.ori %2, %4 : i64
  %6 = arith.index_castui %5 : i64 to index
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
  %8 = flow.dispatch.workload.ordinal %6, 0 : index
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x256xf16>>{%8}
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x256xf32>>{%8}
  %11 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x256xf16>>{%8} -> tensor<?x256xf16>
  %12 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %13 = tensor.empty(%8) : tensor<?x256xf32>
  %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x256xf32>) -> tensor<?x256xf32>
  %15 = linalg.matmul ins(%11, %12 : tensor<?x256xf16>, tensor<256x256xf16>) outs(%14 : tensor<?x256xf32>) -> tensor<?x256xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0], sizes = [%8, 256], strides = [1, 1] : tensor<?x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x256xf32>>{%8}
  return
}
// Check that we have unhandled dynamic dimension.
//       CHECK-NOT: iree_codegen.translation_info<LLVMGPUVectorDistribute

// -----

// CHECK:       #[[$TILE_SIZES:.+]] = #iree_codegen.lowering_config<tile_sizes =  {{\[}}[1, 32, 0, 64, 64]{{\]}}
// CHECK:       #iree_codegen.translation_info<LLVMGPUVectorDistribute
// CHECK-SAME:  subgroup_m_count = 1, subgroup_n_count = 1
// CHECK-NOT:   prefetch_shared_memory

// CHECK-LABEL: func.func @attention_20x4096x64x4096x64()

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
func.func @attention_20x4096x64x4096x64() {
  %cst = arith.constant 1.250000e-01 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %7 = tensor.empty() : tensor<20x4096x64xf16>
  %8 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%4, %5, %6, %cst : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16>
  flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
  return
}
