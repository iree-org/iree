// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --iree-codegen-llvmgpu-use-igemm=true --iree-codegen-llvmgpu-igemm-pad-convolution=false --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefixes=CHECK,GFX942

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=mi300x@hip \
// RUN: --iree-codegen-llvmgpu-use-igemm=true --iree-codegen-llvmgpu-igemm-pad-convolution=false --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefixes=CHECK,MI300X

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --iree-codegen-llvmgpu-use-igemm=true --iree-codegen-llvmgpu-igemm-pad-convolution=true --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=PAD-CONV

func.func @nhwc_conv_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf32>> -> tensor<2x34x34x128xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x64xf32>> -> tensor<3x3x128x64xf32>
  %5 = tensor.empty() : tensor<2x32x32x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x34x34x128xf32>, tensor<3x3x128x64xf32>) outs(%6 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 64], strides = [1, 1, 1, 1] : tensor<2x32x32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  return
}

// CHECK-LABEL: func.func @nhwc_conv_mfma
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.conv_2d_nhwc_hwcf {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
//  CHECK-SAME:     promote_operands = [0, 1]

//  GFX942-SAME:    reduction = [0, 0, 0, 0, 8]
//  GFX942-SAME:    subgroup = [1, 2, 2, 1, 0]
//  GFX942-SAME:    workgroup = [1, 2, 32, 64, 0]

//  MI300X-SAME:    reduction = [0, 0, 0, 0, 8]
//  MI300X-SAME:    subgroup = [1, 1, 1, 1, 0]
//  MI300X-SAME:    workgroup = [1, 1, 16, 64, 0]}>

// -----

func.func @nchw_conv_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x128x34x34xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x64x32x32xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 128, 34, 34], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x128x34x34xf32>> -> tensor<2x128x34x34xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [64, 128, 3, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128x3x3xf32>> -> tensor<64x128x3x3xf32>
  %5 = tensor.empty() : tensor<2x64x32x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x64x32x32xf32>) -> tensor<2x64x32x32xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x128x34x34xf32>, tensor<64x128x3x3xf32>) outs(%6 : tensor<2x64x32x32xf32>) -> tensor<2x64x32x32xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 64, 32, 32], strides = [1, 1, 1, 1] : tensor<2x64x32x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x64x32x32xf32>>
  return
}

// CHECK-LABEL: func.func @nchw_conv_mfma
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.conv_2d_nchw_fchw {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
//  CHECK-SAME:     promote_operands = [0, 1]

// GFX942-SAME:     reduction = [0, 0, 0, 0, 8]
// GFX942-SAME:     subgroup = [1, 2, 2, 1, 0]
// GFX942-SAME:     workgroup = [1, 64, 2, 32, 0]

// MI300X-SAME:     reduction = [0, 0, 0, 0, 8]
// MI300X-SAME:     subgroup = [1, 1, 1, 1, 0]
// MI300X-SAME:     workgroup = [1, 32, 1, 32, 0]

// -----

func.func @nhwc_conv_unaligned_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x33x33x128xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x31x31x64xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 33, 33, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x33x33x128xf32>> -> tensor<2x33x33x128xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x64xf32>> -> tensor<3x3x128x64xf32>
  %5 = tensor.empty() : tensor<2x31x31x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x31x31x64xf32>) -> tensor<2x31x31x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x33x33x128xf32>, tensor<3x3x128x64xf32>) outs(%6 : tensor<2x31x31x64xf32>) -> tensor<2x31x31x64xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 31, 31, 64], strides = [1, 1, 1, 1] : tensor<2x31x31x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x31x31x64xf32>>
  return
}

// CHECK-LABEL: func.func @nhwc_conv_unaligned_mfma
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.conv_2d_nhwc_hwcf {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>

// GFX942-SAME:     padding = [2, 1, 32, 64, 32]
// GFX942-SAME:     promote_operands = [0, 1, 2]
// GFX942-SAME:     reduction = [0, 0, 0, 0, 8]
// GFX942-SAME:     subgroup = [2, 1, 2, 1, 0]
// GFX942-SAME:     workgroup = [2, 1, 32, 64, 0]

// MI300X-SAME:     padding = [1, 1, 16, 64, 32]
// MI300X-SAME:     promote_operands = [0, 1, 2]
// MI300X-SAME:     reduction = [0, 0, 0, 0, 8]
// MI300X-SAME:     subgroup = [1, 1, 1, 1, 0]
// MI300X-SAME:     workgroup = [1, 1, 16, 64, 0]

//    PAD-CONV:     padding_conv = [2, 1, 32, 64, 0, 0, 0]

// -----

func.func @nchw_conv_unaligned_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x128x34x34xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<63x128x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x63x32x32xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 128, 34, 34], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x128x34x34xf32>> -> tensor<2x128x34x34xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [63, 128, 3, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<63x128x3x3xf32>> -> tensor<63x128x3x3xf32>
  %5 = tensor.empty() : tensor<2x63x32x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x63x32x32xf32>) -> tensor<2x63x32x32xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x128x34x34xf32>, tensor<63x128x3x3xf32>) outs(%6 : tensor<2x63x32x32xf32>) -> tensor<2x63x32x32xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 63, 32, 32], strides = [1, 1, 1, 1] : tensor<2x63x32x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x63x32x32xf32>>
  return
}

// CHECK-LABEL: func.func @nchw_conv_unaligned_mfma
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.conv_2d_nchw_fchw {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>

// GFX942-SAME:     padding = [1, 64, 2, 32, 32]
// GFX942-SAME:     promote_operands = [0, 1, 2]
// GFX942-SAME:     reduction = [0, 0, 0, 0, 8]
// GFX942-SAME:     subgroup = [1, 2, 2, 1, 0]
// GFX942-SAME:     workgroup = [1, 64, 2, 32, 0]

// MI300X-SAME:     padding = [1, 32, 1, 32, 32]
// MI300X-SAME:     promote_operands = [0, 1, 2]
// MI300X-SAME:     reduction = [0, 0, 0, 0, 8]
// MI300X-SAME:     subgroup = [1, 1, 1, 1, 0]
// MI300X-SAME:     workgroup = [1, 32, 1, 32, 0]

//    PAD-CONV:     padding_conv = [1, 64, 2, 32, 0, 0, 0]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @conv_nhwc_fhwc_unaligned_channel(%arg0: tensor<16x26x19x287xf16>, %arg1: tensor<287x3x3x287xf16>, %arg2: tensor<16x24x17x287xf32>) -> tensor<16x24x17x287xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x26x19x287xf16>, tensor<287x3x3x287xf16>) outs(%arg2 : tensor<16x24x17x287xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x24x17x287xf32>
  return %0 : tensor<16x24x17x287xf32>
}

// CHECK-LABEL: func.func @conv_nhwc_fhwc_unaligned_channel
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>

// GFX942-SAME:     padding = [1, 8, 32, 32, 32]
// GFX942-SAME:     promote_operands = [0, 1, 2]
// GFX942-SAME:     reduction = [0, 0, 0, 0, 2]
// GFX942-SAME:     subgroup = [1, 8, 1, 1, 0]
// GFX942-SAME:     workgroup = [1, 8, 32, 32, 0]

// MI300X-SAME:     padding = [1, 4, 32, 32, 32]
// MI300X-SAME:     promote_operands = [0, 1, 2]
// MI300X-SAME:     reduction = [0, 0, 0, 0, 2]
// MI300X-SAME:     subgroup = [1, 4, 1, 1, 0]
// MI300X-SAME:     workgroup = [1, 4, 32, 32, 0]

//    PAD-CONV:     padding_conv = [1, 8, 32, 32, 0, 0, 32]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5 * 2, d2 + d6 * 2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @conv_chwn_chwf_unaligned(%arg0: tensor<16x193x129x40xbf16>, %arg1: tensor<16x96x64x40xbf16>, %arg2: tensor<40x3x3x40xf32>) -> tensor<40x3x3x40xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x193x129x40xbf16>, tensor<16x96x64x40xbf16>) outs(%arg2 : tensor<40x3x3x40xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %1 = arith.extf %in : bf16 to f32
    %2 = arith.extf %in_0 : bf16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<40x3x3x40xf32>
  return %0 : tensor<40x3x3x40xf32>
}

// CHECK-LABEL: func.func @conv_chwn_chwf_unaligned
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>
//  CHECK-SAME:     padding = [16, 1, 1, 16, 64]
//  CHECK-SAME:     promote_operands = [0, 1, 2]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 4]
//  CHECK-SAME:     subgroup = [1, 1, 1, 1, 0]
//  CHECK-SAME:     workgroup = [16, 1, 1, 16, 0]

//    PAD-CONV:     padding_conv = [16, 1, 1, 16, 0, 0, 0]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 + d4, d1 + d5, d2, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @group_conv_unaligned(%arg0: tensor<61x93x16x56xbf16>, %arg1: tensor<16x56x3x3x56xbf16>, %arg2: tensor<59x91x16x56xf32>) -> tensor<59x91x16x56xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<61x93x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg2 : tensor<59x91x16x56xf32>) {
    ^bb0(%in: bf16, %in_4: bf16, %out: f32):
      %10 = arith.extf %in : bf16 to f32
      %11 = arith.extf %in_4 : bf16 to f32
      %12 = arith.mulf %10, %11 : f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<59x91x16x56xf32>
  return %0 : tensor<59x91x16x56xf32>
}

// CHECK-LABEL: func.func @group_conv_unaligned
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
// GFX942-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>
// GFX942-SAME:     padding = [1, 32, 1, 64, 32]
// GFX942-SAME:     promote_operands = [0, 1, 2]
// GFX942-SAME:     reduction = [0, 0, 0, 0, 2]
// GFX942-SAME:     subgroup = [1, 2, 0, 1, 0]
// GFX942-SAME:     workgroup = [1, 32, 1, 64, 0]

// MI300X-SAME:     padding = [1, 32, 1, 32, 32]
// MI300X-SAME:     promote_operands = [0, 1, 2]
// MI300X-SAME:     reduction = [0, 0, 0, 0, 2]
// MI300X-SAME:     subgroup = [1, 1, 0, 1, 0]
// MI300X-SAME:     workgroup = [1, 32, 1, 32, 0]

//    PAD-CONV:     padding_conv = [1, 32, 1, 64, 0, 0, 32]
