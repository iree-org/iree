// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx940 \
// RUN: --iree-codegen-llvmgpu-use-igemm=true --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

func.func @nhwc_conv_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<3x3x128x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf32>> -> tensor<2x34x34x128xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x128x64xf32>> -> tensor<3x3x128x64xf32>
  %5 = tensor.empty() : tensor<2x32x32x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x34x34x128xf32>, tensor<3x3x128x64xf32>) outs(%6 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 64], strides = [1, 1, 1, 1] : tensor<2x32x32x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  return
}

// CHECK-LABEL: func.func @nhwc_conv_mfma
//  CHECK-SAME:   #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.conv_2d_nhwc_hwcf {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 8]
//  CHECK-SAME:     subgroup = [1, 2, 2, 1, 0]
//  CHECK-SAME:     workgroup = [1, 2, 2, 4, 0]

// -----

func.func @nchw_conv_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x128x34x34xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<64x128x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x64x32x32xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 128, 34, 34], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x34x34xf32>> -> tensor<2x128x34x34xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [64, 128, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x128x3x3xf32>> -> tensor<64x128x3x3xf32>
  %5 = tensor.empty() : tensor<2x64x32x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x64x32x32xf32>) -> tensor<2x64x32x32xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x128x34x34xf32>, tensor<64x128x3x3xf32>) outs(%6 : tensor<2x64x32x32xf32>) -> tensor<2x64x32x32xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 64, 32, 32], strides = [1, 1, 1, 1] : tensor<2x64x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x64x32x32xf32>>
  return
}

// CHECK-LABEL: func.func @nchw_conv_mfma
//  CHECK-SAME:   #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = true

//       CHECK:   linalg.conv_2d_nchw_fchw {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 8]
//  CHECK-SAME:     subgroup = [1, 2, 2, 1, 0]
//  CHECK-SAME:     workgroup = [1, 4, 2, 2, 0]

// -----

func.func @nhwc_conv_no_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x33x33x128xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<3x3x128x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x31x31x64xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 33, 33, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x33x33x128xf32>> -> tensor<2x33x33x128xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x128x64xf32>> -> tensor<3x3x128x64xf32>
  %5 = tensor.empty() : tensor<2x31x31x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x31x31x64xf32>) -> tensor<2x31x31x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x33x33x128xf32>, tensor<3x3x128x64xf32>) outs(%6 : tensor<2x31x31x64xf32>) -> tensor<2x31x31x64xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 31, 31, 64], strides = [1, 1, 1, 1] : tensor<2x31x31x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x31x31x64xf32>>
  return
}

// CHECK-LABEL: func.func @nhwc_conv_no_mfma
//   CHECK-NOT:   use_igemm_convolution = true

// -----

func.func @nchw_conv_no_mfma() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x128x34x34xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<63x128x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x63x32x32xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 128, 34, 34], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x34x34xf32>> -> tensor<2x128x34x34xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [63, 128, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<63x128x3x3xf32>> -> tensor<63x128x3x3xf32>
  %5 = tensor.empty() : tensor<2x63x32x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x63x32x32xf32>) -> tensor<2x63x32x32xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x128x34x34xf32>, tensor<63x128x3x3xf32>) outs(%6 : tensor<2x63x32x32xf32>) -> tensor<2x63x32x32xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 63, 32, 32], strides = [1, 1, 1, 1] : tensor<2x63x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x63x32x32xf32>>
  return
}

// CHECK-LABEL: func.func @nchw_conv_no_mfma
//   CHECK-NOT:   use_igemm_convolution = true
