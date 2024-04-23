// RUN: iree-opt --split-input-file --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target, canonicalize))' \
// RUN:   %s | FileCheck %s

#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>
module {
  func.func @fit_shared_memory_schedule() {
    %cst = arith.constant 0.000000e+00 : f32
    %c129181184 = arith.constant 129181184 : index
    %c18112 = arith.constant 18112 : index
    %c100980224 = arith.constant 100980224 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c129181184) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x80x1280xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c18112) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x1280xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c100980224) : !flow.dispatch.tensor<writeonly:tensor<64x80x1280xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 80, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x80x1280xf16>> -> tensor<64x80x1280xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 1280, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x1280xf16>> -> tensor<64x1280x1280xf16>
    %5 = tensor.empty() : tensor<64x80x1280xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x80x1280xf32>) -> tensor<64x80x1280xf32>
    %7 = linalg.batch_matmul ins(%3, %4 : tensor<64x80x1280xf16>, tensor<64x1280x1280xf16>) outs(%6 : tensor<64x80x1280xf32>) -> tensor<64x80x1280xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [64, 80, 1280], strides = [1, 1, 1] : tensor<64x80x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x80x1280xf32>>
    return
  }
}


// CHECK-LABEL: func.func @fit_shared_memory_schedule()
