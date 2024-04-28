// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline)" \
// RUN:   --split-input-file %s | FileCheck %s

// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-rocdl-configuration-pipeline)" \
// RUN:   --split-input-file %s | FileCheck %s

// Make sure that the GPU configuration pipelines generalize named ops, e.g., linalg.matmul_transpose_b to linalg.generic.

// CHECK:      linalg.fill
// CHECK-NEXT: linalg.generic
// CHECK-NOT:  linalg.matmul_transpose_b

#target = #iree_gpu.target<api = hip, arch = "gfx942",
  core = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8,
  subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
  mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>],
  subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_size = 1024, max_workgroup_memory_bytes = 65536>>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #target}>
module {
  func.func @warp_reduction_large_vector() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c394240 = arith.constant 394240 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c128) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1280xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c394240) : !flow.dispatch.tensor<writeonly:tensor<1x1280xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1280xf32>> -> tensor<1x1280xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf32>> -> tensor<1280x1280xf32>
    %5 = tensor.empty() : tensor<1x1280xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1280xf32>) -> tensor<1x1280xf32>
    %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<1x1280xf32>, tensor<1280x1280xf32>) outs(%6 : tensor<1x1280xf32>) -> tensor<1x1280xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : tensor<1x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1280xf32>>
    return
  }
}

