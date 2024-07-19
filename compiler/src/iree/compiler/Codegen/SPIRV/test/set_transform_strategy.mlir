// RUN: iree-opt %s --split-input-file --iree-gpu-test-target=volta@vulkan \
// RUN:   --pass-pipeline="builtin.module(iree-spirv-select-lowering-strategy-pass)"\
// RUN:   --iree-spirv-enable-transform-dialect-jit=true

// TODO: Transform script based CodeGen expects fp32-input to target tensor
// core, but there are no such wmma intrinsics. Fix it to support fp16-input.
// TODO: | FileCheck %s

module {
  func.func @matmul() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2052x2556xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2556x2052xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2052x2052xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2052, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2052x2556xf32>> -> tensor<2052x2556xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2556, 2052], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2556x2052xf32>> -> tensor<2556x2052xf32>
    %5 = tensor.empty() : tensor<2052x2052xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2052x2052xf32>) -> tensor<2052x2052xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<2052x2556xf32>, tensor<2556x2052xf32>) outs(%6 : tensor<2052x2052xf32>) -> tensor<2052x2052xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2052, 2052], strides = [1, 1] : tensor<2052x2052xf32> -> !flow.dispatch.tensor<writeonly:tensor<2052x2052xf32>>
    return
  }
}

// CHECK-LABEL: func @matmul

// CHECK: transform.named_sequence

/// The specific vector sizes are tested in the LLVMGPU tests and thus omitted
/// here. This is just to check that masked vectorization is used.
// CHECK-COUNT-3: transform.structured.vectorize

// Verify use of WMMA.
// CHECK: apply_patterns to %{{.*}} {
// CHECK:   transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
// CHECK: } : !transform.any_op
// CHECK: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_wmma}

// Verify asynchronous copy is not used.
// CHECK-NOT: transform.iree.create_async_groups
