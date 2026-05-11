// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN: | FileCheck %s

// Reduction with an inner dim (300) that is not a multiple of the preferred
// subgroup size (64 on gfx942). setReductionConfig must:
//   - keep the inner dim as the reduction dimension (no fallback to product
//     since 300 >= 64),
//   - pick subgroup_size = preferredSubgroupSize = 64,
//   - leave threadLoads at 4 (= 128 / sizeof(f32 in bits)) since
//     divideCeil(300, 4) = 75 >= 64 already satisfies the "at least one
//     full subgroup of work" structural invariant,
//   - compute workgroup_size = divideCeil(75, 64) * 64 = 128 (rounded up
//     to a whole number of subgroups so VectorDistribute can mask the tail).
//
// This shape was empirically verified to compile and execute correctly on
// gfx942 (MI300X) with masking handling the 300-vs-(128*4=512) tail.

// CHECK: #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [128, 1, 1] subgroup_size = 64

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_non_aligned_8x300() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x300xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 300], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x300xf32>> -> tensor<8x300xf32>
  %3 = tensor.empty() : tensor<8xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%2 : tensor<8x300xf32>) outs(%4 : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
  return
}

// -----

// Single-row reduction with a prime inner dim (257). This case used to fail
// the LLVMGPUVectorDistributePass with 'failed to distribute' because the
// downstream getVectorDistributeReductionConfig collapsed the per-thread
// tile to size 1 via GCD with the prime size. Verified e2e on gfx942 with
// the all-ones case (sum=257.0) and with a tail-marker at position 256
// (sum=356.0), so masking correctly preserves the last element.
// CHECK: #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [128, 1, 1] subgroup_size = 64

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_non_aligned_1x257_prime() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x257xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 257], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x257xf32>> -> tensor<1x257xf32>
  %3 = tensor.empty() : tensor<1xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1xf32>) -> tensor<1xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%2 : tensor<1x257xf32>) outs(%4 : tensor<1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xf32>>
  return
}

// -----

// Reduction with a dynamic inner dim. setReductionConfig must:
//   - take the hasDynamicReductionDim path in pickSubgroupSize and
//     pickThreadLoads (no fallback condition triggers, threadLoads stays at
//     128 / sizeof(f32 in bits) = 4),
//   - use kVectorDistributeReductionSizeToTargetIfDynamic (= 1 << 31) as the
//     reductionSize estimate when computing workgroup_size,
//   - cap workgroup_size at maxWorkgroupSize via the GCD branch while still
//     respecting `workgroup_size % subgroup_size == 0`.
// The exact workgroup_size depends on maxWorkgroupSize from the target wgp
// attribute, which can change as gfx942's reported limits evolve. Match
// "any positive integer" instead of pinning to today's 1024.
// CHECK: #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [{{[0-9]+}}, 1, 1] subgroup_size = 64

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_non_aligned_dynamic() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>{%dim}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, %dim], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x?xf32>>{%dim} -> tensor<8x?xf32>
  %3 = tensor.empty() : tensor<8xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%2 : tensor<8x?xf32>) outs(%4 : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
  return
}
