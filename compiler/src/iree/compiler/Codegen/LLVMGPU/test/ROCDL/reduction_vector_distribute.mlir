// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --iree-codegen-llvmgpu-test-vector-distribution-reduction \
// RUN:   --iree-codegen-llvmgpu-use-unaligned-gemm-vector-distribution --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=REDUCTION

// REDUCTION:       #iree_codegen.translation_info<LLVMGPUVectorDistribute
// REDUCTION: workgroup_size = [64, 1, 1] subgroup_size = 64

// REDUDCTION-LABEL: func.func @reduction_2x32x10x4096_f16xf32xf32xf32

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_2x32x10x4096_f16xf32xf32xf32() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.096000e+04 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %c69524992 = arith.constant 69524992 : index
    %c74767872 = arith.constant 74767872 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x32x10x4096xf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x32xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 4096], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x10x4096xf16>> -> tensor<2x32x10x4096xf16>
    %3 = tensor.empty() : tensor<2x32x10x4096xf32>
    %4 = tensor.empty() : tensor<2x32xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x32x10x4096xf16>) outs(%3 : tensor<2x32x10x4096xf32>) {
    ^bb0(%in: f16, %out: f32):
    %11 = arith.extf %in : f16 to f32
    linalg.yield %11 : f32
    } -> tensor<2x32x10x4096xf32>
    %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 512]]>} ins(%cst : f32) outs(%4 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5 : tensor<2x32x10x4096xf32>) outs(%6 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
    %11 = arith.addf %in, %out : f32
    linalg.yield %11 : f32
    } -> tensor<2x32xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<2x32xf32>) outs(%4 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
    %11 = arith.divf %in, %cst_0 : f32
    linalg.yield %11 : f32
    } -> tensor<2x32xf32>
    flow.dispatch.tensor.store %8, %1, offsets = [0, 0], sizes = [2, 32], strides = [1, 1] : tensor<2x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32xf32>>
    return
}

// REDUCTION:                #iree_gpu.lowering_config
// REDUCTION-SAME:                           reduction =  [0, 0, 1, 512]
// REDUCTION-SAME:                           workgroup =  [1, 1, 0, 0]
