// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-test-vector-distribution-on-reduction \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s


#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction() {
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
    %6 = linalg.fill  ins(%cst : f32) outs(%4 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5 : tensor<2x32x10x4096xf32>) outs(%6 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
    %11 = arith.addf %in, %out : f32
    linalg.yield %11 : f32
    } -> tensor<2x32xf32>
    flow.dispatch.tensor.store %7, %1, offsets = [0, 0], sizes = [2, 32], strides = [1, 1] : tensor<2x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32xf32>>
    return
}
// CHECK-LABEL: func.func @reduction
// CHECK:          lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:      partial_reduction = [0, 0, 1, 512]
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 1], [0, 1, 2, 3]
// CHECK-SAME:      thread = [0, 0, 1, 8], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]
// CHECK-SAME:      workgroup = [1, 1, 0, 0]
