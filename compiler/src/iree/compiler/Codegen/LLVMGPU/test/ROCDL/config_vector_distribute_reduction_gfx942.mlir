// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// CHECK:      #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @attention_20x1x64x4096x64() {
  %cst = arith.constant 1.250000e-01 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x1x64xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x1x64xf16>> -> tensor<20x1x64xf16>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %7 = tensor.empty() : tensor<20x1x64xf16>
  %8 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
               affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
               affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
               affine_map<(d0, d1, d2, d3, d4) -> ()>,
               affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
               ins(%4, %5, %6, %cst : tensor<20x1x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x1x64xf16>) {
                ^bb0(%score: f32):
                  iree_linalg_ext.yield %score : f32
               } -> tensor<20x1x64xf16>
  flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : tensor<20x1x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
  return
}

// CHECK:      decomposition_config =
// CHECK-SAME:  pv_attrs =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 1, 4], [0, 1, 3, 4]{{\]}}
// CHECK-SAME:      thread_basis = {{\[}}[1, 1, 64, 1, 1], [0, 1, 3, 4]{{\]}}
// CHECK-SAME:  qk_attrs =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 1, 4], [0, 1, 2, 3]{{\]}}
// CHECK-SAME:      thread_basis = {{\[}}[1, 1, 64, 1, 1], [0, 1, 2, 3]{{\]}}
// CHECK-SAME:  lowering_config =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      reduction = [0, 0, 0, 1, 0]
// CHECK-SAME:      workgroup = [1, 0, 0, 0, 4]
