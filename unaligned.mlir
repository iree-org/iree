// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))"

#config = #iree_gpu.lowering_config< {workgroup = [0, 4, 0],
                                      partial_reduction = [0, 0, 128],
                                      thread = [0, 0, 8],
                                      subgroup_basis = [[1, 1, 1], [0, 1, 2]],
                                      thread_basis   = [[1, 4, 16], [0, 1, 2]]}
>
#translation = #iree_codegen.translation_info< pipeline = LLVMGPUVectorDistribute
                                               workgroup_size = [64, 1, 1]
                                               subgroup_size = 64, {}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4099xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4099xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4099], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4099xf32>> -> tensor<1x4099xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4099], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4099xf16>> -> tensor<32000x4099xf16>
        %tempty = tensor.empty() : tensor<1x4099xf16>
        %trunc = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"] } ins(%3 : tensor<1x4099xf32>) outs(%tempty : tensor<1x4099xf16>) {
        ^bb0(%in : f32, %out : f16):
          %trunced = arith.truncf %in : f32 to f16
          linalg.yield %trunced : f16
        } -> tensor<1x4099xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%trunc, %4 : tensor<1x4099xf16>, tensor<32000x4099xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}