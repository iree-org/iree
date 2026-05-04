// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

#distribute_attention_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 0, 0, 128]]>
#distribute_attention_translation = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<Distribute> workgroup_size = [128, 1, 1] subgroup_size = 64>
#distribute_attention_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @attention_skinny_k2_distribute {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_skinny_k2_distribute ordinal(0) layout(#distribute_attention_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_skinny_k2_distribute() attributes {translation_info = #distribute_attention_translation} {
        %cst = arith.constant 1.0 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#distribute_attention_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x64x64xf16>>
        %1 = hal.interface.binding.subspan layout(#distribute_attention_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2x64xf16>>
        %2 = hal.interface.binding.subspan layout(#distribute_attention_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2x64xf16>>
        %3 = hal.interface.binding.subspan layout(#distribute_attention_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x64x64xf32>>
        %q = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x64x64xf16>> -> tensor<2x64x64xf16>
        %k = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, 2, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2x64xf16>> -> tensor<2x2x64xf16>
        %v = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [2, 2, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2x64xf16>> -> tensor<2x2x64xf16>
        %empty = tensor.empty() : tensor<2x64x64xf32>
        %empty_red = tensor.empty() : tensor<2x64xf32>
        %att:3 = iree_linalg_ext.online_attention
            {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                              affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                              affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                              affine_map<(d0, d1, d2, d3, d4) -> ()>,
                              affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
                              affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                              affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>],
             lowering_config = #distribute_attention_config}
            ins(%q, %k, %v, %cst : tensor<2x64x64xf16>, tensor<2x2x64xf16>, tensor<2x2x64xf16>, f16)
            outs(%empty, %empty_red, %empty_red : tensor<2x64x64xf32>, tensor<2x64xf32>, tensor<2x64xf32>) {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score : f32
        } -> tensor<2x64x64xf32>, tensor<2x64xf32>, tensor<2x64xf32>
        iree_tensor_ext.dispatch.tensor.store %att#0, %3, offsets = [0, 0, 0], sizes = [2, 64, 64], strides = [1, 1, 1] : tensor<2x64x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x64x64xf32>>
        return
      }
    }
  }
}

// CHECK:       #translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<Distribute> workgroup_size = [128, 1, 1] subgroup_size = 64>
// CHECK-LABEL: func.func @attention_skinny_k2_distribute
// CHECK-NOT:   iree_linalg_ext.online_attention
// CHECK:       linalg.generic
