// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-configure-target-executable-variants{target=rocm})))" \
// RUN:   --iree-hip-enable-tensor-ukernels \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Make sure we can match and insert a tensor ukernel.

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_f8 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f8() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128x4096xf8E4M3FNUZ>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf8E4M3FNUZ>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x128x1024xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 128, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128x4096xf8E4M3FNUZ>> -> tensor<1x128x4096xf8E4M3FNUZ>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf8E4M3FNUZ>> -> tensor<1024x4096xf8E4M3FNUZ>
        %5 = tensor.empty() : tensor<1x128x1024xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
        %7 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x128x4096xf8E4M3FNUZ>, tensor<1024x4096xf8E4M3FNUZ>) outs(%6 : tensor<1x128x1024xf32>) {
        ^bb0(%in: f8E4M3FNUZ, %in_4: f8E4M3FNUZ, %out: f32):
          %12 = arith.extf %in : f8E4M3FNUZ to f32
          %13 = arith.extf %in_4 : f8E4M3FNUZ to f32
          %14 = arith.mulf %12, %13 : f32
          %15 = arith.addf %out, %14 : f32
          linalg.yield %15 : f32
        } -> tensor<1x128x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [1, 128, 1024], strides = [1, 1, 1] : tensor<1x128x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x128x1024xf32>>
        return
      }
    }
  }
}
// CHECK:      #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
// CHECK:      func.func @matmul_f8
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK:        linalg.generic
// CHECK-SAME:     iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"pingpong_medium_f8_expanded", tensor>
// CHECK-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK:        util.func private @pingpong_medium_f8_expanded
// CHECK:          iree_codegen.inner_tiled
