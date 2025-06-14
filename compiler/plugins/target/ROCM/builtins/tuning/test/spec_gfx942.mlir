// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-configure-target-executable-variants{target=rocm})))" \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-test-notify-transform-strategy-application \
// RUN:   --verify-diagnostics %s | FileCheck %s

// RUN: iree-opt --split-input-file --iree-gpu-test-target=mi300x@hip \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-configure-target-executable-variants{target=rocm})))" \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-test-notify-transform-strategy-application \
// RUN:   --verify-diagnostics %s | FileCheck %s --check-prefix=MI300X

// Check that the default configuration for mmt_2048x1280x5120_f16_f16_f32
// applies to the `linalg.matmul_transpose_b` below.

// CHECK-LABEL:  func.func @mmt_2048x1280x5120_f16_f16_f32
// CHECK:          linalg.generic
// CHECK-SAME:       __tuning_spec_applied__

// MI300X-LABEL:  func.func @mmt_2048x1280x5120_f16_f16_f32
// MI300X:          linalg.generic
// MI300X-SAME:       __tuning_spec_applied__

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {
    iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">
  }>) {
    hal.executable.export public @matmul_transpose_b ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
      func.func @mmt_2048x1280x5120_f16_f16_f32() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x5120xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x5120xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x1280xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 5120], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x5120xf16>> -> tensor<2048x5120xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 5120], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x5120xf16>> -> tensor<1280x5120xf16>
        %5 = tensor.empty() : tensor<2048x1280xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
        %7 = linalg.matmul_transpose_b
          ins(%3, %4 : tensor<2048x5120xf16>, tensor<1280x5120xf16>)
          outs(%6 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : tensor<2048x1280xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x1280xf32>>
        return
      }
    }
  }
}

// -----

// CHECK-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
// CHECK:          iree_linalg_ext.attention
// CHECK-SAME:       __tuning_spec_applied__

// MI300X-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
// MI300X:          iree_linalg_ext.attention
// MI300X-SAME:       __tuning_spec_applied__

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {
    iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">
  }>) {
    hal.executable.export public @attention ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
      func.func @attention_2x10x4096x64x64x64_f16(
        %query: tensor<2x10x4096x64xf16>,
        %key: tensor<2x10x64x64xf16>,
        %value: tensor<2x10x64x64xf16>
      ) -> tensor<2x10x4096x64xf16> {

        %cst = arith.constant 1.250000e-01 : f16
        %output = tensor.empty() : tensor<2x10x4096x64xf16>

        // Apply the attention operation directly to function inputs.
        %result = iree_linalg_ext.attention {
            indexing_maps = [#map, #map1, #map2, #map3, #map4]
        } ins(%query, %key, %value, %cst :
            tensor<2x10x4096x64xf16>, tensor<2x10x64x64xf16>, tensor<2x10x64x64xf16>, f16)
          outs(%output : tensor<2x10x4096x64xf16>) {
            ^bb0(%arg0: f32):
              iree_linalg_ext.yield %arg0 : f32
          } -> tensor<2x10x4096x64xf16>

        return %result : tensor<2x10x4096x64xf16>
      }
    }
  }
}

// -----

// CHECK-LABEL:  func.func @attention_3x10x4096x64x64x32_f16
// CHECK:          iree_linalg_ext.attention
// CHECK-NOT:       __tuning_spec_applied__

// MI300X-LABEL:  func.func @attention_3x10x4096x64x64x32_f16
// MI300X:          iree_linalg_ext.attention
// MI300X-NOT:       __tuning_spec_applied__

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {
    iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">
  }>) {
    hal.executable.export public @attention ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
      func.func @attention_3x10x4096x64x64x32_f16(
        %query: tensor<3x10x4096x64xf16>,
        %key: tensor<3x10x32x64xf16>,
        %value: tensor<3x10x64x32xf16>
      ) -> tensor<3x10x4096x64xf16> {

        %cst = arith.constant 1.250000e-01 : f16
        %output = tensor.empty() : tensor<3x10x4096x64xf16>

        // Apply the attention operation directly to function inputs.
        %result = iree_linalg_ext.attention {
            indexing_maps = [#map, #map1, #map2, #map3, #map4]
        } ins(%query, %key, %value, %cst :
            tensor<3x10x4096x64xf16>, tensor<3x10x32x64xf16>, tensor<3x10x64x32xf16>, f16)
          outs(%output : tensor<3x10x4096x64xf16>) {
            ^bb0(%arg0: f32):
              iree_linalg_ext.yield %arg0 : f32
          } -> tensor<3x10x4096x64xf16>

        return %result : tensor<3x10x4096x64xf16>
      }
    }
  }
}
