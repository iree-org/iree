// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-configure-target-executable-variants{target=rocm})))" \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-notify-transform-strategy-application \
// RUN:   --verify-diagnostics %s | FileCheck %s

// RUN: iree-opt --split-input-file --iree-gpu-test-target=mi300x@hip \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-configure-target-executable-variants{target=rocm})))" \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-notify-transform-strategy-application \
// RUN:   --verify-diagnostics %s | FileCheck %s --check-prefix=MI308X

// Check that the default configuration for mmt_2048x1280x5120_f16_f16_f32
// applies to the `linalg.matmul_transpose_b` below.

// CHECK-LABEL:  func.func @mmt_2048x1280x5120_f16_f16_f32
// CHECK:          linalg.generic
// CHECK-SAME:       __tuning_spec_applied__

// MI308X-LABEL:  func.func @mmt_2048x1280x5120_f16_f16_f32
// MI308X:          linalg.generic
// MI308X-SAME:       __tuning_spec_applied__

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
      func.func @mmt_2048x1280x5120_f16_f16_f32() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x5120xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x5120xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x1280xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 5120], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x5120xf16>> -> tensor<2048x5120xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 5120], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x5120xf16>> -> tensor<1280x5120xf16>
        %5 = tensor.empty() : tensor<2048x1280xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
        %7 = linalg.matmul_transpose_b
          ins(%3, %4 : tensor<2048x5120xf16>, tensor<1280x5120xf16>)
          outs(%6 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : tensor<2048x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x1280xf32>>
        return
      }
    }
  }
}
