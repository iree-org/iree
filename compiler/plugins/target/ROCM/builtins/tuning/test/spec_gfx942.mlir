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

// -----

// CHECK-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
// CHECK:          iree_linalg_ext.attention
// CHECK-SAME:       __tuning_spec_applied__

// MI300X-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
// MI300X:          iree_linalg_ext.attention
// MI300X-SAME:       __tuning_spec_applied__

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
      func.func @attention_2x10x4096x64x64x64_f16() {
        %c85251584 = arith.constant 85251584 : index
        %c283904 = arith.constant 283904 : index
        %cst = arith.constant 1.250000e-01 : f16
        %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4:2 = util.assume.int
            %2[<umin = 101640704, umax = 101640704, udiv = 101640704>, <umin = 101640704, umax = 101640704, udiv = 101640704>, <umin = 74765824, umax = 74765824, udiv = 74765824>],
            %3[<umin = 91154944, umax = 91154944, udiv = 91154944>, <umin = 91154944, umax = 91154944, udiv = 91154944>, <umin = 64280064, umax = 64280064, udiv = 64280064>]
          : index, index
        %5 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c283904) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<3x2x10x64x64xf16>>
        %6 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c85251584) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<3x2x10x64x64xf16>>
        %7 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%4#0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>>
        %8 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%4#1) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x4096x10x64xf16>>
        %9 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [2, 10, 4096, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>> -> tensor<2x10x4096x64xf16>
        %10 = tensor.empty() : tensor<2x4096x10x64xf16>
        %11 = tensor.empty() : tensor<2x10x4096x64xf16>
        %12 = flow.dispatch.tensor.load %5, offsets = [2, 0, 0, 0, 0], sizes = [1, 2, 10, 64, 64], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x2x10x64x64xf16>> -> tensor<2x10x64x64xf16>
        %13 = flow.dispatch.tensor.load %6, offsets = [2, 0, 0, 0, 0], sizes = [1, 2, 10, 64, 64], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x2x10x64x64xf16>> -> tensor<2x10x64x64xf16>
        %14 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]} ins(%9, %13, %12, %cst : tensor<2x10x4096x64xf16>, tensor<2x10x64x64xf16>, tensor<2x10x64x64xf16>, f16) outs(%11 : tensor<2x10x4096x64xf16>) {
        ^bb0(%arg0: f32):
          iree_linalg_ext.yield %arg0 : f32
        } -> tensor<2x10x4096x64xf16>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<2x10x4096x64xf16>) outs(%10 : tensor<2x4096x10x64xf16>) {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<2x4096x10x64xf16>
        flow.dispatch.tensor.store %15, %8, offsets = [0, 0, 0, 0], sizes = [2, 4096, 10, 64], strides = [1, 1, 1, 1] : tensor<2x4096x10x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x4096x10x64xf16>>
        return
      }
    }
  }
}
