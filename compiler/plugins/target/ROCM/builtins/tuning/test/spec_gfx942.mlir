// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-configure-target-executable-variants{target=rocm})))" \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-notify-transform-strategy-application \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:      func.func @placeholder

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @placeholder ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
      func.func @placeholder() {
        return
      }
    }
  }
}
