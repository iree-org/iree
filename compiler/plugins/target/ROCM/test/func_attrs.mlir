// RUN: iree-opt --split-input-file --iree-hal-resolve-device-aliases \
// RUN:   --iree-hal-target-device=hip --iree-hip-target=gfx90a %s | FileCheck --check-prefix=CHECK-NONE %s

// RUN: iree-opt --split-input-file --iree-hal-resolve-device-aliases \
// RUN:   --iree-hal-target-device=hip --iree-hip-target=gfx90a \
// RUN:   --iree-hip-denormal-fp-math-f32=ieee --iree-hip-waves-per-eu=1 %s | FileCheck --check-prefix=CHECK-IEEE %s

// RUN: iree-opt --split-input-file --iree-hal-resolve-device-aliases \
// RUN:   --iree-hal-target-device=hip --iree-hip-target=gfx90a \
// RUN:   --iree-hip-denormal-fp-math-f32=positive-zero --iree-hip-waves-per-eu=2 %s | FileCheck --check-prefix=CHECK-POSITIVE-ZERO %s

// RUN: iree-opt --split-input-file --iree-hal-resolve-device-aliases \
// RUN:   --iree-hal-target-device=hip --iree-hip-target=gfx90a \
// RUN:   --iree-hip-denormal-fp-math-f32=preserve-sign --iree-hip-waves-per-eu=3 %s | FileCheck --check-prefix=CHECK-PRESERVE-SIGN %s

// RUN: iree-opt --split-input-file --iree-hal-resolve-device-aliases \
// RUN:   --iree-hal-target-device=hip --iree-hip-target=gfx90a \
// RUN:   --iree-hip-denormal-fp-math-f32=dynamic --iree-hip-waves-per-eu=4 %s | FileCheck --check-prefix=CHECK-DYNAMIC %s

module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  // CHECK-NONE-LABEL: #hal.executable.target<
  // CHECK-NONE-NOT: denormal_fp_math_f32
  // CHECK-NONE-NOT: waves_per_eu

  // CHECK-IEEE-LABEL: #hal.executable.target<
  // CHECK-IEEE-SAME: denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<ieee>
  // CHECK-IEEE-SAME: waves_per_eu = 1

  // CHECK-POSITIVE-ZERO-LABEL: #hal.executable.target<
  // CHECK-POSITIVE-ZERO-SAME: denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"positive-zero">
  // CHECK-POSITIVE-ZERO-SAME: waves_per_eu = 2

  // CHECK-PRESERVE-SIGN-LABEL: #hal.executable.target<
  // CHECK-PRESERVE-SIGN-SAME: denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">
  // CHECK-PRESERVE-SIGN-SAME: waves_per_eu = 3

  // CHECK-DYNAMIC-LABEL: #hal.executable.target<
  // CHECK-DYNAMIC-SAME: denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<dynamic>
  // CHECK-DYNAMIC-SAME: waves_per_eu = 4
  util.global private @__device_0 = #hal.device.alias<"hip"> : !hal.device
  util.func public @softmax_static_10x256x256xf32() {
    util.return
  }
}
