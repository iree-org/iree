// aarch64:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=aarch64-none-elf \
// RUN:   --iree-input-demote-f64-to-f32=false \
// RUN:   --iree-hal-dump-executable-intermediates-to=- %s -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=AARCH64

// AARCH64-LABEL: {{^}}roundeven_f32_dispatch
// AARCH64:         frintn v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// AARCH64-LABEL: {{^}}roundeven_f64_dispatch
// AARCH64:         frintn v{{[0-9]+}}.2d, v{{[0-9]+}}.2d

func.func @roundeven_f32(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = math.roundeven %arg0 : tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

func.func @roundeven_f64(%arg0: tensor<1024xf64>) -> tensor<1024xf64> {
  %0 = math.roundeven %arg0 : tensor<1024xf64>
  return %0 : tensor<1024xf64>
}
