// x86-64, SSE4.1:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf \
// RUN:   --iree-llvmcpu-target-cpu-features=+sse4.1 \
// RUN:   --iree-input-demote-f64-to-f32=false \
// RUN:   --iree-hal-dump-executable-intermediates-to=- %s -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=SSE41

// x86-64, AVX2 (implies SSE4.1):
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf \
// RUN:   --iree-llvmcpu-target-cpu-features=+avx2 \
// RUN:   --iree-input-demote-f64-to-f32=false \
// RUN:   --iree-hal-dump-executable-intermediates-to=- %s -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=AVX2

// Immediate 8 is round-to-nearest-even with the precision exception
// suppressed.
// SSE41-LABEL: {{^}}roundeven_f32_dispatch
// SSE41:         roundps $8, {{.*}}%xmm
// SSE41-LABEL: {{^}}roundeven_f64_dispatch
// SSE41:         roundpd $8, {{.*}}%xmm

// AVX2-LABEL: {{^}}roundeven_f32_dispatch
// AVX2:         vroundps $8, {{.*}}%ymm
// AVX2-LABEL: {{^}}roundeven_f64_dispatch
// AVX2:         vroundpd $8, {{.*}}%ymm

func.func @roundeven_f32(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = math.roundeven %arg0 : tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

func.func @roundeven_f64(%arg0: tensor<1024xf64>) -> tensor<1024xf64> {
  %0 = math.roundeven %arg0 : tensor<1024xf64>
  return %0 : tensor<1024xf64>
}
