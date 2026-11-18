// riscv64, RVV with D:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=riscv64-unknown-elf \
// RUN:   --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v \
// RUN:   --iree-input-demote-f64-to-f32=false \
// RUN:   --iree-hal-dump-executable-intermediates-to=- %s -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=RISCV64

// Rounding mode 0 is round-to-nearest-even.
// RISCV64-LABEL: {{^}}roundeven_f32_dispatch
// RISCV64:         fsrmi [[RM:[a-z0-9]+]], 0
// RISCV64:         vfcvt.x.f.v
// RISCV64:         fsrm [[RM]]
// RISCV64-LABEL: {{^}}roundeven_f64_dispatch
// RISCV64:         fsrmi [[RM:[a-z0-9]+]], 0
// RISCV64:         vfcvt.x.f.v
// RISCV64:         fsrm [[RM]]

func.func @roundeven_f32(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = math.roundeven %arg0 : tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

func.func @roundeven_f64(%arg0: tensor<1024xf64>) -> tensor<1024xf64> {
  %0 = math.roundeven %arg0 : tensor<1024xf64>
  return %0 : tensor<1024xf64>
}
