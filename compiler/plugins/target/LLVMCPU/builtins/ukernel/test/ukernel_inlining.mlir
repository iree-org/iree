// RUN: rm -rf %t && mkdir -p %t

// x86-64, AVX-512 VNNI:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf \
// RUN:   --iree-llvmcpu-target-cpu=znver4 \
// RUN:   --iree-opt-data-tiling --iree-llvmcpu-enable-ukernels=all \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t/x86 %s -o /dev/null
// RUN: cat %t/x86/*.linked.ll | FileCheck %s --check-prefix=LINKED
// RUN: cat %t/x86/*.optimized.ll | FileCheck %s --check-prefix=OPT

// aarch64, i8mm:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=aarch64-none-elf \
// RUN:   --iree-llvmcpu-target-cpu-features=+i8mm \
// RUN:   --iree-opt-data-tiling --iree-llvmcpu-enable-ukernels=mmt4d \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t/i8mm %s -o /dev/null
// RUN: cat %t/i8mm/*.linked.ll | FileCheck %s --check-prefix=LINKED
// RUN: cat %t/i8mm/*.optimized.ll | FileCheck %s --check-prefix=OPT

// aarch64, dotprod:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=aarch64-none-elf \
// RUN:   --iree-llvmcpu-target-cpu-features=+dotprod \
// RUN:   --iree-opt-data-tiling --iree-llvmcpu-enable-ukernels=mmt4d \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t/dotprod %s -o /dev/null
// RUN: cat %t/dotprod/*.linked.ll | FileCheck %s --check-prefix=LINKED
// RUN: cat %t/dotprod/*.optimized.ll | FileCheck %s --check-prefix=OPT

// riscv64, RVV + xsmtvdot:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=riscv64-unknown-elf \
// RUN:   --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+xsmtvdot,+zvl256b \
// RUN:   --iree-opt-data-tiling --iree-llvmcpu-enable-ukernels=mmt4d \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t/rvv %s -o /dev/null
// RUN: cat %t/rvv/*.linked.ll | FileCheck %s --check-prefix=LINKED
// RUN: cat %t/rvv/*.optimized.ll | FileCheck %s --check-prefix=OPT

// Ukernels must inline into the dispatch so their compile-time specialization
// can be DCE'd. For each target above, check that a ukernel is linked in and
// that, after optimization, no call to a ukernel symbol survives in the dispatch.

// LINKED: call {{.*}}@iree_uk_mmt4d(
// OPT: define {{.*}}_dispatch_0
// OPT-NOT: call {{.*}}@iree_uk_

func.func @matmul(%a: tensor<128x128xi8>, %b: tensor<128x128xi8>)
    -> tensor<128x128xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<128x128xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<128x128xi32>)
      -> tensor<128x128xi32>
  %result = linalg.matmul ins(%a, %b : tensor<128x128xi8>, tensor<128x128xi8>)
      outs(%fill : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %result : tensor<128x128xi32>
}
