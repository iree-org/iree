// RUN: rm -rf %t && mkdir -p %t

// riscv64, RVV + xsmtvdot:
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=riscv64-unknown-elf \
// RUN:   --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+xsmtvdot,+zvl256b \
// RUN:   --iree-opt-data-tiling --iree-llvmcpu-enable-ukernels=mmt4d \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t/rvv %s -o /dev/null
// RUN: cat %t/rvv/*.linked.ll | FileCheck %s --check-prefix=LINKED
// RUN: cat %t/rvv/*.optimized.ll | FileCheck %s --check-prefix=OPT

// Same checks as ukernel_inlining.mlir, on riscv64. This test needs the RISCV
// LLVM target and is tagged requires-llvm-target-riscv in BUILD.bazel so that
// builds without that target can skip it.

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
