// RUN: mkdir -p %t-dump
// RUN: iree-compile %s -output-format=vm-asm -o %t.mlir --iree-hal-target-backends=llvm-cpu -iree-llvmcpu-target-triple=riscv64-linux-gnu \
// RUN:   --iree-llvmcpu-target-cpu-features='+m,+d,+zvl1024b,+v' --iree-hal-dump-executable-intermediates-to=%t-dump
// RUN: cat %t-dump/*.codegen.ll | FileCheck %s --check-prefixes=FEATURES-ONLY,CHECK
// RUN: iree-compile %s -output-format=vm-asm -o %t.mlir --iree-hal-target-backends=llvm-cpu -iree-llvmcpu-target-triple=riscv64-linux-gnu \
// RUN:   --iree-llvmcpu-target-cpu=sifive-x390 --iree-hal-dump-executable-intermediates-to=%t-dump
// RUN: cat %t-dump/*.codegen.ll | FileCheck %s --check-prefixes=TARGET-CPU,CHECK

// CHECK: attributes
// FEATURES-ONLY-NOT: "target-cpu"
// TARGET-CPU-SAME: "target-cpu"="sifive-x390"
// CHECK-SAME: "target-features"="{{.*}}zvl1024b
func.func @simple(%arg0: tensor<2xf32>, %arg1 : tensor<2xf32>) -> tensor<2xf32> {
  %0 = tensor.empty() : tensor<2xf32>
  %r = linalg.add ins(%arg0, %arg1 : tensor<2xf32>, tensor<2xf32>) outs(%0: tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
