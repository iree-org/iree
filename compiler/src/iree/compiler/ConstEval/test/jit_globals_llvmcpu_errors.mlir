// RUN: iree-opt --split-input-file --iree-consteval-jit-globals --iree-consteval-jit-target-device=llvm-cpu --iree-consteval-jit-debug --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @eval_i4_tensor
module @eval_i4_tensor {
  util.global private @hoisted : tensor<5x6xi4>
  util.func public @main() -> tensor<5x6xi4> {
    %hoisted = util.global.load @hoisted : tensor<5x6xi4>
    util.return %hoisted : tensor<5x6xi4>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<3> : tensor<5x6xi4>
    util.global.store %cst, @hoisted : tensor<5x6xi4>
    util.return
  }
}
