// RUN: iree-opt --split-input-file --iree-consteval-jit-globals --iree-consteval-jit-target-device=vmvx --iree-consteval-jit-debug --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @eval_f16_tensor
module @eval_f16_tensor {
  util.global private @hoisted : tensor<5x6xf16>
  util.func public @main() -> tensor<5x6xf16> {
    %hoisted = util.global.load @hoisted : tensor<5x6xf16>
    util.return %hoisted : tensor<5x6xf16>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<2.0e+2> : tensor<5x6xf16>
    util.global.store %cst, @hoisted : tensor<5x6xf16>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_bf16_tensor
module @eval_bf16_tensor {
  util.global private @hoisted : tensor<5x6xbf16>
  util.func public @main() -> tensor<5x6xbf16> {
    %hoisted = util.global.load @hoisted : tensor<5x6xbf16>
    util.return %hoisted : tensor<5x6xbf16>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<2.0e+2> : tensor<5x6xbf16>
    util.global.store %cst, @hoisted : tensor<5x6xbf16>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_f64_tensor
module @eval_f64_tensor {
  util.global private @hoisted : tensor<2xf64>
  util.func public @main() -> tensor<2xf64> {
    %hoisted = util.global.load @hoisted : tensor<2xf64>
    util.return %hoisted : tensor<2xf64>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2.0e+2, 3.2e+3]> : tensor<2xf64>
    util.global.store %cst, @hoisted : tensor<2xf64>
    util.return
  }
}

// -----
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
