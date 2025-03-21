// RUN: iree-opt %s \
// RUN:     --split-input-file \
// RUN:     --iree-consteval-jit-globals \
// RUN:     --iree-consteval-jit-target-device=local \
// RUN:     --iree-hal-local-host-device-backends=vmvx \
// RUN:     --iree-consteval-jit-debug \
// RUN:     --verify-diagnostics | \
// RUN:     FileCheck %s

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

// -----

// expected-error @+2 {{resource data missing in input IR}}
// expected-error @+1 {{serializeToBuffer failed}}
util.global private @resource = dense_resource<missing> : tensor<f32>
util.initializer attributes {iree.compiler.consteval} {
  %0 = util.global.load @resource : tensor<f32>
  util.return
}
