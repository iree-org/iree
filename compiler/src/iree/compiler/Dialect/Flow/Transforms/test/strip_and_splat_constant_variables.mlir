// RUN: iree-opt --split-input-file --iree-flow-strip-and-splat-constant-variables %s | FileCheck %s

func.func @fn() -> () {
  // CHECK: util.global private @float_0 {noinline} = dense<1.000000e+00> : tensor<3xf32>
  util.global private @float_0 = dense<"0x012345670123456701234567"> : tensor<3xf32>
  // CHECK: util.global private @float_1 {noinline} = dense<5.000000e-01> : tensor<3xf32>
  util.global private @float_1 = dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xf32>
  // CHECK: util.global private @int_1 {noinline} = dense<3> : tensor<3xi32>
  util.global private @int_1 = dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xi32>
  return
}

// -----

func.func @fn() -> (tensor<3xf32>, tensor<3xf32>, tensor<3xi32>) {
  // CHECK: arith.constant dense<1.000000e+00> : tensor<3xf32>
  %0 = arith.constant dense<"0x012345670123456701234567"> : tensor<3xf32>
  // CHECK: arith.constant dense<5.000000e-01> : tensor<3xf32>
  %1 = arith.constant dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xf32>
  // CHECK: arith.constant dense<3> : tensor<3xi32>
  %2 = arith.constant dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xi32>
  return %0, %1, %2 : tensor<3xf32>, tensor<3xf32>, tensor<3xi32>
}
