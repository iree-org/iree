// RUN: iree-opt -split-input-file -iree-flow-strip-and-splat-constant-variables %s | IreeFileCheck %s

func @fn() -> () {
  // CHECK: flow.variable @float_0 dense<1.000000e+00> : tensor<3xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @float_0 dense<"0x012345670123456701234567"> : tensor<3xf32> attributes {sym_visibility = "private"}
  // CHECK: flow.variable @float_1 dense<5.000000e-01> : tensor<3xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @float_1 dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xf32> attributes {sym_visibility = "private"}
  // CHECK: flow.variable @int_1 dense<3> : tensor<3xi32> attributes {noinline, sym_visibility = "private"}
  flow.variable @int_1 dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xi32> attributes {sym_visibility = "private"}
  return
}
