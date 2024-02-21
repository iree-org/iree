// RUN: iree-opt --split-input-file --iree-util-strip-and-splat-constants %s | FileCheck %s

module {
  // CHECK: util.global private @float_0 = #util.byte_pattern<1> : tensor<3xf32>
  util.global private @float_0 = dense<"0x012345670123456701234567"> : tensor<3xf32>
  // CHECK: util.global private @float_1 = #util.byte_pattern<2> : tensor<3xf32>
  util.global private @float_1 = dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xf32>
  // CHECK: util.global private @int_1 = #util.byte_pattern<3> : tensor<3xi32>
  util.global private @int_1 = dense<"0x89ABCDEF89ABCDEF89ABCDEF"> : tensor<3xi32>
}
