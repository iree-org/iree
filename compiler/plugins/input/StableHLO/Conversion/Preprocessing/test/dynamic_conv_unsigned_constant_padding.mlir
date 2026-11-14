// The upstream DynamicConvOp verifier reads constant padding as signed and
// asserts on the negative bound it derives from an unsigned value with its
// high bit set. Parsing therefore skips verification; the pass output
// carries signed i64 padding and is verified as usual.
// RUN: iree-opt --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --iree-stablehlo-to-stablehlo-preprocessing %s | FileCheck %s

// The upstream static-padding matcher reads APInt as signed; widen first.
// CHECK-LABEL: @dynamic_conv_ui8_constant_padding
// CHECK: %[[P:.+]] = stablehlo.constant dense<{{\[\[128, 0\]\]}}> : tensor<1x2xi64>
// CHECK: "stablehlo.dynamic_conv"(%arg0, %arg1, %[[P]])
func.func @dynamic_conv_ui8_constant_padding(%a: tensor<1x2x1xf32>, %k: tensor<1x1x1xf32>) -> tensor<1x?x1xf32> {
  %p = stablehlo.constant dense<[[128, 0]]> : tensor<1x2xui8>
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1>, lhs_dilation = array<i64: 1>, rhs_dilation = array<i64: 1>
  } : (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x2xui8>) -> tensor<1x?x1xf32>
  return %r : tensor<1x?x1xf32>
}
