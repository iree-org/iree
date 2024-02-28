// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-generalize-named-convolution-ops))" %s | FileCheck %s

func.func @nhwc_convolution(%arg0: tensor<1x1x32x32xf16>, %arg1: tensor<1x1x32x128xf16>) -> tensor<1x1x32x128xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1x1x32x128xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1x1x32x128xf16>) -> tensor<1x1x32x128xf16>
  %2 = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : vector<2xi64>,
    strides = dense<1> : vector<2xi64>,
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 128, 1, 1, 32]]>
  }
         ins(%arg0, %arg1 : tensor<1x1x32x32xf16>, tensor<1x1x32x128xf16>)
         outs(%1 : tensor<1x1x32x128xf16>) -> tensor<1x1x32x128xf16>
  return %2 : tensor<1x1x32x128xf16>
}

//               CHECK: #[[$CONFIG:.+]] = #iree_codegen.lowering_config
// CHECK-SAME{LITERAL}: <tile_sizes = [[1, 1, 32, 128, 1, 1, 32]]>

// CHECK-LABEL: func.func @nhwc_convolution
//       CHECK:   linalg.generic
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]
