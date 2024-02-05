// RUN: iree-opt --pass-pipeline="builtin.module(iree-io-export-parameters{archive-path="%t.irpa" scope=opt minimum-size=0})" %s | FileCheck %s

// CHECK-LABEL: module @parameter_example
module @parameter_example {
// CHECK-DAG: util.global private @array_global_0 = #stream.parameter.named<"opt"::"array_global_0"> : tensor<1x2xf32>
// CHECK-DAG: util.global private @dense_global_1 = #stream.parameter.named<"opt"::"dense_global_1"> : tensor<2x2xf32>
// CHECK-DAG: util.global private @constant_hoisted = #stream.parameter.named<"opt"::"constant_hoisted"> : tensor<1x2xf32>
// CHECK-DAG: util.global private @dense_global_2 = #stream.parameter.named<"opt"::"dense_global_2"> : tensor<2x2xf32>
  util.global private @array_global_0 = dense<[[11.0, 12.0]]> : tensor<1x2xf32>
  util.global private @dense_global_1 = dense<"0x0000E040000000410000104100002041"> : tensor<2x2xf32>
  util.global private @dense_global_2 = dense<"0x0000803F000000400000404000008040"> : tensor<2x2xf32>
  func.func @parameter_example(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %3 = util.global.load @array_global_0 : tensor<1x2xf32>
    %4 = util.global.load @dense_global_1 : tensor<2x2xf32>
    %5 = arith.constant dense<"0x0000A0400000C040"> : tensor<1x2xf32>
    %6 = util.global.load @dense_global_2 : tensor<2x2xf32>
    %empty = tensor.empty() : tensor<1x2xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %8 = linalg.matmul ins(%arg0, %6 : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
    %10 = linalg.add ins(%8, %5 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %12 = linalg.matmul ins(%10, %4 : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
    %14 = linalg.add ins(%12, %3 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    return %14 : tensor<1x2xf32>
  }
}
