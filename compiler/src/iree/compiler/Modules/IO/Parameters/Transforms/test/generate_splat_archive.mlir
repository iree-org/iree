// RUN: iree-opt --pass-pipeline="builtin.module(iree-io-generate-splat-parameter-archive{archive-path="%t.irpa"})" %s | FileCheck %s
// RUN: iree-dump-parameters --parameters=%t.irpa | FileCheck %s --check-prefix=DUMP

// CHECK-LABEL: @parameter_example
module @parameter_example {
  // CHECK: util.global private @array_global_0 = #stream.parameter.named<"model"::"global_0">
  // CHECK: util.global private @dense_global_1 = #stream.parameter.named<"model"::"global_1">
  // CHECK: util.global private @dense_global_2 = #stream.parameter.named<"model"::"global_2">
  // CHECK: util.global private @dense_global_3 = #stream.parameter.named<"model"::"global_3">
  util.global private @array_global_0 = #stream.parameter.named<"model"::"global_0"> : tensor<1x2xi32>
  util.global private @dense_global_1 = #stream.parameter.named<"model"::"global_1"> : tensor<2x2xi32>
  util.global private @dense_global_2 = #stream.parameter.named<"model"::"global_2"> : tensor<1x2xi32>
  util.global private @dense_global_3 = #stream.parameter.named<"model"::"global_3"> : tensor<2x2xi32>
  func.func @forward(%arg0: tensor<1x2xi32>) -> tensor<1x2xi32> {
    %cst = arith.constant 0 : i32
    %3 = util.global.load @array_global_0 : tensor<1x2xi32>
    %4 = util.global.load @dense_global_1 : tensor<2x2xi32>
    %5 = util.global.load @dense_global_2 : tensor<1x2xi32>
    %6 = util.global.load @dense_global_3 : tensor<2x2xi32>
    %empty = tensor.empty() : tensor<1x2xi32>
    %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<1x2xi32>) -> tensor<1x2xi32>
    %8 = linalg.matmul ins(%arg0, %6 : tensor<1x2xi32>, tensor<2x2xi32>) outs(%fill : tensor<1x2xi32>) -> tensor<1x2xi32>
    %10 = linalg.add ins(%8, %5 : tensor<1x2xi32>, tensor<1x2xi32>) outs(%empty : tensor<1x2xi32>) -> tensor<1x2xi32>
    %12 = linalg.matmul ins(%10, %4 : tensor<1x2xi32>, tensor<2x2xi32>) outs(%fill : tensor<1x2xi32>) -> tensor<1x2xi32>
    %14 = linalg.add ins(%12, %3 : tensor<1x2xi32>, tensor<1x2xi32>) outs(%empty : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %14 : tensor<1x2xi32>
  }
}

// Verify the generated archive is what we expect.
// DUMP: - |{{.*}} - |{{.*}} 8 | `global_0`
// DUMP: - |{{.*}} - |{{.*}} 16 | `global_1`
// DUMP: - |{{.*}} - |{{.*}} 8 | `global_2`
// DUMP: - |{{.*}} - |{{.*}} 16 | `global_3`

