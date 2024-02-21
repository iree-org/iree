module @parameter_example {
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

// RUN: iree-compile %s \
// RUN:   --iree-hal-target-backends=vmvx \
// RUN:   --iree-opt-splat-parameter-archive-export-file=%t.irpa | \
// RUN: iree-run-module --device=local-task --module=- \
// RUN:   --input=1x2xi32=1 \
// RUN:   --parameters=model=%t.irpa \
// RUN:   --function=forward | FileCheck %s

// CHECK-LABEL: EXEC @forward
// CHECK: 1x2xi32=[0 0]
