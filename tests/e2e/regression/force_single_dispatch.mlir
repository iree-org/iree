// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-preprocessing-make-single-dispatch-for-function))" %s | iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --input="1" -

func.func @simple_test_with_cfg(%arg0 : i8) -> tensor<2x4xf32> {
    %c0_i8 = arith.constant 0 : i8
    %cond = arith.cmpi eq, %arg0, %c0_i8 : i8
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %0 = tensor.empty() : tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  ^bb2:
    %1 = arith.constant dense<1.0> : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
}
// CHECK-LABEL: EXEC @simple_test_with_cfg
//       CHECK: 2x4xf32=[1 1 1 1][1 1 1 1]
