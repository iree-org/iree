// RUN: iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=vmvx %s | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=llvm-cpu %s | FileCheck %s

// CHECK-LABEL: EXEC @collatz
func.func @collatz() -> tensor<f32> {
  %0 = util.unfoldable_constant dense<1.780000e+02> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  %3 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  cf.br ^bb1(%4, %0 : tensor<f32>, tensor<f32>)
^bb1(%5: tensor<f32>, %6: tensor<f32>):  // 2 preds: ^bb0, ^bb5
  %7 = stablehlo.compare  GT, %6, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %extracted = tensor.extract %7[] : tensor<i1>
  cf.cond_br %extracted, ^bb2(%5, %6 : tensor<f32>, tensor<f32>), ^bb6(%5 : tensor<f32>)
^bb2(%8: tensor<f32>, %9: tensor<f32>):  // pred: ^bb1
  %10 = stablehlo.add %8, %1 : tensor<f32>
  %11 = stablehlo.remainder %9, %3 : tensor<f32>
  %12 = stablehlo.compare  NE, %11, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %13 = stablehlo.constant dense<false> : tensor<i1>
  %14 = stablehlo.compare  LT, %11, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %15 = stablehlo.compare  NE, %13, %14 : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %16 = stablehlo.and %12, %15 : tensor<i1>
  %17 = stablehlo.add %11, %3 : tensor<f32>
  %18 = stablehlo.select %16, %17, %11 : tensor<i1>, tensor<f32>
  %19 = stablehlo.compare  GT, %18, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %extracted_0 = tensor.extract %19[] : tensor<i1>
  cf.cond_br %extracted_0, ^bb3, ^bb4
^bb3:  // pred: ^bb2
  %20 = stablehlo.multiply %9, %2 : tensor<f32>
  %21 = stablehlo.add %20, %1 : tensor<f32>
  cf.br ^bb5(%21 : tensor<f32>)
^bb4:  // pred: ^bb2
  %22 = stablehlo.divide %9, %3 : tensor<f32>
  cf.br ^bb5(%22 : tensor<f32>)
^bb5(%23: tensor<f32>):  // 2 preds: ^bb3, ^bb4
  cf.br ^bb1(%10, %23 : tensor<f32>, tensor<f32>)
^bb6(%24: tensor<f32>):  // pred: ^bb1
  return %24 : tensor<f32>
}
// CHECK: f32=31
