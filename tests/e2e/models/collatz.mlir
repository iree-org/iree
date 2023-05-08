// RUN: iree-run-mlir --Xcompiler,iree-input-type=mhlo --Xcompiler,iree-hal-target-backends=vmvx %s | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-input-type=mhlo --Xcompiler,iree-hal-target-backends=vulkan-spirv %s | FileCheck %s)

// CHECK-LABEL: EXEC @collatz
func.func @collatz() -> tensor<f32> {
  %arg0 = util.unfoldable_constant dense<178.0> : tensor<f32>
  %0 = mhlo.constant dense<1.0> : tensor<f32>
  %1 = mhlo.constant dense<3.0> : tensor<f32>
  %2 = mhlo.constant dense<2.0> : tensor<f32>
  %3 = mhlo.constant dense<0.0> : tensor<f32>
  cf.br ^bb1(%3, %arg0 : tensor<f32>, tensor<f32>)
^bb1(%4: tensor<f32>, %5: tensor<f32>):
  %6 = "mhlo.compare"(%5, %0) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %7 = tensor.extract %6[] : tensor<i1>
  cf.cond_br %7, ^bb2(%4, %5 : tensor<f32>, tensor<f32>), ^bb6(%4 : tensor<f32>)
^bb2(%8: tensor<f32>, %9: tensor<f32>):
  %10 = mhlo.add %8, %0 : tensor<f32>
  %11 = mhlo.remainder %9, %2 : tensor<f32>
  %12 = "mhlo.compare"(%11, %3) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %13 = "mhlo.compare"(%2, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %14 = "mhlo.compare"(%11, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %15 = "mhlo.compare"(%13, %14) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %16 = mhlo.and %12, %15 : tensor<i1>
  %17 = mhlo.add %11, %2 : tensor<f32>
  %18 = "mhlo.select"(%16, %17, %11) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %19 = "mhlo.compare"(%18, %3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %20 = tensor.extract %19[] : tensor<i1>
  cf.cond_br %20, ^bb3, ^bb4
^bb3: // pred: ^bb2
  %21 = mhlo.multiply %9, %1 : tensor<f32>
  %22 = mhlo.add %21, %0 : tensor<f32>
  cf.br ^bb5(%22 : tensor<f32>)
^bb4: // pred: ^bb2
  %23 = mhlo.divide %9, %2 : tensor<f32>
  cf.br ^bb5(%23 : tensor<f32>)
^bb5(%24: tensor<f32>): // 2 preds: ^bb3, ^bb4
  cf.br ^bb1(%10, %24 : tensor<f32>, tensor<f32>)
^bb6(%25: tensor<f32>): // pred: ^bb1
  return %25 : tensor<f32>
}
// CHECK: f32=31
