// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @collatz
func @collatz() -> tensor<f32> {
  %arg0 = iree.unfoldable_constant dense<178.0> : tensor<f32>
  %0 = xla_hlo.constant dense<1.0> : tensor<f32>
  %1 = xla_hlo.constant dense<3.0> : tensor<f32>
  %2 = xla_hlo.constant dense<2.0> : tensor<f32>
  %3 = xla_hlo.constant dense<0.0> : tensor<f32>
  br ^bb1(%3, %arg0 : tensor<f32>, tensor<f32>)
^bb1(%4: tensor<f32>, %5: tensor<f32>):
  %6 = "xla_hlo.compare"(%5, %0) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %7 = extract_element %6[] : tensor<i1>
  cond_br %7, ^bb2(%4, %5 : tensor<f32>, tensor<f32>), ^bb6(%4 : tensor<f32>)
^bb2(%8: tensor<f32>, %9: tensor<f32>):
  %10 = xla_hlo.add %8, %0 : tensor<f32>
  %11 = xla_hlo.remainder %9, %2 : tensor<f32>
  %12 = "xla_hlo.compare"(%11, %3) {comparison_direction = "NE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %13 = "xla_hlo.compare"(%2, %3) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %14 = "xla_hlo.compare"(%11, %3) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %15 = "xla_hlo.compare"(%13, %14) {comparison_direction = "NE"} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %16 = xla_hlo.and %12, %15 : tensor<i1>
  %17 = xla_hlo.add %11, %2 : tensor<f32>
  %18 = "xla_hlo.select"(%16, %17, %11) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %19 = "xla_hlo.compare"(%18, %3) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %20 = extract_element %19[] : tensor<i1>
  cond_br %20, ^bb3, ^bb4
^bb3: // pred: ^bb2
  %21 = xla_hlo.mul %9, %1 : tensor<f32>
  %22 = xla_hlo.add %21, %0 : tensor<f32>
  br ^bb5(%22 : tensor<f32>)
^bb4: // pred: ^bb2
  %23 = xla_hlo.div %9, %2 : tensor<f32>
  br ^bb5(%23 : tensor<f32>)
^bb5(%24: tensor<f32>): // 2 preds: ^bb3, ^bb4
  br ^bb1(%10, %24 : tensor<f32>, tensor<f32>)
^bb6(%25: tensor<f32>): // pred: ^bb1
  return %25 : tensor<f32>
}
// CHECK: f32=31
