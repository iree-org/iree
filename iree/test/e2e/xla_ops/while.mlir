// NOTE: this has already been legalized to CFG form in the TF import tools.
func @while() attributes {iree.module.export} {
  %start = iree.unfoldable_constant dense<1> : tensor<i32>
  %bound = iree.unfoldable_constant dense<3> : tensor<i32>
  %cst_1 = constant dense<4> : tensor<i32>
  br ^bb1(%start : tensor<i32>)
^bb1(%2: tensor<i32>):
  %3 = "mhlo.compare"(%2, %bound) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %4 = tensor.extract %3[] : tensor<i1>
  cond_br %4, ^bb2(%2 : tensor<i32>), ^bb3(%2 : tensor<i32>)
^bb2(%5: tensor<i32>):
  %6 = mhlo.add %5, %5 : tensor<i32>
  br ^bb1(%6 : tensor<i32>)
^bb3(%7: tensor<i32>):
  check.expect_eq_const(%7, dense<4> : tensor<i32>) : tensor<i32>
  return
}
