// NOTE: this has already been legalized to CFG form in the TF import tools.
func.func @while() {
  %start = util.unfoldable_constant dense<1> : tensor<i32>
  %bound = util.unfoldable_constant dense<3> : tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<i32>
  cf.br ^bb1(%start : tensor<i32>)
^bb1(%2: tensor<i32>):
  %3 = "stablehlo.compare"(%2, %bound) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %4 = tensor.extract %3[] : tensor<i1>
  cf.cond_br %4, ^bb2(%2 : tensor<i32>), ^bb3(%2 : tensor<i32>)
^bb2(%5: tensor<i32>):
  %6 = stablehlo.add %5, %5 : tensor<i32>
  cf.br ^bb1(%6 : tensor<i32>)
^bb3(%7: tensor<i32>):
  check.expect_eq_const(%7, dense<4> : tensor<i32>) : tensor<i32>
  return
}
