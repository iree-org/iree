func.func public @householder_test() -> () {
  %m = util.unfoldable_constant dense<[
    [0.19151945, 0.62210877, 0.43772774],
    [0.78535858, 0.77997581, 0.27259261],
    [0.27646426, 0.80187218, 0.95813935],
    [0.87593263, 0.35781727, 0.50099513]]> : tensor<4x3xf32>
  %t = util.unfoldable_constant dense<[0.68346294, 0.71270203]> : tensor<2xf32>
  %result = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%m, %t) : (tensor<4x3xf32>, tensor<2xf32>) -> tensor<4x3xf32>
  check.expect_almost_eq_const(%result, dense<[
    [ 0.31653708,  0.10644531,  0.32681817],
    [-0.53676350,  0.37089570, -0.31482652],
    [-0.18895307, -0.54206765,  0.63208690],
    [-0.59866750, -0.16177818,  0.08177957]]> : tensor<4x3xf32>): tensor<4x3xf32>
  return
}
