func.func @philox_i32() {
  %inp = util.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32> 
  %0:2 = "stablehlo.rng_bit_generator"(%inp) {rng_algorithm = #stablehlo<rng_algorithm PHILOX>} : (tensor<4xi32>) -> (tensor<4xi32>, tensor<8xi32>)
  check.expect_eq_const(%0#1, dense<[-1788415499, 854201270, -855525523, 2043148971, 110723240, 146396481, -1258660138, -1968502964]> : tensor<8xi32>) : tensor<8xi32>
  return
}
