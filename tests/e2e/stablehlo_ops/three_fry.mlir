func.func @three_fry_i32() {
  %inp = util.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %0:2 = "stablehlo.rng_bit_generator"(%inp) {rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>} : (tensor<4xi32>) -> (tensor<4xi32>, tensor<8xi32>)
  check.expect_eq_const(%0#1, dense<[-1997982863, -261361928, -1008514867, 1226850200, 1419974734, -277475325, 1033030661, -1926332264]> : tensor<8xi32>) : tensor<8xi32>
  return
}
