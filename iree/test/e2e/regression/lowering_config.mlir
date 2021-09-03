#config1 = {tileSizes = [[32, 32, 32]], passPipeline = 1 : i32}
#config2 = {tileSizes = [[64, 64, 64]], passPipeline = 1 : i32}
func @lowering_config_test() {
  %a = util.unfoldable_constant dense<1.0> : tensor<128x256xf32>
  %b = util.unfoldable_constant dense<2.0> : tensor<256x512xf32>
  %c = util.unfoldable_constant dense<2.0> : tensor<256x1024xf32>
  %d = "mhlo.dot"(%a, %b) {lowering.config = #config1} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %e = "mhlo.dot"(%a, %c) {lowering.config = #config2} : (tensor<128x256xf32>, tensor<256x1024xf32>) -> tensor<128x1024xf32>
  check.expect_almost_eq_const(%d, dense<512.0> : tensor<128x512xf32>) : tensor<128x512xf32>
  check.expect_almost_eq_const(%e, dense<512.0> : tensor<128x1024xf32>) : tensor<128x1024xf32>
  return
}
