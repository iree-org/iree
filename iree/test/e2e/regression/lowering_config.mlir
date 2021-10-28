#compilation0 = #iree_codegen.compilation.info<
    #iree_codegen.lowering.config<tile_sizes = [[32, 32, 32]], native_vector_size = []>,
    #iree_codegen.translation.info<"CPUVectorization", workload_per_wg = [32, 32]>,
    workgroup_size = []>
#compilation1 = #iree_codegen.compilation.info<
    #iree_codegen.lowering.config<tile_sizes = [[64, 64, 64]], native_vector_size = []>,
    #iree_codegen.translation.info<"CPUVectorization", workload_per_wg = [64, 64]>,
    workgroup_size = []>
func @lowering_config_test() {
  %a = util.unfoldable_constant dense<1.0> : tensor<128x256xf32>
  %b = util.unfoldable_constant dense<2.0> : tensor<256x512xf32>
  %c = util.unfoldable_constant dense<2.0> : tensor<256x1024xf32>
  %d = "mhlo.dot"(%a, %b) {compilation.info = #compilation0} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %e = "mhlo.dot"(%a, %c) {compilation.info = #compilation1} : (tensor<128x256xf32>, tensor<256x1024xf32>) -> tensor<128x1024xf32>
  check.expect_almost_eq_const(%d, dense<512.0> : tensor<128x512xf32>) : tensor<128x512xf32>
  check.expect_almost_eq_const(%e, dense<512.0> : tensor<128x1024xf32>) : tensor<128x1024xf32>
  return
}
