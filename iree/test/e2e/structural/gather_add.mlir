func @torch_select_index_0() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [[01, 02, 03, 04, 05]],
    [[06, 07, 08, 09, 10]],
    [[11, 12, 13, 14, 15]],
    [[16, 17, 18, 19, 20]],
    [[21, 22, 23, 24, 25]]]> : tensor<5x1x5xi32>
  %indices = iree.unfoldable_constant dense<[0, 2]> : tensor<2xi32>
  %workload = constant 10 : index
  %result = flow.dispatch.region[%workload: index](
      %arg0 = %input : tensor<5x1x5xi32>,
      %arg1 = %indices : tensor<2xi32>) -> tensor<2x1x5xi32> {
    %0 = "mhlo.torch_index_select"(%arg0, %arg1) {
      dim = 0 : i64,
      batch_dims = 0 : i64
    } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
    %1 = mhlo.add %0, %0 : tensor<2x1x5xi32>
    flow.return %1 : tensor<2x1x5xi32>
  }

  check.expect_eq_const(%result, dense<[[[02, 04, 06, 08, 10]],
                                        [[22, 24, 26, 28, 30]]]> : tensor<2x1x5xi32>) : tensor<2x1x5xi32>
  return
}
