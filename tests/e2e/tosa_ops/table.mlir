func.func @table() {
  %input = arith.constant dense<[-5405, 15214, -14896, 22008, 12529, -13501]> : tensor<6xi16>

  // This generates [0, ... 512] for a constant value to avoid an excessively large constant.
  %init = tensor.empty() : tensor<513xi16>
  %cst = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    outs(%init: tensor<513xi16>) {
    ^bb0(%arg1: i16):
      %i = linalg.index 0 : index
      %0 = arith.index_cast %i : index to i16
      linalg.yield %0 : i16
    } -> tensor<513xi16>

  %result = tosa.table %input, %cst : (tensor<6xi16>, tensor<513xi16>) -> tensor<6xi32>
  check.expect_eq_const(%result, dense<[27363, 47982, 17872, 54776, 45297, 19267]> : tensor<6xi32>) : tensor<6xi32>
  return
}
