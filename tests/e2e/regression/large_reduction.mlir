func.func @reduction_aligned() {
  %in = util.unfoldable_constant dense<1.0> : tensor<128x384xf32>
  %cst = arith.constant 0.0 : f32
  %init = linalg.init_tensor [128] : tensor<128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<128x384xf32>) outs(%fill : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<128xf32>
  check.expect_eq_const(%result, dense<384.0> : tensor<128xf32>) : tensor<128xf32>
  return
}

func.func @reduction_unaligned() {
  %in = util.unfoldable_constant dense<1.0> : tensor<129x384xf32>
  %cst = arith.constant 0.0 : f32
  %init = linalg.init_tensor [129] : tensor<129xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<129xf32>) -> tensor<129xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<129x384xf32>) outs(%fill : tensor<129xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<129xf32>
  check.expect_eq_const(%result, dense<384.0> : tensor<129xf32>) : tensor<129xf32>
  return
}
