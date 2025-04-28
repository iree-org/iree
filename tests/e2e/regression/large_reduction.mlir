func.func @reduction_aligned() {
  %in = util.unfoldable_constant dense<1.0> : tensor<128x384xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<128x384xf32>) outs(%fill : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<128xf32>
  check.expect_eq_const(%result, dense<384.0> : tensor<128xf32>) : tensor<128xf32>
  return
}

func.func @reduction_unaligned() {
  %in = util.unfoldable_constant dense<1.0> : tensor<129x384xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<129xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<129xf32>) -> tensor<129xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<129x384xf32>) outs(%fill : tensor<129xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<129xf32>
  check.expect_eq_const(%result, dense<384.0> : tensor<129xf32>) : tensor<129xf32>
  return
}

// Reduction dimension larger than the max number of threads per group on a gpu.
func.func @reduction_aligned_larger() {
  %in = util.unfoldable_constant dense<0.001> : tensor<2x40960xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2xf32>) -> tensor<2xf32>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<2x40960xf32>) outs(%fill : tensor<2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<2xf32>
  check.expect_almost_eq_const(%result, dense<40.96> : tensor<2xf32>) : tensor<2xf32>
  return
}

func.func @half_reduction_aligned() {
  %in = util.unfoldable_constant dense<0.001> : tensor<2x4096xf16>
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<2xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<2xf16>) -> tensor<2xf16>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<2x4096xf16>) outs(%fill : tensor<2xf16>) {
    ^bb0(%arg3: f16, %arg4: f16):
      %2 = arith.addf %arg3, %arg4 : f16
      linalg.yield %2 : f16
    } -> tensor<2xf16>
  check.expect_almost_eq_const(%result, dense<4.096> : tensor<2xf16>) : tensor<2xf16>
  return
}

func.func @quarter_reduction_aligned_smaller() {
  %in = util.unfoldable_constant dense<1> : tensor<128x128xi8>
  %cst = arith.constant 0 : i8
  %init = tensor.empty() : tensor<128xi8>
  %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<128xi8>) -> tensor<128xi8>
  %result = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<128x128xi8>) outs(%fill : tensor<128xi8>) {
    ^bb0(%arg3: i8, %arg4: i8):
      %2 = arith.addi %arg3, %arg4 : i8
      linalg.yield %2 : i8
    } -> tensor<128xi8>
  check.expect_eq_const(%result, dense<128> : tensor<128xi8>) : tensor<128xi8>
  return
}
