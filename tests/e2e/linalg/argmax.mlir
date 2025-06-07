func.func @argmax_1d() {
  %seven = arith.constant 7.0 : f32
  %input_init = tensor.empty() : tensor<131072xf32>
  %input_filled = linalg.fill ins(%seven : f32) outs(%input_init : tensor<131072xf32>) -> tensor<131072xf32>

  %large = arith.constant 53.0 : f32
  %index = arith.constant 131071 : index
  %input = tensor.insert %large into %input_filled[%index] : tensor<131072xf32>

  %neg_inf = arith.constant 0xFF800000 : f32  // -inf
  %c0_i32 = arith.constant 0 : i32
  %init_val_buf = tensor.empty() : tensor<f32>
  %init_idx_buf = tensor.empty() : tensor<i32>
  %init_val = linalg.fill ins(%neg_inf : f32) outs(%init_val_buf : tensor<f32>) -> tensor<f32>
  %init_idx = linalg.fill ins(%c0_i32 : i32) outs(%init_idx_buf : tensor<i32>) -> tensor<i32>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%input : tensor<131072xf32>)
      outs(%init_val, %init_idx : tensor<f32>, tensor<i32>) {
    ^bb0(%in: f32, %val: f32, %idx: i32):
      %i = linalg.index 0 : index
      %i_cast = arith.index_cast %i : index to i32
      %maxval = arith.maximumf %in, %val : f32
      %cmp = arith.cmpf ogt, %in, %val : f32
      %sel_idx = arith.select %cmp, %i_cast, %idx : i32
      linalg.yield %maxval, %sel_idx : f32, i32
  } -> (tensor<f32>, tensor<i32>)

  check.expect_almost_eq_const(%result#0, dense<53.0> : tensor<f32>) : tensor<f32>
  check.expect_eq_const(%result#1, dense<131071> : tensor<i32>) : tensor<i32>

  return
}
