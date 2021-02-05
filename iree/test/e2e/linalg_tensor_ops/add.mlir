func @tensor() attributes { iree.module.export } {
  %A = iree.unfoldable_constant dense<
    [[1.0, 2.0, 3.0, 4.0, 5.0],
     [6.0, 7.0, 8.0, 9.0, 10.0],
     [11.0, 12.0, 13.0, 14.0, 15.0],
     [16.0, 17.0, 18.0, 19.0, 20.0],
     [21.0, 22.0, 23.0, 24.0, 25.0],
     [26.0, 27.0, 28.0, 29.0, 30.0],
     [31.0, 32.0, 33.0, 34.0, 35.0],
     [36.0, 37.0, 38.0, 39.0, 40.0]]> : tensor<8x5xf32>
  %B = iree.unfoldable_constant dense<
     [11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<5xf32>
  %C = linalg.init_tensor [8, 5] : tensor<8x5xf32>
  %D = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%A, %B: tensor<8x5xf32>, tensor<5xf32>)
    outs(%C: tensor<8x5xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2 : f32):
      %0 = addf %arg0, %arg1 : f32
      linalg.yield %0 : f32
    } -> tensor<8x5xf32>
  check.expect_almost_eq_const(%D, dense<
    [[12.0, 14.0, 16.0, 18.0, 20.0],
     [17.0, 19.0, 21.0, 23.0, 25.0],
     [22.0, 24.0, 26.0, 28.0, 30.0],
     [27.0, 29.0, 31.0, 33.0, 35.0],
     [32.0, 34.0, 36.0, 38.0, 40.0],
     [37.0, 39.0, 41.0, 43.0, 45.0],
     [42.0, 44.0, 46.0, 48.0, 50.0],
     [47.0, 49.0, 51.0, 53.0, 55.0]]> : tensor<8x5xf32>) : tensor<8x5xf32>
  return
}

func @tensor_large() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<20.0> : tensor<250x500xf32>
  %rhs = iree.unfoldable_constant dense<22.0> : tensor<500xf32>
  %init = linalg.init_tensor [500, 250] : tensor<500x250xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%lhs, %rhs: tensor<250x500xf32>, tensor<500xf32>)
    outs(%init: tensor<500x250xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2 : f32):
      %0 = addf %arg0, %arg1 : f32
      linalg.yield %0 : f32
    } -> tensor<500x250xf32>
  check.expect_almost_eq_const(%result, dense<42.0> : tensor<500x250xf32>) : tensor<500x250xf32>
  return
}