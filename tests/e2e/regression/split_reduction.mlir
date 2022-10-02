#map0 = affine_map<(d1,d2) -> (d1,d2)>
#map1 = affine_map<(d1,d2) -> (d1)>

func.func @split_reduction_pass1() {
  %input = util.unfoldable_constant dense<1> : tensor<64x32xi32>
  %cst_0 = arith.constant 0 : i32
  %init = linalg.init_tensor [64] : tensor<64xi32>
  %zeros = linalg.fill ins(%cst_0 : i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
  %result = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%input : tensor<64x32xi32>) outs(%init : tensor<64xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):
    %add = arith.addi %arg1, %arg2 : i32
    linalg.yield %add : i32
  } -> tensor<64xi32>
  check.expect_eq_const(%result, dense<32> : tensor<64xi32>) : tensor<64xi32>
  return
}

// -----

#map2 = affine_map<(d1,d2,d3) -> (d1,d2,d3)>
#map3 = affine_map<(d1,d2,d3) -> (d1,d2)>

func.func @split_reduction_pass2() {
  %input = util.unfoldable_constant dense<1> : tensor<512x256x128xi32>
  %cst_0 = arith.constant 0 : i32
  %init = linalg.init_tensor [512,256] : tensor<512x256xi32>
  %zeros = linalg.fill ins(%cst_0 : i32) outs(%init : tensor<512x256xi32>) -> tensor<512x256xi32>
  %result = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%input : tensor<512x256x128xi32>) outs(%init : tensor<512x256xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):
    %add = arith.addi %arg1, %arg2 : i32
    linalg.yield %add : i32
  } -> tensor<512x256xi32>
  check.expect_eq_const(%result, dense<128> : tensor<512x256xi32>) : tensor<512x256xi32>
  return
}

// -----

#map4 = affine_map<(d1,d2,d3) -> (d1,d2,d3)>
#map5 = affine_map<(d1,d2,d3) -> (d1,d3)>

func.func @split_reduction_fail1() {
  %input = util.unfoldable_constant dense<1> : tensor<128x32x64xi32>
  %cst_0 = arith.constant 0 : i32
  %init = linalg.init_tensor [128,64] : tensor<128x64xi32>
  %zeros = linalg.fill ins(%cst_0 : i32) outs(%init : tensor<128x64xi32>) -> tensor<128x64xi32>
  %result = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%input : tensor<128x32x64xi32>) outs(%init : tensor<128x64xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):
    %add = arith.addi %arg1, %arg2 : i32
    linalg.yield %add : i32
  } -> tensor<128x64xi32>
  check.expect_eq_const(%result, dense<32> : tensor<128x64xi32>) : tensor<128x64xi32>
  return
}

// -----

#map6 = affine_map<(d1,d2,d3) -> (d1,d2,d3)>
#map7 = affine_map<(d1,d2,d3) -> (d1)>

func.func @split_reduction_fail2() {
  %input = util.unfoldable_constant dense<1> : tensor<128x32x64xi32>
  %cst_0 = arith.constant 0 : i32
  %init = linalg.init_tensor [128] : tensor<128xi32>
  %zeros = linalg.fill ins(%cst_0 : i32) outs(%init : tensor<128xi32>) -> tensor<128xi32>
  %result = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "reduction", "reduction"]} ins(%input : tensor<128x32x64xi32>) outs(%init : tensor<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):
    %add = arith.addi %arg1, %arg2 : i32
    linalg.yield %add : i32
  } -> tensor<128xi32>
  check.expect_eq_const(%result, dense<2048> : tensor<128xi32>) : tensor<128xi32>
  return
}
