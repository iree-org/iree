func.func @simple_large_reduction_tiling_static() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %normalize = arith.constant 131200.0 : f32
  %input_empty = tensor.empty() : tensor<128x131072xf32>
  %input = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%input_empty : tensor<128x131072xf32>) {
    ^bb0(%b0 : f32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = arith.addi %0, %1 : index
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.uitofp %3 : i32 to f32
      %5 = arith.divf %4, %normalize : f32
      linalg.yield %5 : f32
  } -> (tensor<128x131072xf32>)
  %input_opt_barrier = util.optimization_barrier %input : tensor<128x131072xf32>
  %empty = tensor.empty() : tensor<128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %normal_reduce = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input_opt_barrier : tensor<128x131072xf32>) outs(%fill : tensor<128xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<128xf32>
  %split_reduction = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"],
      iree_linalg_ext.split_reduction = [128]}
      ins(%input_opt_barrier : tensor<128x131072xf32>) outs(%fill : tensor<128xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<128xf32>
  check.expect_almost_eq (%normal_reduce, %split_reduction, atol 0.0, rtol 0.0001) : tensor<128xf32>
  return
}

func.func @simple_large_reduction_2d_tiling_static() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %normalize = arith.constant 131200.0 : f32
  %input_empty = tensor.empty() : tensor<128x2048x128xf32>
  %input = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      outs(%input_empty : tensor<128x2048x128xf32>) {
    ^bb0(%b0 : f32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %11 = linalg.index 2 : index
      %2 = arith.addi %0, %1 : index
      %21 = arith.addi %2, %11 : index
      %3 = arith.index_cast %21 : index to i32
      %4 = arith.uitofp %3 : i32 to f32
      %5 = arith.divf %4, %normalize : f32
      linalg.yield %5 : f32
  } -> (tensor<128x2048x128xf32>)
  %input_opt_barrier = util.optimization_barrier %input : tensor<128x2048x128xf32>
  %empty = tensor.empty() : tensor<128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %normal_reduce = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%input_opt_barrier : tensor<128x2048x128xf32>) outs(%fill : tensor<128xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<128xf32>
  %split_reduction = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction"],
      iree_linalg_ext.split_reduction = [128, 64]}
      ins(%input_opt_barrier : tensor<128x2048x128xf32>) outs(%fill : tensor<128xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<128xf32>
  check.expect_almost_eq (%normal_reduce, %split_reduction, atol 0.0, rtol 0.0001) : tensor<128xf32>
  return
}
