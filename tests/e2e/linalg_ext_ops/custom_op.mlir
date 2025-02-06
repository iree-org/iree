func.func private @init_tensor(%arg0 : index) -> tensor<?xf32> {
  %0 = tensor.empty(%arg0) : tensor<?xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%0 : tensor<?xf32>) {
    ^bb0(%b0 : f32) :
      %index = linalg.index 0 : index
      %index_i32 = arith.index_cast %index : index to i32
      %index_f32 = arith.uitofp %index_i32 : i32 to f32
      %arg0_i32 = arith.index_cast %arg0 : index to i32
      %arg0_f32 = arith.uitofp %arg0_i32 : i32 to f32
      %val = arith.divf %index_f32, %arg0_f32 : f32
      linalg.yield %val : f32
  } -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

func.func @matmul_bias_add() {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %lhs_size = arith.muli %c128, %c256 : index
  %rhs_size = arith.muli %c256, %c512 : index
  %lhs_linear = func.call @init_tensor(%lhs_size): (index) -> (tensor<?xf32>)
  %lhs = tensor.expand_shape %lhs_linear [[0, 1]] output_shape [%c128, %c256]
      : tensor<?xf32> into tensor<?x?xf32>
  %lhs_static = tensor.cast %lhs : tensor<?x?xf32> to tensor<128x256xf32>
  %rhs_linear = func.call @init_tensor(%rhs_size) : (index) -> (tensor<?xf32>)
  %rhs = tensor.expand_shape %rhs_linear [[0, 1]] output_shape [%c256, %c512]
      : tensor<?xf32> into tensor<?x?xf32>
  %rhs_static = tensor.cast %rhs : tensor<?x?xf32> to tensor<256x512xf32>
  %bias = func.call @init_tensor(%c512) : (index) -> (tensor<?xf32>)
  %bias_static = tensor.cast %bias : tensor<?xf32> to tensor<512xf32>
  %lhs_input = util.optimization_barrier %lhs_static : tensor<128x256xf32>
  %rhs_input = util.optimization_barrier %rhs_static : tensor<256x512xf32>
  %bias_input = util.optimization_barrier %bias_static : tensor<512xf32>
  %empty = tensor.empty() : tensor<128x512xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  %matmul = linalg.matmul
      ins(%lhs_input, %rhs_input : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
  %bias_add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul, %bias_input : tensor<128x512xf32>, tensor<512xf32>)
      outs(%empty : tensor<128x512xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 :f32):
      %addf = arith.addf %b0, %b1 : f32
      linalg.yield %addf : f32
  } -> tensor<128x512xf32>

  %custom_op = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                       affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d1)>,
                       affine_map<(d0, d1)[s0] -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>]}
      ins(%lhs_input, %rhs_input, %bias_input : tensor<128x256xf32>, tensor<256x512xf32>, tensor<512xf32>)
      outs(%empty : tensor<128x512xf32>) {
    ^bb0(%t0 : tensor<?x?xf32>, %t1 : tensor<?x?xf32>, %t2 : tensor<?xf32>, %t3 : tensor<?x?xf32>):
      %2 = linalg.fill ins(%cst : f32) outs(%t3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %3 = linalg.matmul ins(%t0, %t1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %4 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3, %t2 : tensor<?x?xf32>, tensor<?xf32>)
          outs(%t3 : tensor<?x?xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %5 = arith.addf %b0, %b1 : f32
          linalg.yield %5 : f32
      } -> tensor<?x?xf32>
      iree_linalg_ext.yield %4 : tensor<?x?xf32>
  } -> tensor<128x512xf32>
  check.expect_eq(%bias_add, %custom_op) : tensor<128x512xf32>
  return
}
