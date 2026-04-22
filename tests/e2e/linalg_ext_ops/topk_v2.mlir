func.func @topk_1d_dim0_max() {
  %input_values = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>
  %input_indices = util.unfoldable_constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %out_indices_empty = tensor.empty() : tensor<3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<3xi32>) -> tensor<3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values, %input_indices : tensor<10xf32>, tensor<10xi32>)
      outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[10.0, 9.0, 8.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[9, 8, 7]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

func.func @topk_1d_dim0_max_no_input_indices() {
  %input_values = util.unfoldable_constant dense<[4.0, 5.0, 8.0, 1.0, 2.0, 10.0, 7.0, 3.0, 9.0, 6.0]> : tensor<10xf32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %out_indices_empty = tensor.empty() : tensor<3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<3xi32>) -> tensor<3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values : tensor<10xf32>)
      outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[10.0, 9.0, 8.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[5, 8, 2]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

func.func @topk_1d_dim0_min() {
  %input_values = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %out_indices_empty = tensor.empty() : tensor<3xi32>
  %pos_inf = arith.constant 0x7F800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%pos_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<3xi32>) -> tensor<3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values : tensor<10xf32>)
      outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf olt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[0, 1, 2]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

func.func @topk_2d_dim1_max() {
  %input_values = util.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]> : tensor<2x6xf32>

  %out_values_empty = tensor.empty() : tensor<2x3xf32>
  %out_indices_empty = tensor.empty() : tensor<2x3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<2x3xf32>) -> tensor<2x3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<2x3xi32>) -> tensor<2x3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(1) is_sorted
      ins(%input_values : tensor<2x6xf32>)
      outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<2x3xf32>, tensor<2x3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[[6.0, 5.0, 4.0], [12.0, 11.0, 10.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[[5, 4, 3], [5, 4, 3]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  return
}

func.func @topk_1d_repeat_max() {
  %input_values = util.unfoldable_constant dense<[1.0, 1.5, 3.0, 5.0, 5.0, 3.0, 5.0, 2.0, 2.0, 10.0]> : tensor<10xf32>

  %out_values_empty = tensor.empty() : tensor<5xf32>
  %out_indices_empty = tensor.empty() : tensor<5xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<5xf32>) -> tensor<5xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<5xi32>) -> tensor<5xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values : tensor<10xf32>)
      outs(%out_values, %out_indices : tensor<5xf32>, tensor<5xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<5xf32>, tensor<5xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[10.0, 5.0, 5.0, 5.0, 3.0]> : tensor<5xf32>
  ) : tensor<5xf32>

  check.expect_eq_const(
      %0#1,
      dense<[9, 3, 6, 4, 2]> : tensor<5xi32>
  ) : tensor<5xi32>

  return
}

func.func @topk_1d_dim0_max_double() {
  %input_values = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]> : tensor<18xf32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %out_indices_empty = tensor.empty() : tensor<3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<3xi32>) -> tensor<3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values : tensor<18xf32>)
      outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[18.0, 17.0, 16.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[17, 16, 15]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

func.func @unsorted() {
  %input_values = util.unfoldable_constant dense<[4.0, 5.0, 8.0, 1.0, 2.0, 10.0, 7.0, 3.0, 9.0, 6.0]> : tensor<10xf32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %out_indices_empty = tensor.empty() : tensor<3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<3xi32>) -> tensor<3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0)
      ins(%input_values : tensor<10xf32>)
      outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%0#0, %0#1 : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : i32, %arg3 : i32):
        %1 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %1 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  check.expect_almost_eq_const(
      %sorted#0,
      dense<[10.0, 9.0, 8.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  check.expect_eq_const(
      %sorted#1,
      dense<[5, 8, 2]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

func.func @values_only() {
  %input_values = util.unfoldable_constant dense<[4.0, 5.0, 8.0, 1.0, 2.0, 10.0, 7.0, 3.0, 9.0, 6.0]> : tensor<10xf32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %0 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values : tensor<10xf32>)
      outs(%out_values : tensor<3xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>

  check.expect_almost_eq_const(
      %0,
      dense<[10.0, 9.0, 8.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  return
}

func.func @topk_1d_dynamic_max() {
  %input_values = flow.tensor.dynamic_constant
      dense<[4.0, 5.0, 8.0, 1.0, 2.0, 10.0, 7.0, 3.0, 9.0, 6.0]>
      : tensor<10xf32> -> tensor<?xf32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %out_indices_empty = tensor.empty() : tensor<3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<3xi32>) -> tensor<3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values : tensor<?xf32>)
      outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[10.0, 9.0, 8.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[5, 8, 2]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

func.func @topk_2d_dynamic_dim1_max() {
  %input_values = flow.tensor.dynamic_constant
      dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]>
      : tensor<2x6xf32> -> tensor<2x?xf32>

  %out_values_empty = tensor.empty() : tensor<2x3xf32>
  %out_indices_empty = tensor.empty() : tensor<2x3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<2x3xf32>) -> tensor<2x3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<2x3xi32>) -> tensor<2x3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(1) is_sorted
      ins(%input_values : tensor<2x?xf32>)
      outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<2x3xf32>, tensor<2x3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[[6.0, 5.0, 4.0], [12.0, 11.0, 10.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[[5, 4, 3], [5, 4, 3]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  return
}

func.func @topk_1d_dynamic_k() {
  %input_values = util.unfoldable_constant dense<[4.0, 5.0, 8.0, 1.0, 2.0, 10.0, 7.0, 3.0, 9.0, 6.0]> : tensor<10xf32>

  %out_values_init = flow.tensor.dynamic_constant
      dense<[0xFF800000, 0xFF800000, 0xFF800000]>
      : tensor<3xf32> -> tensor<?xf32>
  %out_indices_init = flow.tensor.dynamic_constant
      dense<[0, 0, 0]> : tensor<3xi32> -> tensor<?xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values : tensor<10xf32>)
      outs(%out_values_init, %out_indices_init : tensor<?xf32>, tensor<?xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<?xf32>, tensor<?xi32>

  %expected_values = flow.tensor.dynamic_constant
      dense<[10.0, 9.0, 8.0]> : tensor<3xf32> -> tensor<?xf32>
  %expected_indices = flow.tensor.dynamic_constant
      dense<[5, 8, 2]> : tensor<3xi32> -> tensor<?xi32>

  check.expect_almost_eq(%0#0, %expected_values) : tensor<?xf32>
  check.expect_eq(%0#1, %expected_indices) : tensor<?xi32>

  return
}

func.func @with_input_indices() {
  %input_values = util.unfoldable_constant dense<[4.0, 5.0, 8.0, 1.0, 2.0, 10.0, 7.0, 3.0, 9.0, 6.0]> : tensor<10xf32>
  %input_indices = util.unfoldable_constant dense<[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]> : tensor<10xi32>

  %out_values_empty = tensor.empty() : tensor<3xf32>
  %out_indices_empty = tensor.empty() : tensor<3xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<3xf32>) -> tensor<3xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<3xi32>) -> tensor<3xi32>
  %0:2 = iree_linalg_ext.topk_v2 dimension(0) is_sorted
      ins(%input_values, %input_indices : tensor<10xf32>, tensor<10xi32>)
      outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.cmpf ogt, %arg0, %arg1 : f32
        iree_linalg_ext.yield %0 : i1
      } -> tensor<3xf32>, tensor<3xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[10.0, 9.0, 8.0]> : tensor<3xf32>
  ) : tensor<3xf32>

  check.expect_eq_const(
      %0#1,
      dense<[105, 108, 102]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}
