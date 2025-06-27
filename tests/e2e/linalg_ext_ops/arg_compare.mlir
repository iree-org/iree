func.func @arg_compare_1d_dim0_max() {
  %input_values = util.unfoldable_constant dense<[1.0, 5.0, 2.0, 10.0, 7.0]> : tensor<5xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<5xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<10.0> : tensor<f32>
  ) : tensor<f32>

  check.expect_eq_const(
    %0#1,
    dense<3> : tensor<i32>
  ) : tensor<i32>

  return
}

func.func @arg_compare_1d_dim0_max_with_base() {
  %input_values = util.unfoldable_constant dense<[1.0, 5.0, 2.0, 10.0, 7.0]> : tensor<5xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %index_base = arith.constant 100 : index

  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<5xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>)  index_base(%index_base : index) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<10.0> : tensor<f32>
  ) : tensor<f32>

  check.expect_eq_const(
    %0#1,
    dense<103> : tensor<i32>
  ) : tensor<i32>

  return
}

func.func @arg_compare_1d_dim0_min() {
  %input_values = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %pos_inf = arith.constant 0x7F800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%pos_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<10xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf olt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<1.0> : tensor<f32>
  ) : tensor<f32>

  check.expect_eq_const(
    %0#1,
    dense<0> : tensor<i32>
  ) : tensor<i32>

  return
}

func.func @arg_compare_2d_dim1_max() {
  %input_values = util.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]> : tensor<2x6xf32>

  %out_value_empty = tensor.empty() : tensor<2xf32>
  %out_index_empty = tensor.empty() : tensor<2xi32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<2xf32>) -> tensor<2xf32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<2xi32>) -> tensor<2xi32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values : tensor<2x6xf32>)
    outs(%out_value, %out_index : tensor<2xf32>, tensor<2xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<2xf32>, tensor<2xi32>

  check.expect_almost_eq_const(
    %0#0,
    dense<[6.0, 12.0]> : tensor<2xf32>
  ) : tensor<2xf32>

  check.expect_eq_const(
    %0#1,
    dense<[5, 5]> : tensor<2xi32>
  ) : tensor<2xi32>

  return
}

func.func @arg_compare_2d_dim1_min() {
  %input_values = util.unfoldable_constant dense<[[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]> : tensor<2x6xf32>

  %out_value_empty = tensor.empty() : tensor<2xf32>
  %out_index_empty = tensor.empty() : tensor<2xi32>

  %pos_inf = arith.constant 0x7F800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%pos_inf : f32) outs(%out_value_empty : tensor<2xf32>) -> tensor<2xf32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<2xi32>) -> tensor<2xi32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values : tensor<2x6xf32>)
    outs(%out_value, %out_index : tensor<2xf32>, tensor<2xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf olt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<2xf32>, tensor<2xi32>

  check.expect_almost_eq_const(
    %0#0,
    dense<[1.0, 7.0]> : tensor<2xf32>
  ) : tensor<2xf32>

  check.expect_eq_const(
    %0#1,
    dense<[5, 0]> : tensor<2xi32>
  ) : tensor<2xi32>

  return
}

func.func @arg_compare_1d_repeat_max() {
  %input_values = util.unfoldable_constant dense<[1.0, 1.5, 3.0, 5.0, 5.0, 10.0, 5.0, 2.0, 2.0, 10.0]> : tensor<10xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<10xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<10.0> : tensor<f32>
  ) : tensor<f32>

  check.expect_eq_const(
    %0#1,
    dense<5> : tensor<i32>
  ) : tensor<i32>

  return
}


func.func @arg_compare_1d_max_double() {
  %input_values = util.unfoldable_constant dense<
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
  > : tensor<18xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<18xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<18.0> : tensor<f32>
  ) : tensor<f32>

  check.expect_eq_const(
    %0#1,
    dense<17> : tensor<i32>
  ) : tensor<i32>

  return
}
