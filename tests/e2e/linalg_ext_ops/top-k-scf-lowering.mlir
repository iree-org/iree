func.func @vector_topk_test_32() {
  %input_values = util.unfoldable_constant dense<[[1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.4, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.8, 1.0, 2.0, 3.0, 4.23, 1.0, 2.0, 3.0, 4.2, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]> : tensor<1x32xf32>
  %out_values_empty = tensor.empty() : tensor<1x32xf32>
  %out_indices_empty = tensor.empty() : tensor<1x32xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<1x32xf32>) -> tensor<1x32xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<1x32xi32>) -> tensor<1x32xi32>
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values: tensor<1x32xf32>)
        outs(%out_values, %out_indices : tensor<1x32xf32>, tensor<1x32xi32>) {
        ^bb0(%arg0 : f32, %arg1 : f32):
         %0 = arith.cmpf ogt, %arg0, %arg1 : f32
         iree_linalg_ext.yield %0 : i1
        } -> tensor<1x32xf32>, tensor<1x32xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[[4.8, 4.23, 4.2, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]> : tensor<1x32xf32>
  ) : tensor<1x32xf32>

  check.expect_eq_const(
      %0#1,
      dense<[[15, 19, 23, 3, 7, 11, 27, 31, 2, 6, 10, 14, 18, 22, 26, 30, 1, 5, 9, 13, 17, 21, 25, 29, 8, 0, 4, 12, 16, 20, 24, 28]]> : tensor<1x32xi32>
  ) : tensor<1x32xi32>

  return
}

func.func @vector_topk_test_35() {
  %input_values = util.unfoldable_constant dense<[[1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.4, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.8, 1.0, 2.0, 3.0, 4.23, 1.0, 2.0, 3.0, 4.2, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 8.0, 8.1, 8.2]]> : tensor<1x35xf32>
  %out_values_empty = tensor.empty() : tensor<1x35xf32>
  %out_indices_empty = tensor.empty() : tensor<1x35xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<1x35xf32>) -> tensor<1x35xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<1x35xi32>) -> tensor<1x35xi32>
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values: tensor<1x35xf32>)
        outs(%out_values, %out_indices : tensor<1x35xf32>, tensor<1x35xi32>) {
        ^bb0(%arg0 : f32, %arg1 : f32):
         %0 = arith.cmpf ogt, %arg0, %arg1 : f32
         iree_linalg_ext.yield %0 : i1
        } -> tensor<1x35xf32>, tensor<1x35xi32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[[8.2, 8.1, 8.0, 4.8, 4.23, 4.2, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]> : tensor<1x35xf32>
  ) : tensor<1x35xf32>

  check.expect_eq_const(
      %0#1,
      dense<[[34, 33, 32, 15, 19, 23, 3, 7, 11, 27, 31, 2, 6, 10, 14, 18, 22, 26, 30, 1, 5, 9, 13, 17, 21, 25, 29, 8, 0, 4, 12, 16, 20, 24, 28]]> : tensor<1x35xi32>
  ) : tensor<1x35xi32>

  return
}

// int32 type not supported in the scalar implementation.
//func.func @vector_topk_test_32_int() {
//  %input_values = util.unfoldable_constant dense<[[1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]> : tensor<1x32xi32>
//  %out_values_empty = tensor.empty() : tensor<1x32xi32>
//  %out_indices_empty = tensor.empty() : tensor<1x32xi32>
//  %neg_1 = arith.constant 0xFFFFFFFF : i32
//  %c0 = arith.constant 0 : i32
//  %out_values = linalg.fill ins(%neg_1 : i32) outs(%out_values_empty : tensor<1x32xi32>) -> tensor<1x32xi32>
//  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<1x32xi32>) -> tensor<1x32xi32>
//  %0:2 = iree_linalg_ext.topk
//        dimension(1)
//        ins(%input_values: tensor<1x32xi32>)
//        outs(%out_values, %out_indices : tensor<1x32xi32>, tensor<1x32xi32>) {
//        ^bb0(%arg0 : i32, %arg1 : i32):
//         %0 = arith.cmpi eq, %arg0, %arg1 : i32
//         iree_linalg_ext.yield %0 : i1
//        } -> tensor<1x32xi32>, tensor<1x32xi32>
//
//  check.expect_eq_const(
//      %0#0,
//      dense<[[7, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]]> : tensor<1x32xi32>
//  ) : tensor<1x32xi32>
//
//  check.expect_eq_const(
//      %0#1,
//      dense<[[3, 19, 7, 11, 15, 23, 27, 31, 2, 6, 10, 14, 18, 22, 26, 30, 1, 5, 9, 13, 17, 21, 25, 29, 0, 4, 8, 12, 16, 20, 24, 28]]> : tensor<1x32xi32>
//  ) : tensor<1x32xi32>
//
//  return
//}

//func.func @vector_topk_test_4x64_int() {
//  %input_values = util.unfoldable_constant dense<[[
//    1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 18, 2, 3, 4],[
//    1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 22, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],[
//    1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 46, 2, 3, 4,
//    1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],[
//    1, 2, 323, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 7, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
//    1, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]> : tensor<4x64xi32>
//  %out_values_empty = tensor.empty() : tensor<4x40xi32>
//  %out_indices_empty = tensor.empty() : tensor<4x40xi32>
//  %neg_1 = arith.constant 0xFFFFFFFF : i32
//  %c0 = arith.constant 0 : i32
//  %out_values = linalg.fill ins(%neg_1 : i32) outs(%out_values_empty : tensor<4x40xi32>) -> tensor<4x40xi32>
//  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<4x40xi32>) -> tensor<4x40xi32>
//  %0:2 = iree_linalg_ext.topk
//        dimension(1)
//        ins(%input_values: tensor<4x64xi32>)
//        outs(%out_values, %out_indices : tensor<4x40xi32>, tensor<4x40xi32>) {
//        ^bb0(%arg0 : i32, %arg1 : i32):
//         %0 = arith.cmpi eq, %arg0, %arg1 : i32
//         iree_linalg_ext.yield %0 : i1
//        } -> tensor<4x40xi32>, tensor<4x40xi32>
//
//  check.expect_eq_const(
//      %0#0,
//      dense<[[18, 7, 7, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2],
//             [22, 7, 7, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
//             [46, 7, 7, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2],
//             [323, 7, 7, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]]>:tensor<4x40xi32>
//  ) : tensor<4x40xi32>
//
//  check.expect_eq_const(
//      %0#1,
//      dense<[[60, 3, 35, 19, 51, 7, 11, 15, 23, 27, 31, 39, 43, 47, 55, 59, 63, 2, 6, 10, 14, 18,
//              22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 1, 5, 9, 13, 17, 21, 25],
//             [23, 3, 35, 19, 51, 7, 11, 15, 27, 31, 39, 43, 47, 55, 59, 63, 2, 6, 10, 14, 18, 22,
//              26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 1, 5, 9, 13, 17, 21, 25, 29],
//             [28, 3, 35, 19, 51, 7, 11, 15, 23, 27, 31, 39, 43, 47, 55, 59, 63, 2, 6, 10, 14, 18,
//              22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 1, 5, 9, 13, 17, 21, 25],
//             [2, 3, 35, 19, 51, 7, 11, 15, 23, 27, 31, 39, 43, 47, 55, 59, 63, 6, 10, 14, 18, 22,
//              26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 1, 5, 9, 13, 17, 21, 25, 29]]> : tensor<4x40xi32>
//  ) : tensor<4x40xi32>
//
//  return
//}

func.func @vector_call_topk_1x256() {
  %input_values = util.unfoldable_constant dense<[[
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 123.45, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 8.9, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.3, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 4.28, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 7.8, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]> : tensor<1x256xf32>
  %out_values_empty = tensor.empty() : tensor<1x40xf32>
  %out_indices_empty = tensor.empty() : tensor<1x40xi32>
  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_values = linalg.fill ins(%neg_inf : f32) outs(%out_values_empty : tensor<1x40xf32>) -> tensor<1x40xf32>
  %out_indices = linalg.fill ins(%c0 : i32) outs(%out_indices_empty : tensor<1x40xi32>) -> tensor<1x40xi32>
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values: tensor<1x256xf32>)
        outs(%out_values, %out_indices : tensor<1x40xf32>, tensor<1x40xi32>) {
        ^bb0(%arg0 : f32, %arg1 : f32):
         %0 = arith.cmpf ogt, %arg0, %arg1 : f32
         iree_linalg_ext.yield %0 : i1
        } -> tensor<1x40xf32>, tensor<1x40xi32>
 check.expect_almost_eq_const(
     %0#0,
     dense<[[123.45, 8.9, 7.8, 4.28, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]]> : tensor<1x40xf32>
 ) : tensor<1x40xf32>

  check.expect_eq_const(
      %0#1,
      dense<[[135, 169, 247, 233, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 139, 143, 147]]> : tensor<1x40xi32>
  ) : tensor<1x40xi32>
 return
}
