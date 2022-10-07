// This test compares the output of linalg.quantized_matmul to an equivalent
// computation where the matmul part itself is done by linalg.matmul and the
// zero-points are handled by additional linalg.generic ops.
//
// It is meant to be compiled with IREE's TOSA input pipeline, which includes
// the --iree-linalg-quantized-matmul-to-matmul pass, which performs a similar
// transformation of quantized_matmul, so that it acts as an end-to-end
// correctness test for it.
//
// Reference: Section 2.3 of https://arxiv.org/abs/1712.05877.

// Equivalent to linalg.quantized_matmul, but not using linalg.quantized_matmul
func.func private @quantized_matmul_as_matmul_3x4x5(%lhs : tensor<3x4xi8>, %rhs : tensor<4x5xi8>,  %lhs_zp : i32, %rhs_zp : i32) -> tensor<3x5xi32> {
  %c_0 = arith.constant 0 : i32
  %init_acc_uninitialized =  tensor.empty() : tensor<3x5xi32>
  %zero_acc = linalg.fill ins(%c_0 : i32) outs(%init_acc_uninitialized : tensor<3x5xi32>) -> tensor<3x5xi32>

  // compute the matmul itself, which would be the end result already in the case
  // where both zero-point values %lhs_zp and %rhs_zp are zero.
  %matmul_result = linalg.matmul ins(%lhs, %rhs : tensor<3x4xi8>, tensor<4x5xi8>) outs(%zero_acc : tensor<3x5xi32>) -> tensor<3x5xi32>

  %k_size = arith.constant 4 : i32  // = dim 1 of %lhs = dim 0 of %rhs

  // compute the sums along rows of %lhs.
  %lhs_i32 = arith.extsi %lhs : tensor<3x4xi8> to tensor<3x4xi32>
  %init_lhs_sums_uninitialized = tensor.empty() : tensor<3xi32>
  %zero_lhs_sums = linalg.fill ins(%c_0 : i32) outs(%init_lhs_sums_uninitialized : tensor<3xi32>) -> tensor<3xi32>
  %lhs_sums = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%lhs_i32 : tensor<3x4xi32>)
      outs(%zero_lhs_sums : tensor<3xi32>) {
      ^bb0(%arg0: i32, %arg1: i32) :
          %1 = arith.addi %arg0, %arg1 : i32
          linalg.yield %1 : i32
      } -> tensor<3xi32>

  // compute the sums along columns of %rhs.
  %rhs_i32 = arith.extsi %rhs : tensor<4x5xi8> to tensor<4x5xi32>
  %init_rhs_sums_uninitialized = tensor.empty() : tensor<5xi32>
  %zero_rhs_sums = linalg.fill ins(%c_0 : i32) outs(%init_rhs_sums_uninitialized : tensor<5xi32>) -> tensor<5xi32>
  %rhs_sums = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%rhs_i32 : tensor<4x5xi32>)
      outs(%zero_rhs_sums : tensor<5xi32>) {
      ^bb0(%arg0: i32, %arg1: i32) :
          %1 = arith.addi %arg0, %arg1 : i32
          linalg.yield %1 : i32
      } -> tensor<5xi32>

  // add all the terms together.
  %quantized_matmul_from_matmul_result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul_result, %lhs_sums, %rhs_sums, %lhs_zp, %rhs_zp, %k_size : tensor<3x5xi32>, tensor<3xi32>, tensor<5xi32>, i32, i32, i32)
      outs(%init_acc_uninitialized : tensor<3x5xi32>) {
      ^bb0(%matmul_result_val : i32, %lhs_sums_val: i32, %rhs_sums_val: i32, %lhs_zp_val: i32, %rhs_zp_val: i32, %k : i32, %acc_val: i32) :
          %linear_term_in_rhs_zp = arith.muli %lhs_sums_val, %rhs_zp_val : i32
          %linear_term_in_lhs_zp = arith.muli %rhs_sums_val, %lhs_zp_val : i32
          %linear_term = arith.addi %linear_term_in_rhs_zp, %linear_term_in_lhs_zp : i32
          %product_of_zp = arith.muli %lhs_zp_val, %rhs_zp_val : i32
          %quadratic_term = arith.muli %k, %product_of_zp : i32
          %corrected_for_linear_term = arith.subi %matmul_result_val, %linear_term : i32
          %corrected = arith.addi %corrected_for_linear_term, %quadratic_term : i32
          linalg.yield %corrected : i32
      } -> tensor<3x5xi32>
  return %quantized_matmul_from_matmul_result : tensor<3x5xi32>
}

// Equivalent to linalg.quantized_matmul, but not using linalg.quantized_matmul
func.func private @quantized_matmul_as_matmul_dynamic(%lhs : tensor<?x?xi8>, %rhs : tensor<?x?xi8>,  %lhs_zp : i32, %rhs_zp : i32) -> tensor<?x?xi32> {
  %c_0_index = arith.constant 0 : index
  %c_1_index = arith.constant 1 : index
  %m_size = tensor.dim %lhs, %c_0_index : tensor<?x?xi8>
  %k_size = tensor.dim %lhs, %c_1_index : tensor<?x?xi8>
  %n_size = tensor.dim %rhs, %c_1_index : tensor<?x?xi8>
  %k_size_i32 = arith.index_cast %k_size : index to i32

  %c_0 = arith.constant 0 : i32
  %init_acc_uninitialized =  tensor.empty(%m_size, %n_size) : tensor<?x?xi32>
  %zero_acc = linalg.fill ins(%c_0 : i32) outs(%init_acc_uninitialized : tensor<?x?xi32>) -> tensor<?x?xi32>

  // compute the matmul itself, which would be the end result already in the case
  // where both zero-point values %lhs_zp and %rhs_zp are zero.
  %matmul_result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xi8>, tensor<?x?xi8>) outs(%zero_acc : tensor<?x?xi32>) -> tensor<?x?xi32>

  // compute the sums along rows of %lhs.
  %lhs_i32 = arith.extsi %lhs : tensor<?x?xi8> to tensor<?x?xi32>
  %init_lhs_sums_uninitialized = tensor.empty(%m_size) : tensor<?xi32>
  %zero_lhs_sums = linalg.fill ins(%c_0 : i32) outs(%init_lhs_sums_uninitialized : tensor<?xi32>) -> tensor<?xi32>
  %lhs_sums = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%lhs_i32 : tensor<?x?xi32>)
      outs(%zero_lhs_sums : tensor<?xi32>) {
      ^bb0(%arg0: i32, %arg1: i32) :
          %1 = arith.addi %arg0, %arg1 : i32
          linalg.yield %1 : i32
      } -> tensor<?xi32>

  // compute the sums along columns of %rhs.
  %rhs_i32 = arith.extsi %rhs : tensor<?x?xi8> to tensor<?x?xi32>
  %init_rhs_sums_uninitialized = tensor.empty(%n_size) : tensor<?xi32>
  %zero_rhs_sums = linalg.fill ins(%c_0 : i32) outs(%init_rhs_sums_uninitialized : tensor<?xi32>) -> tensor<?xi32>
  %rhs_sums = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%rhs_i32 : tensor<?x?xi32>)
      outs(%zero_rhs_sums : tensor<?xi32>) {
      ^bb0(%arg0: i32, %arg1: i32) :
          %1 = arith.addi %arg0, %arg1 : i32
          linalg.yield %1 : i32
      } -> tensor<?xi32>

  // add all the terms together.
  %quantized_matmul_from_matmul_result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul_result, %lhs_sums, %rhs_sums, %lhs_zp, %rhs_zp, %k_size_i32 : tensor<?x?xi32>, tensor<?xi32>, tensor<?xi32>, i32, i32, i32)
      outs(%init_acc_uninitialized : tensor<?x?xi32>) {
      ^bb0(%matmul_result_val : i32, %lhs_sums_val: i32, %rhs_sums_val: i32, %lhs_zp_val: i32, %rhs_zp_val: i32, %k : i32, %acc_val: i32) :
          %linear_term_in_rhs_zp = arith.muli %lhs_sums_val, %rhs_zp_val : i32
          %linear_term_in_lhs_zp = arith.muli %rhs_sums_val, %lhs_zp_val : i32
          %linear_term = arith.addi %linear_term_in_rhs_zp, %linear_term_in_lhs_zp : i32
          %product_of_zp = arith.muli %lhs_zp_val, %rhs_zp_val : i32
          %quadratic_term = arith.muli %k, %product_of_zp : i32
          %corrected_for_linear_term = arith.subi %matmul_result_val, %linear_term : i32
          %corrected = arith.addi %corrected_for_linear_term, %quadratic_term : i32
          linalg.yield %corrected : i32
      } -> tensor<?x?xi32>
  return %quantized_matmul_from_matmul_result : tensor<?x?xi32>
}

// Checks that linalg.quantized_matmul agrees with @quantized_matmul_as_matmul_3x4x5
func.func private @check_one_quantized_matmul_as_matmul_3x4x5(%lhs : tensor<3x4xi8>, %rhs : tensor<4x5xi8>, %lhs_zp : i32, %rhs_zp : i32) {
    %c_0 = arith.constant 0 : i32
    %init_acc_uninitialized =  tensor.empty() : tensor<3x5xi32>
    %zero_acc = linalg.fill ins(%c_0 : i32) outs(%init_acc_uninitialized : tensor<3x5xi32>) -> tensor<3x5xi32>
    %result_of_quantized_matmul = linalg.quantized_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) outs(%zero_acc : tensor<3x5xi32>) -> tensor<3x5xi32>
    %result_of_quantized_matmul_as_matmul = call @quantized_matmul_as_matmul_3x4x5(%lhs, %rhs, %lhs_zp, %rhs_zp) : (tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) -> tensor<3x5xi32>
    check.expect_eq(%result_of_quantized_matmul, %result_of_quantized_matmul_as_matmul) : tensor<3x5xi32>
    return
}

// Checks that linalg.quantized_matmul agrees with @quantized_matmul_as_matmul_dynamic
func.func private @check_one_quantized_matmul_as_matmul_dynamic(%lhs : tensor<?x?xi8>, %rhs : tensor<?x?xi8>, %lhs_zp : i32, %rhs_zp : i32) {
    %c_0_index = arith.constant 0 : index
    %c_1_index = arith.constant 1 : index
    %m_size = tensor.dim %lhs, %c_0_index : tensor<?x?xi8>
    %n_size = tensor.dim %rhs, %c_1_index : tensor<?x?xi8>

    %c_0 = arith.constant 0 : i32
    %init_acc_uninitialized =  tensor.empty(%m_size, %n_size) : tensor<?x?xi32>
    %zero_acc = linalg.fill ins(%c_0 : i32) outs(%init_acc_uninitialized : tensor<?x?xi32>) -> tensor<?x?xi32>

    %result_of_quantized_matmul = linalg.quantized_matmul ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<?x?xi8>, tensor<?x?xi8>, i32, i32) outs(%zero_acc : tensor<?x?xi32>) -> tensor<?x?xi32>
    %result_of_quantized_matmul_as_matmul = call @quantized_matmul_as_matmul_dynamic(%lhs, %rhs, %lhs_zp, %rhs_zp) : (tensor<?x?xi8>, tensor<?x?xi8>, i32, i32) -> tensor<?x?xi32>
    check.expect_eq(%result_of_quantized_matmul, %result_of_quantized_matmul_as_matmul) : tensor<?x?xi32>
    return
}

func.func @test_quantized_matmul_as_matmul() {
  %lhs_3x4_1 = util.unfoldable_constant dense<[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<3x4xi8>
  %rhs_4x5_1 = util.unfoldable_constant dense<[
      [5, 4, 3, 2, 9],
      [1, 0, -1, -2, 8],
      [-3, -4, -5, -6, 7],
      [2, 3, 5, 7, 11]]> : tensor<4x5xi8>
  // matrices with larger values including the interval bounds -128 and +127.
  %lhs_3x4_2 = util.unfoldable_constant dense<[
      [127, -128, 0, 51],
      [-47, 101, -119, 0],
      [-128, 89, -63, 127]]> : tensor<3x4xi8>
  %rhs_4x5_2 = util.unfoldable_constant dense<[
      [123, -125, 127, -128, 91],
      [-70, 37, 0, -40, 57],
      [-128, 127, -121, -100, 99],
      [127, 105, 83, 51, -128]]> : tensor<4x5xi8>
  %c_0 = arith.constant 0 : i32
  %c_minus2 = arith.constant -2 : i32
  %c_plus3 = arith.constant 3 : i32
  %c_plus41 = arith.constant 41 : i32
  %c_minus57 = arith.constant -57 : i32
  %c_minus128 = arith.constant -128 : i32
  %c_plus127 = arith.constant 127 : i32

  // Test special case: both zero points are 0
  call @check_one_quantized_matmul_as_matmul_3x4x5(%lhs_3x4_1, %rhs_4x5_1, %c_0, %c_0) : (tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) -> ()
  // Test special cases: one of the zero points is 0
  call @check_one_quantized_matmul_as_matmul_3x4x5(%lhs_3x4_1, %rhs_4x5_1, %c_0, %c_plus3) : (tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) -> ()
  call @check_one_quantized_matmul_as_matmul_3x4x5(%lhs_3x4_1, %rhs_4x5_1, %c_minus2, %c_0) : (tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) -> ()
  // Test general cases: both zero points are nonzero
  call @check_one_quantized_matmul_as_matmul_3x4x5(%lhs_3x4_1, %rhs_4x5_1, %c_minus2, %c_plus3) : (tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) -> ()
  call @check_one_quantized_matmul_as_matmul_3x4x5(%lhs_3x4_2, %rhs_4x5_2, %c_plus41, %c_minus57) : (tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) -> ()
  call @check_one_quantized_matmul_as_matmul_3x4x5(%lhs_3x4_2, %rhs_4x5_2, %c_minus128, %c_plus127) : (tensor<3x4xi8>, tensor<4x5xi8>, i32, i32) -> ()

  %lhs_3x4_dynamic = tensor.cast %lhs_3x4_1 : tensor<3x4xi8> to tensor<?x?xi8>
  %rhs_4x5_dynamic = tensor.cast %rhs_4x5_1 : tensor<4x5xi8> to tensor<?x?xi8>
  call @check_one_quantized_matmul_as_matmul_dynamic(%lhs_3x4_dynamic, %rhs_4x5_dynamic, %c_minus128, %c_plus127) : (tensor<?x?xi8>, tensor<?x?xi8>, i32, i32) -> ()

  return
}
