// This file is meant to be compiled by
//   iree-opt --iree-flow-export-matmul-test-funcs-pass
// which generates the actual test code. The functions provided in this file are
// only helpers to perform a refernce matmul to compare against, and to generate
// test input matrices.

// Note that the test input matrix generator functions below determine what
// data types are going to be tested by the generated tests. If they generate f32
// matrices then the test will cover f32 matmuls.

module @matmul_i8 {
  // Reference matmul implementation.
  // It must not undergo the same compiler transformations as linalg.matmul,
  // otherwise the whole test becomes vacuous (tests x == x).
  // This relies on linalg.generic NOT being de-generalized (what's the term?) into
  // linalg.matmul.
  // TODO: clarify how we ensure that that is the case?
  func private @reference_matmul(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %acc : tensor<?x?xi32>) -> tensor<?x?xi32> {
    %result =  linalg.generic {
                indexing_maps = [
                  affine_map<(m, k, n) -> (m, k)>,
                  affine_map<(m, k, n) -> (k, n)>,
                  affine_map< (m, k, n) -> (m, n)>
                ],
                iterator_types = ["parallel", "reduction", "parallel"]
              }
              ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>)
              outs(%acc: tensor<?x?xi32>) {
      ^bb0(%lhs_value: i8, %rhs_value: i8, %acc_value: i32):
        %lhs_value_i32 = std.sexti %lhs_value : i8 to i32
        %rhs_value_i32 = std.sexti %rhs_value : i8 to i32
        %product = std.muli %lhs_value_i32, %rhs_value_i32: i32
        %sum = std.addi %product, %acc_value: i32
        linalg.yield %sum: i32
    } -> tensor<?x?xi32>
    return %result : tensor<?x?xi32>
  }

  // Generate a zero matrix. Used to generate LHS and RHS test input matrices.
  func private @zero_matrix(%rows : index, %cols : index) -> tensor<?x?xi8> {
    %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xi8>
    %c0 = constant 0 : i8
    %1 = linalg.fill (%c0, %0) : i8, tensor<?x?xi8> -> tensor<?x?xi8>
    return %1 : tensor<?x?xi8>
  }

  // Generate a zero matrix. Used to generate Accumulator test input matrices.
  func private @zero_accumulator_matrix(%rows : index, %cols : index) -> tensor<?x?xi32> {
    %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xi32>
    %c0 = constant 0 : i32
    %1 = linalg.fill (%c0, %0) : i32, tensor<?x?xi32> -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
  }

  // Generate a pseudorandom matrix. Used to generate LHS and RHS test input matrices.
  // For test portability/reproducibility/determinism, the pseudorandom generator
  // should be fully specified (not implementation-defined).
  // TODO: clarify if that guarantee is offered by linalg.fill_rng_2d?
  func private @random_matrix(%rows : index, %cols : index, %seed : i32) -> tensor<?x?xi8> {
    %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xi8>
    %min = constant -128.0 : f64
    %max = constant 127.0: f64
    %1 = linalg.fill_rng_2d ins(%min, %max, %seed: f64, f64, i32) outs (%0 : tensor<?x?xi8>) -> tensor<?x?xi8>
    return %1 : tensor<?x?xi8>
  }

  // Generate a pseudorandom matrix. Used to generate Accumulator test input matrices.
  func private @random_accumulator_matrix(%rows : index, %cols : index, %seed : i32) -> tensor<?x?xi32> {
    %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xi32>
    %min = constant -1.0e4 : f64
    %max = constant 1.0e4: f64
    %1 = linalg.fill_rng_2d ins(%min, %max, %seed: f64, f64, i32) outs (%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
  }

  // Generate an "identity" matrix, that is, a matrix whose entries are 1 on the main diagonal
  // and 0 elsewhere. Used to generate LHS and RHS test input matrices. This is applied to any
  // rectangular shape, not just square shapes as in true identity matrices. Correspondingly
  // the definining property of being units for multiplication is slightly loosened: I*X
  // is equal to X *suitably zero-extended*, and likewise X*I, for any "identity" rectangular matrix I
  // and matrix X of compatible shape.
  func private @identity_matrix(%rows : index, %cols : index) -> tensor<?x?xi8> {
    %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xi8>
    %1 = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%0 : tensor<?x?xi8>)
        outs(%0 : tensor<?x?xi8>) {
    ^bb0(%arg0: i8, %arg1: i8):  // no predecessors
      %8 = linalg.index 0 : index
      %9 = linalg.index 1 : index
      %10 = cmpi eq, %8, %9 : index
      %cst0 = constant 0 : i8
      %cst1 = constant 1 : i8
      %11 = select %10, %cst1, %cst0 : i8
      linalg.yield %11 : i8
    } -> tensor<?x?xi8>
    return %1 : tensor<?x?xi8>
  }

  // Generate an "identity" matrix. Used to generate Accumulator test input matrices.
  func private @identity_accumulator_matrix(%rows : index, %cols : index) -> tensor<?x?xi32> {
    %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xi32>
    %1 = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%0 : tensor<?x?xi32>)
        outs(%0 : tensor<?x?xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
      %8 = linalg.index 0 : index
      %9 = linalg.index 1 : index
      %10 = cmpi eq, %8, %9 : index
      %cst0 = constant 0 : i32
      %cst1 = constant 1 : i32
      %11 = select %10, %cst1, %cst0 : i32
      linalg.yield %11 : i32
    } -> tensor<?x?xi32>
    return %1 : tensor<?x?xi32>
  }
}
