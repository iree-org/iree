// RUN: iree-dialects-opt -linalg-interp-transforms %s 
// TODO: enable once https://reviews.llvm.org/D121369 lands
// | FileCheck %s

// This test is verifying that a non-trivial 2*tiling+padding+vectorization transformation completes successfully

// CHECK-LABEL: func @matmul_tensors(
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // Pack transposed padding of 1st operand.
  //      CHECK:    tensor.pad
  //      CHECK:    linalg.generic

  // Pack padding of 2nd operand.
  //      CHECK:    tensor.pad

  //      CHECK:      scf.for
  //      CHECK:        scf.for
  //      CHECK:          scf.for
  //      CHECK:            scf.for
  //      CHECK:              scf.for
  //      CHECK:                linalg.generic
  //      CHECK:                vector.contract
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}

pdl.pattern @pdl_target: benefit(1) {
  %args = operands
  %results= types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  rewrite %0 with "iree_linalg_transform.apply"
}
iree_linalg_transform.sequence {
  %0 = match @pdl_target
  %1 = tile %0 {interchange = [0, 2, 1], peel = [], scalarize_dyn_dims = false, sizes = [32, 32, 32]}
  %2 = tile %1 {interchange = [0, 1, 2], peel = [], scalarize_dyn_dims = false, sizes = [4, 4, 1]}
  %3 = pad %2 {pack_paddings = [1, 1, 1], hoist_paddings = [6, 6, 0], transpose_paddings = [[1, 0], [0, 1]]}
  %4 = vectorize %3  {vectorize_padding = true}
}
