// RUN: iree-dialects-opt -linalg-interp-transforms %s | FileCheck %s


// CHECK-LABEL: func @pad_unary
func @pad_unary(%arg0: tensor<24x12xf32>,
                %arg1: tensor<24x12xf32>) -> tensor<24x12xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c4 = arith.constant 4 : index

  //     CHECK:   scf.for
  //     CHECK:     tensor.pad
  //     CHECK:     linalg.generic
  //     CHECK:   scf.for
  %0 = scf.for %arg3 = %c0 to %c12 step %c4 iter_args(%arg2 = %arg1) -> (tensor<24x12xf32>) {
    %1 = tensor.extract_slice %arg0[0, %arg3] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>
    %2 = tensor.extract_slice %arg2[0, %arg3] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>

    //     CHECK:     linalg.generic
    //     CHECK:     tensor.pad
    //     CHECK:     linalg.elemwise_unary
    %3 = linalg.elemwise_unary ins(%1 : tensor<24x4xf32>)
                              outs(%2: tensor<24x4xf32>) -> tensor<24x4xf32>
    %4 = tensor.insert_slice %3 into %arg2[0, %arg3] [24, 4] [1, 1] : tensor<24x4xf32> into tensor<24x12xf32>
    scf.yield %4 : tensor<24x12xf32>
  }
  return %0 : tensor<24x12xf32>
}

pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = pdl.operation "linalg.elemwise_unary"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@pad_unary](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  %1 = pad %0 {pack_paddings=[1, 1], hoist_paddings=[1, 0], transpose_paddings=[[1, 0], [0, 1]]}
}
