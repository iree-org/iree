// RUN: iree-dialects-opt -linalg-interp-transforms %s | FileCheck %s


// CHECK-LABEL: func @generalize_unary
func @generalize_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  // CHECK-NOT:   linalg.elemwise_unary
  //     CHECK:   linalg.generic
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}


pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = pdl.operation "linalg.elemwise_unary"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@generalize_unary](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  generalize %0
}
