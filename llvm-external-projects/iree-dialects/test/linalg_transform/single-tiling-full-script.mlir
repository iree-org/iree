// RUN: iree-dialects-opt -linalg-interp-transforms %s | FileCheck %s

// CHECK-LABEL: func @matmul_tensors
// CHECK-NOT: linalg
// CHECK: llvm
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}


pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  %1 = tile %0 {sizes = [4, 4, 4]}
  %2 = vectorize %1 {vectorize_padding = true}
  bufferize
  lower_vectors { multireduction_lowering = "innerreduce"}
  lower_to_llvm
}
