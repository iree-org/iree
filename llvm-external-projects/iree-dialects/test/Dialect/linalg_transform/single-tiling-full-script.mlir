// RUN: iree-dialects-opt --transform-dialect-interpreter %s | FileCheck %s

// CHECK-LABEL: func @matmul_tensors
// CHECK-NOT: linalg
// CHECK: llvm
func.func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}


transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile %0 [4, 4, 4]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %2 { vectorize_padding }
  transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module_op
    {bufferize_function_boundaries = true}
  %3 = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> !pdl.operation
  transform.vector.lower_vectors %3 multireduction_lowering = "innerreduction"

  // lower_to_llvm is the only remaining op not upstreamed, at the same time we
  // upstreamed --test-lower-to-llvm.
  lower_to_llvm
}
