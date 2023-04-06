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


transform.sequence failures(propagate) {
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


  %func = transform.structured.match ops{["func.func"]} in %module_op
    : (!pdl.operation) -> !pdl.operation
  %func_e_2 = transform.vector.lower_contraction %func
    lowering_strategy = "outerproduct"
      : (!pdl.operation) -> !pdl.operation
  %func_e_3 = transform.vector.lower_transpose %func_e_2
    lowering_strategy = "shuffle"
      : (!pdl.operation) -> !pdl.operation

  lower_to_llvm %module_op : (!pdl.operation) -> !pdl.operation
}
