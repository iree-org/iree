// RUN: iree-dialects-opt --transform-dialect-drop-schedule %s | FileCheck %s

func.func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-NOT: pdl.pattern
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute = @matmul_tensors
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.apply"
  }

  // CHECK-NOT: sequence
  transform.sequence %arg0: !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_target in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.structured.tile %0 [4, 4, 4]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  }
}
