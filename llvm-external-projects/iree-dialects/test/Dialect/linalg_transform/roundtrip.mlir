// RUN: iree-dialects-opt %s | FileCheck %s

// CHECK: transform.sequence
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // CHECK: %[[OPS:.*]] = pdl_match @match1 in %{{.*}}
  %0 = pdl_match @match1 in %arg0 : (!pdl.operation) -> !pdl.operation
  // CHECK: %[[TILED:.*]], %{{.*}}:3 = transform.structured.tile %[[OPS]][4, 4, 4]
  %1, %loops1:3 = transform.structured.tile %0 [4, 4, 4]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  // CHECK: %[[TILED2:.*]], %{{.*}}:3 = transform.structured.tile %[[TILED]]
  %2, %loops2:3  = transform.structured.tile %1 [2, 2, 2]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  // CHECK: %[[PADDED:.*]] = transform.structured.pad %[[TILED2]] {pack_paddings = [1, 1, 0]}
  %3 = transform.structured.pad %2 {pack_paddings = [1, 1, 0]} : (!pdl.operation) -> !pdl.operation
  // CHECK: %{{.*}} = transform.structured.vectorize %[[PADDED]] {vectorize_padding}
  %4 = transform.structured.vectorize %3 { vectorize_padding } : (!pdl.operation) -> !pdl.operation
  // CHECK: %[[OPS2:.*]] = pdl_match @{{.*}}
  %5 = pdl_match @match2 in %arg0 : (!pdl.operation) -> !pdl.operation
  // CHECK: transform.structured.vectorize %[[OPS2]]
  transform.structured.vectorize %5 : (!pdl.operation) -> !pdl.operation
  // CHECK: %[[FUNC:.*]] = transform.structured.match ops{["func.func"]} in %arg0
  // CHECK: apply_patterns.vector.lower_contraction
  %6 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.apply_patterns to %6 {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
  } : !pdl.operation
  // CHECK: lower_to_llvm
  lower_to_llvm %arg0 : (!pdl.operation) -> !pdl.operation
}
