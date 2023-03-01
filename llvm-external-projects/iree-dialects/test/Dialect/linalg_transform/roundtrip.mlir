// RUN: iree-dialects-opt %s | FileCheck %s

// CHECK: transform.structured.canonicalized_sequence
transform.structured.canonicalized_sequence failures(propagate) {
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
  %3 = transform.structured.pad %2 {pack_paddings = [1, 1, 0]}
  // CHECK: %{{.*}} = transform.structured.vectorize %[[PADDED]] {vectorize_padding}
  %4 = transform.structured.vectorize %3 { vectorize_padding }
  // CHECK: %[[OPS2:.*]] = pdl_match @{{.*}}
  %5 = pdl_match @match2 in %arg0 : (!pdl.operation) -> !pdl.operation
  // CHECK: transform.structured.vectorize %[[OPS2]]
  transform.structured.vectorize %5
  // CHECK: %[[FUNC:.*]] = transform.structured.match ops{["func.func"]} in %arg0
  // CHECK: lower_vectors %[[FUNC]] {{.*}} multireduction_lowering = innerreduction
  %6 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.vector.lower_vectors %6 multireduction_lowering = "innerreduction"
  // CHECK: lower_to_llvm
  lower_to_llvm
}
