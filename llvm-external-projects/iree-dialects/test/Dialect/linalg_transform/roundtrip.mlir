// RUN: iree-dialects-opt %s | FileCheck %s

// CHECK: transform.sequence
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // CHECK: %[[OPS:.*]] = pdl_match @match1 in %{{.*}}
  %0 = pdl_match @match1 in %arg0 : (!transform.any_op) -> !transform.any_op
  // CHECK: %[[TILED:.*]], %{{.*}}:3 = transform.structured.tile %[[OPS]][4, 4, 4]
  %1, %loops1:3 = transform.structured.tile %0 [4, 4, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  // CHECK: %[[TILED2:.*]], %{{.*}}:3 = transform.structured.tile %[[TILED]]
  %2, %loops2:3  = transform.structured.tile %1 [2, 2, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  // CHECK: %[[PADDED:.*]] = transform.structured.pad %[[TILED2]] {pack_paddings = [1, 1, 0]}
  %3 = transform.structured.pad %2 {pack_paddings = [1, 1, 0]}
  // CHECK: %{{.*}} = transform.structured.vectorize %[[PADDED]] {vectorize_padding}
  %4 = transform.structured.vectorize %3 { vectorize_padding }
  // CHECK: %[[OPS2:.*]] = pdl_match @{{.*}}
  %5 = pdl_match @match2 in %arg0 : (!transform.any_op) -> !transform.any_op
  // CHECK: transform.structured.vectorize %[[OPS2]]
  transform.structured.vectorize %5
  // CHECK: %[[FUNC:.*]] = transform.structured.match ops{["func.func"]} in %arg0
  // CHECK: vector.lower_contraction %[[FUNC]] {{.*}}
  %6 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.vector.lower_contraction %6
    lowering_strategy = "outerproduct"
      : (!transform.any_op) -> !transform.any_op
  // CHECK: lower_to_llvm
  lower_to_llvm %arg0 : (!transform.any_op) -> !transform.any_op
}
