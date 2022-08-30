// RUN: iree-dialects-opt %s | FileCheck %s

// CHECK: transform.structured.canonicalized_sequence
transform.structured.canonicalized_sequence {
^bb0(%arg0: !pdl.operation):
  // CHECK: %[[OPS:.*]] = pdl_match @match1 in %{{.*}}
  %0 = pdl_match @match1 in %arg0
  // CHECK: %[[TILED:.*]], %{{.*}}:3 = transform.structured.tile %[[OPS]][4, 4, 4]
  %1, %loops1:3 = transform.structured.tile %0 [4, 4, 4]
  // CHECK: %[[TILED2:.*]], %{{.*}}:3 = transform.structured.tile %[[TILED]]
  %2, %loops2:3  = transform.structured.tile %1 [2, 2, 2]
  // CHECK: %[[PADDED:.*]] = transform.structured.pad %[[TILED2]] {pack_paddings = [1, 1, 0]}
  %3 = transform.structured.pad %2 {pack_paddings = [1, 1, 0]}
  // CHECK: %{{.*}} = transform.structured.vectorize %[[PADDED]] {vectorize_padding = true}
  %4 = transform.structured.vectorize %3 {vectorize_padding = true}
  // CHECK: %[[OPS2:.*]] = pdl_match @{{.*}}
  %5 = pdl_match @match2 in %arg0
  // CHECK: transform.structured.vectorize %[[OPS2]]
  transform.structured.vectorize %5
  // CHECK: bufferize
  bufferize
  // CHECK: lower_vectors {multireduction_lowering = "innerreduce"}
  lower_vectors { multireduction_lowering = "innerreduce"}
  // CHECK: lower_to_llvm
  lower_to_llvm
}
