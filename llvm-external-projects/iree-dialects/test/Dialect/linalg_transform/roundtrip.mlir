// RUN: iree-dialects-opt %s | FileCheck %s

// CHECK: iree_linalg_transform.sequence
iree_linalg_transform.sequence {
  // CHECK: %[[OPS:.*]] = match @{{.*}}
  %0 = match @match1
  // CHECK: %[[TILED:.*]], %{{.*}}:3 = tile %[[OPS]] {
  // CHECK-DAG: sizes = [4, 4, 4]
  // CHECK: }
  %1, %loops1:3 = tile %0 {sizes = [4, 4, 4]}
  // CHECK: %[[TILED2:.*]], %{{.*}}:3 = tile %[[TILED]]
  %2, %loops2:3  = tile %1 {sizes = [2, 2, 2]}
  // CHECK: %[[PADDED:.*]] = pad %[[TILED2]] {pack_paddings = [1, 1, 0]}
  %3 = pad %2 {pack_paddings = [1, 1, 0]}
  // CHECK: decompose
  decompose
  // CHECK: %{{.*}} = vectorize %[[PADDED]] {vectorize_padding = true}
  %4 = vectorize %3 {vectorize_padding = true}
  // CHECK: %[[OPS2:.*]] = match @{{.*}}
  %5 = match @match2
  // CHECK: %{{.*}} = vectorize %[[OPS2]]
  vectorize %5
  // CHECK-NOT: %
  // CHECK: vectorize
  // CHECK-NOT: %
  vectorize
  // CHECK: bufferize
  bufferize
  // CHECK: lower_vectors {multireduction_lowering = "innerreduce"}
  lower_vectors { multireduction_lowering = "innerreduce"}
  // CHECK: lower_to_llvm
  lower_to_llvm
}
