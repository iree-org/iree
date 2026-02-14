// RUN: iree-opt %s --iree-transform-dialect-interpreter --split-input-file | FileCheck %s

// Test retained to check that this upstreamed functionality still works.t
// CHECK-LABEL: @read_read_write
func.func @read_read_write(%arg0: memref<?xf32>, %arg1: index) attributes {__parallel_region_boundary_for_test} {
  // CHECK: load
  %0 = memref.load %arg0[%arg1] : memref<?xf32>
  // The barrier between loads can be removed.
  // CHECK-NOT: barrier
  gpu.barrier
  // CHECK: load
  %1 = memref.load %arg0[%arg1] : memref<?xf32>
  %2 = arith.addf %0, %1 : f32
  // The barrier between load and store cannot be removed (unless we reason about accessed subsets).
  // CHECK: barrier
  gpu.barrier
  // CHECK: store
  memref.store %2, %arg0[%arg1] : memref<?xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.iree.eliminate_gpu_barriers %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module
