// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// CHECK-LABEL: @store_to_load
//  CHECK-SAME:   (%[[ARG:.+]]: vector<4xf32>)
//   CHECK-NOT:   memref.alloc()
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   return %[[ARG]] : vector<4xf32>
func.func @store_to_load(%arg: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  vector.transfer_write %arg, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, memref<64xf32>
  %r = vector.transfer_read %alloc[%c0], %cst_1 {in_bounds = [true]} : memref<64xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  transform.sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_buffer_optimizations %0 : (!transform.any_op) -> ()
  }
}
