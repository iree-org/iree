// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @distribute_lane_forall(%arg0: memref<128x128xf32>, %dest: memref<128x128xf32>) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.forall (%id) in (64) {
      %ids:2 = affine.delinearize_index %id into (%c4, %c16) : index, index
      %3 = affine.apply #map(%ids#0)
      %4 = affine.apply #map1(%ids#1)
      %in_view = memref.subview %arg0[%3, %4] [32, 8] [1, 1] : memref<128x128xf32> to memref<32x8xf32, strided<[128, 1], offset: ?>>
      %out_view = memref.subview %dest[%3, %4] [32, 8] [1, 1] : memref<128x128xf32> to memref<32x8xf32, strided<[128, 1], offset: ?>>
      linalg.copy ins(%in_view : memref<32x8xf32, strided<[128, 1], offset: ?>>) outs(%out_view : memref<32x8xf32, strided<[128, 1], offset: ?>>)
    } {mapping = [#iree_gpu.lane_id<0>]}
    return
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_lanes %func : (!transform.any_op) -> ()
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 16)>

// CHECK-LABEL: func @distribute_lane_forall
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//   CHECK-NOT:   scf.forall
//       CHECK:   affine.delinearize_index %[[LANE_ID]] into (%c4, %c16) : index, index
//       CHECK:   linalg.copy
