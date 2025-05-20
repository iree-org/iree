// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | FileCheck %s

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0)>
func.func @pack_consumer_fusion(%arg0: tensor<32xf32>) -> tensor<2x16xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = scf.forall (%arg1) in (2) shared_outs(%arg2 = %0) -> (tensor<32xf32>) {
    %3 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<16xf32>) outs(%extracted_slice_0 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %in : f32
      linalg.yield %5 : f32
    } -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %arg2[%3] [16] [1] : tensor<16xf32> into tensor<32xf32>
    }
  }
  %2 = tensor.empty() : tensor<2x16xf32>
  %pack = linalg.pack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %2 : tensor<32xf32> -> tensor<2x16xf32>
  return %pack : tensor<2x16xf32>
}
// CHECK-LABEL: @pack_consumer_fusion
// CHECK:       scf.forall
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK:         %[[PACK:.+]] = linalg.pack %[[GENERIC]]
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice %[[PACK]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %slice_op = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.iree.fuse_consumer %slice_op in (%loop)
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
     transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }
}
