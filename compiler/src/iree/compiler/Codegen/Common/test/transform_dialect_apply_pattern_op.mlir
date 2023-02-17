// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi eq, %arg0, %arg1 : i64
  %1 = arith.select %0, %arg0, %arg1 : i64
  return %1 : i64
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %0 { canonicalization }
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0 * 4)>

// CHECK-LABEL: @promote
func.func @promote() -> (tensor<16x128xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  %empty = tensor.empty() : tensor<16x128xf32>
  %filled = linalg.fill ins(%f0 : f32) outs(%empty : tensor<16x128xf32>) -> tensor<16x128xf32>

  // CHECK: forall{{.*}}shared_outs(%[[ARG:.*]] =
  // CHECK:   %[[A:.*]] = tensor.extract_slice %[[ARG]]
  // CHECK:   %[[B:.*]] = tensor.extract_slice %[[ARG]]
  // CHECK:   %[[C:.*]] = linalg.generic{{.*}}ins(%[[A]]{{.*}}outs(%[[B]]
  %10 = scf.forall (%arg0, %arg1) in (%c16, %c32) shared_outs(%arg2 = %filled) -> (tensor<16x128xf32>) {
    %11 = affine.apply #map2(%arg1)
    %extracted_slice = tensor.extract_slice %filled[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
    %extracted_slice_2 = tensor.extract_slice %arg2[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x4xf32>) outs(%extracted_slice_2 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %res = arith.addf %in, %in: f32
      linalg.yield %res : f32
    } -> tensor<1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg2[%arg0, %11] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<16x128xf32>
    }
  }
  return %10 : tensor<16x128xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = transform.cast %0 : !pdl.operation to !transform.op<"scf.forall">
  transform.iree.share_forall_operands %1 share_operands = [0] : (!transform.op<"scf.forall">) -> !transform.op<"scf.forall">
}

// -----

#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func private @mutate(f32) -> f32

// CHECK-LABEL: @bubble_up
func.func @bubble_up(%arg0: tensor<32x64xf32>) -> tensor<32x2x32xf32> {
  // Check that shape expansion precedes linalg.generic after the patterns were applied.
  // CHECK: tensor.expand_shape
  // CHECK: tensor.expand_shape
  // CHECK: linalg.generic
  %init = tensor.empty() : tensor<32x64xf32>
  %result = linalg.generic {
    indexing_maps = [#map2, #map2],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg0: tensor<32x64xf32>) outs(%init: tensor<32x64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %0 = func.call @mutate(%arg1) : (f32) -> f32
    linalg.yield %0 : f32
  } -> tensor<32x64xf32>
  %out = tensor.expand_shape %result[[0], [1, 2]] : tensor<32x64xf32> into tensor<32x2x32xf32>
  return %out : tensor<32x2x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %0 { bubble_collapse_expand }
}
