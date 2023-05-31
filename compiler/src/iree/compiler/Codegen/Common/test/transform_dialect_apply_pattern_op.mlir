// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi eq, %arg0, %arg1 : i64
  %1 = arith.select %0, %arg0, %arg1 : i64
  return %1 : i64
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %0 { canonicalization } : (!transform.any_op) -> ()
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
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.forall">
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
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %0 { bubble_expand } : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: @pad_fill_to_fill
func.func @pad_fill_to_fill(%arg0: tensor<31x62xf32>) -> tensor<32x64xf32> {
  // Check that a pad of a fill with the same constant is replaced by a
  // bigger fill.
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.0{{0*e\+00}} : f32
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[PADDED_FILL:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[EMPTY]] : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: return %[[PADDED_FILL]]
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %fill = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<31x62xf32>) -> tensor<31x62xf32>
  %padded = tensor.pad %fill low[%c0, %c0] high[%c1, %c2] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
  } : tensor<31x62xf32> to tensor<32x64xf32>
  return %padded : tensor<32x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %0 { tiling_canonicalization } : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: @pad_fill_different_ssa_value_but_same_cst
func.func @pad_fill_different_ssa_value_but_same_cst(%arg0: tensor<31x62xf32>) -> tensor<32x64xf32> {
  // Check that a pad of a fill with the same constant is replaced by a
  // bigger fill even when the constant comes from different ssa value.
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.0{{0*e\+00}} : f32
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[PADDED_FILL:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[EMPTY]] : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: return %[[PADDED_FILL]]
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %fill = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<31x62xf32>) -> tensor<31x62xf32>
  %padded = tensor.pad %fill low[%c0, %c0] high[%c1, %c2] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst2 : f32
  } : tensor<31x62xf32> to tensor<32x64xf32>
  return %padded : tensor<32x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %0 { tiling_canonicalization } : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: @pad_extract_fill_to_fill
func.func @pad_extract_fill_to_fill(%arg0: tensor<31x62xf32>,
    %size0 : index, %size1 : index,
    %high0 : index, %high1 : index) -> tensor<32x64xf32> {
  // Check that a pad of a fill with the same constant is replaced by a
  // bigger fill even when the fill is hidden behind an extract_slice.
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.0{{0*e\+00}} : f32
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[PADDED_FILL:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[EMPTY]] : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: return %[[PADDED_FILL]]
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %fill = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<31x62xf32>) -> tensor<31x62xf32>
  %extracted_slice = tensor.extract_slice %fill[0, 0] [%size0, %size1] [1, 1] : tensor<31x62xf32> to tensor<?x?xf32>
  %padded = tensor.pad %extracted_slice low[%c0, %c0] high[%high0, %high1] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst2 : f32
  } : tensor<?x?xf32> to tensor<32x64xf32>
  return %padded : tensor<32x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %0 { tiling_canonicalization } : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: @pad_extract_extract_fill_to_fill
func.func @pad_extract_extract_fill_to_fill(%arg0: tensor<31x62xf32>,
    %size0a : index, %size1a : index,
    %size0b : index, %size1b : index,
    %high0 : index, %high1 : index) -> tensor<32x64xf32> {
  // Check that a pad of a fill with the same constant is replaced by a
  // bigger fill even when the fill is hidden behind a few `extract_slice`s.
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.0{{0*e\+00}} : f32
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[PADDED_FILL:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[EMPTY]] : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: return %[[PADDED_FILL]]
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %fill = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<31x62xf32>) -> tensor<31x62xf32>
  %extracted_sliceA = tensor.extract_slice %fill[0, 0] [%size0a, %size1a] [1, 1] : tensor<31x62xf32> to tensor<?x?xf32>
  %extracted_sliceB = tensor.extract_slice %extracted_sliceA[0, 0] [%size0b, %size1b] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %padded = tensor.pad %extracted_sliceB low[%c0, %c0] high[%high0, %high1] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst2 : f32
  } : tensor<?x?xf32> to tensor<32x64xf32>
  return %padded : tensor<32x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %0 { tiling_canonicalization } : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: @pad_extract_bigger_fill_to_fill
func.func @pad_extract_bigger_fill_to_fill(%arg0: tensor<253x123xf32>,
    %size0 : index, %size1 : index,
    %high0 : index, %high1 : index) -> tensor<32x64xf32> {
  // Check that a pad of a bigger fill with the same constant is replaced by a
  // fill of the right size.
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.0{{0*e\+00}} : f32
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[PADDED_FILL:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[EMPTY]] : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: return %[[PADDED_FILL]]
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %fill = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<253x123xf32>) -> tensor<253x123xf32>
  %extracted_slice = tensor.extract_slice %fill[0, 0] [%size0, %size1] [1, 1] : tensor<253x123xf32> to tensor<?x?xf32>
  %padded = tensor.pad %extracted_slice low[%c0, %c0] high[%high0, %high1] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst2 : f32
  } : tensor<?x?xf32> to tensor<32x64xf32>
  return %padded : tensor<32x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %0 { tiling_canonicalization } : (!transform.any_op) -> ()
}
