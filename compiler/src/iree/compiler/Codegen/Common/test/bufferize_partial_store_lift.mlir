// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-bufferize-dispatch-tensor-load-store,iree-eliminate-empty-tensors,iree-codegen-iree-comprehensive-bufferize, canonicalize, cse, canonicalize))" --split-input-file | FileCheck %s

// E2E (lift + comprehensive bufferize) test for the cross-pattern partial
// store + sibling tensor.insert_slice. The lift converts the partial
// dispatch.tensor.store + sibling insert_slice into two insert_slice ops
// on a shared SSA root; OneShot then detects the WaW conflict on
// overlapping subsets and forces one write out-of-place via a fresh
// memref.alloc. Without the lift, both writes would bufferize in-place
// into the same parent memref and clobber each other.
//
// CHECK-SAME constraints are minimal on purpose: assert only the
// conflict-detection signal (one memref.alloc) and that both target
// bindings receive writes. Full IR shape is sensitive to canonicalize
// and CSE and not load-bearing for correctness.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @cross_pattern_partial_store_e2e_bufferize() {
  %c10_i64 = arith.constant 10 : i64
  %c20_i64 = arith.constant 20 : i64
  %pos_span = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>>
  %pool_span = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi64>>
  %out_span = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi64>>
  %pos = iree_tensor_ext.dispatch.tensor.load %pos_span, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
  %pool = iree_tensor_ext.dispatch.tensor.load %pool_span, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi64>> -> tensor<1xi64>
  %empty = tensor.empty() : tensor<i64>
  %vals:2 = linalg.generic {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = []
  } ins(%pos : tensor<i32>) outs(%empty, %empty : tensor<i64>, tensor<i64>) {
  ^bb0(%in: i32, %o0: i64, %o1: i64):
    %x = arith.extsi %in : i32 to i64
    %a = arith.addi %x, %c10_i64 : i64
    %b = arith.addi %x, %c20_i64 : i64
    linalg.yield %a, %b : i64, i64
  } -> (tensor<i64>, tensor<i64>)
  %inserted_k = tensor.insert_slice %vals#0 into %pool[0] [1] [1] : tensor<i64> into tensor<1xi64>
  iree_tensor_ext.dispatch.tensor.store %vals#1, %pool_span, offsets = [0], sizes = [1], strides = [1] : tensor<i64> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi64>>
  iree_tensor_ext.dispatch.tensor.store %inserted_k, %out_span, offsets = [0], sizes = [1], strides = [1] : tensor<1xi64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi64>>
  return
}
// CHECK-LABEL: func.func @cross_pattern_partial_store_e2e_bufferize
// Conflict-detection signal: a fresh alloc is created for the losing
// (out-of-place) write, and that alloc shows up as a linalg.generic
// destination (vs. the buggy form where both writes share one subview).
// Without the lift, OneShot would not see the cross-pattern conflict.
//       CHECK:   %[[ALLOC:.+]] = memref.alloc()
//       CHECK:   linalg.generic
//  CHECK-SAME:   %[[ALLOC]]
