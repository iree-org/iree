// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-eliminate-empty-tensors))" %s | FileCheck %s

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @eliminate_empty_tensors_with_store_op() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x384xf32>>
  %1 = tensor.empty() : tensor<32x384xf32>
  scf.for %arg0 = %c0 to %c128 step %c32 {
    %2 = scf.for %arg1 = %c0 to %c32 step %c8 iter_args(%arg2 = %1) -> (tensor<32x384xf32>) {
      scf.yield %arg2 : tensor<32x384xf32>
    }
    iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [%arg0, 0], sizes = [32, 384], strides = [1, 1] : tensor<32x384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x384xf32>>
  }
  return
}

// CHECK-LABEL: @eliminate_empty_tensors_with_store_op
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK: %[[SPAN:.+]] = hal.interface.binding.subspan{{.+}}binding(0){{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x384xf32>>
// CHECK: scf.for %[[ARG0:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SPAN]], offsets = [%[[ARG0]], 0], sizes = [32, 384], strides = [1, 1]
// CHECK:   %[[RES:.+]] = scf.for {{.+}} = %[[C0]] to %[[C32]] step %[[C8]] iter_args({{.+}} = %[[LOAD]])
// CHECK:   iree_tensor_ext.dispatch.tensor.store %[[RES]], %[[SPAN]], offsets = [%[[ARG0]], 0], sizes = [32, 384], strides = [1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @eliminate_empty_tensors_with_store_to_buffer_op() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<128xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<128xf32, #hal.descriptor_type<storage_buffer>>
  %2 = iree_codegen.load_from_buffer %0 : memref<128xf32, #hal.descriptor_type<storage_buffer>> -> tensor<128xf32>
  %3 = tensor.empty() : tensor<128xf32>
  %copy = linalg.copy ins(%2 : tensor<128xf32>) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
  iree_codegen.store_to_buffer %copy, %1 : tensor<128xf32> into memref<128xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: @eliminate_empty_tensors_with_store_to_buffer_op
//     CHECK: %[[INPUT_SPAN:.+]] = hal.interface.binding.subspan{{.+}}binding(0){{.+}}: memref<128xf32
//     CHECK: %[[RESULT_SPAN:.+]] = hal.interface.binding.subspan{{.+}}binding(1){{.+}}: memref<128xf32
// CHECK-DAG: %[[INPUT:.+]] = iree_codegen.load_from_buffer %[[INPUT_SPAN]] : memref<128xf32{{.+}} -> tensor<128xf32>
// CHECK-DAG: %[[INIT:.+]] = iree_codegen.load_from_buffer %[[RESULT_SPAN]] : memref<128xf32{{.+}} -> tensor<128xf32>
//     CHECK: %[[COPY:.+]] = linalg.copy ins(%[[INPUT]] : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
//     CHECK: iree_codegen.store_to_buffer %[[COPY]], %[[RESULT_SPAN]] : tensor<128xf32> into memref<128xf32

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @eliminate_empty_tensors_store_to_buffer_op_with_reshape() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<128xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<4x32xf32>
  %2 = iree_codegen.load_from_buffer %0 : memref<128xf32> -> tensor<128xf32>
  %3 = tensor.empty() : tensor<128xf32>
  %copy = linalg.copy ins(%2 : tensor<128xf32>) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
  %4 = memref.collapse_shape %1 [[0, 1]] : memref<4x32xf32> into memref<128xf32>
  iree_codegen.store_to_buffer %copy, %4 : tensor<128xf32> into memref<128xf32>
  return
}

// CHECK-LABEL: @eliminate_empty_tensors_store_to_buffer_op_with_reshape
//     CHECK: %[[INPUT_SPAN:.+]] = hal.interface.binding.subspan{{.+}}binding(0){{.+}}: memref<128xf32>
//     CHECK: %[[RESULT_SPAN:.+]] = hal.interface.binding.subspan{{.+}}binding(1){{.+}}: memref<4x32xf32>
// CHECK-DAG: %[[RESULT_RESHAPE:.+]] = memref.collapse_shape %[[RESULT_SPAN]] {{.+}} : memref<4x32xf32> into memref<128xf32>
// CHECK-DAG: %[[INPUT:.+]] = iree_codegen.load_from_buffer %[[INPUT_SPAN]] : memref<128xf32> -> tensor<128xf32>
// CHECK-DAG: %[[INIT:.+]] = iree_codegen.load_from_buffer %[[RESULT_RESHAPE]] : memref<128xf32> -> tensor<128xf32>
//     CHECK: %[[COPY:.+]] = linalg.copy ins(%[[INPUT]] : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
//     CHECK: iree_codegen.store_to_buffer %[[COPY]], %[[RESULT_RESHAPE]] : tensor<128xf32> into memref<128xf32>

// -----

// Cross-pattern partial store + sibling tensor.insert_slice into the same
// HAL binding. A multi-output `linalg.generic` produces two scalars: one
// flows to `tensor.insert_slice` into a `load_from_buffer` of the binding,
// the other to a partial `store_to_buffer` to a `memref.subview` of the
// same binding. Without the lift these two writes are SSA-disjoint at
// the tensor level, so OneShot's read-after-write analysis fails to
// detect the overlap, both writes bufferize in-place, and one clobbers
// the other.
//
// The lift rewrites the partial store as `store_to_buffer (insert_slice
// %v into load_from_buffer(%parent)[subset]), %parent`, reusing any
// existing full-tensor load of the parent. After the lift, both writes
// are `tensor.insert_slice` on the same SSA root — OneShot detects the
// conflict and bufferizes one out-of-place (writes to a fresh buffer
// instead of the original memory).
#pipeline_layout_kv = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @lift_partial_store_to_buffer_with_sibling_insert() {
  %c0 = arith.constant 0 : index
  %c10_i64 = arith.constant 10 : i64
  %c20_i64 = arith.constant 20 : i64
  %pos = hal.interface.binding.subspan layout(#pipeline_layout_kv) binding(0) alignment(64) offset(%c0) : memref<i32, #hal.descriptor_type<storage_buffer>>
  %pool = hal.interface.binding.subspan layout(#pipeline_layout_kv) binding(1) alignment(64) offset(%c0) : memref<1xi64, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %pool[0] [1] [1] : memref<1xi64, #hal.descriptor_type<storage_buffer>> to memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  %out = hal.interface.binding.subspan layout(#pipeline_layout_kv) binding(2) alignment(64) offset(%c0) : memref<1xi64, #hal.descriptor_type<storage_buffer>>
  %loaded_pos = iree_codegen.load_from_buffer %pos : memref<i32, #hal.descriptor_type<storage_buffer>> -> tensor<i32>
  %loaded_pool = iree_codegen.load_from_buffer %pool : memref<1xi64, #hal.descriptor_type<storage_buffer>> -> tensor<1xi64>
  %empty = tensor.empty() : tensor<i64>
  %vals:2 = linalg.generic {
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = []
  } ins(%loaded_pos : tensor<i32>) outs(%empty, %empty : tensor<i64>, tensor<i64>) {
  ^bb0(%in: i32, %o0: i64, %o1: i64):
    %x = arith.extsi %in : i32 to i64
    %a = arith.addi %x, %c10_i64 : i64
    %b = arith.addi %x, %c20_i64 : i64
    linalg.yield %a, %b : i64, i64
  } -> (tensor<i64>, tensor<i64>)
  %inserted_k = tensor.insert_slice %vals#0 into %loaded_pool[0] [1] [1] : tensor<i64> into tensor<1xi64>
  iree_codegen.store_to_buffer %vals#1, %subview : tensor<i64> into memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  iree_codegen.store_to_buffer %inserted_k, %out : tensor<1xi64> into memref<1xi64, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: @lift_partial_store_to_buffer_with_sibling_insert
//   CHECK-DAG:   %[[POOL_SPAN:.+]] = hal.interface.binding.subspan{{.+}}binding(1){{.+}}: memref<1xi64, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[OUT_SPAN:.+]] = hal.interface.binding.subspan{{.+}}binding(2){{.+}}: memref<1xi64, #hal.descriptor_type<storage_buffer>>
// The lift inserts (or reuses) a single `load_from_buffer` of the parent
// memref. Both the original `tensor.insert_slice` and the lifted one must
// share that load as their SSA source.
//       CHECK:   %[[POOL_TENSOR:.+]] = iree_codegen.load_from_buffer %[[POOL_SPAN]] : memref<1xi64, #hal.descriptor_type<storage_buffer>> -> tensor<1xi64>
// Check that there is exactly one `load_from_buffer` of the parent —
// confirming the buildSubsetExtraction reuse logic kept SSA singular.
//   CHECK-NOT:   iree_codegen.load_from_buffer %[[POOL_SPAN]]
// The original `tensor.insert_slice` (K-side) extracts from POOL_TENSOR.
//       CHECK:   %{{.+}}:2 = linalg.generic
//       CHECK:   tensor.insert_slice {{.+}} into %[[POOL_TENSOR]][0] [1] [1]
// The lifted partial store becomes a `tensor.insert_slice` into
// POOL_TENSOR (V-side) followed by a full-tensor `store_to_buffer` to the
// parent. Reuse of POOL_TENSOR ties the two writes to the same SSA root.
//       CHECK:   tensor.insert_slice {{.+}} into %[[POOL_TENSOR]][0] [1] [1]
//       CHECK:   iree_codegen.store_to_buffer {{.+}}, %[[POOL_SPAN]]
//       CHECK:   iree_codegen.store_to_buffer {{.+}}, %[[OUT_SPAN]]

// -----

// Verify the lift handles a parent memref with dynamic shape. The
// `RankedTensorType::get` constructor must preserve dynamic dims, and
// the reused `iree_codegen.load_from_buffer` must already type as a
// `tensor<?xi64>` matching what the lift expects.
//
// The sibling reader is an in-block `load_from_buffer %pool`. Without
// it the lift would not fire (see `no_lift_when_no_sibling_reader`
// below).
#pipeline_layout_dyn = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @lift_partial_store_dynamic_parent(%size: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  %pool = hal.interface.binding.subspan layout(#pipeline_layout_dyn) binding(0) alignment(64) offset(%c0) : memref<?xi64, #hal.descriptor_type<storage_buffer>>{%size}
  %out = hal.interface.binding.subspan layout(#pipeline_layout_dyn) binding(1) alignment(64) offset(%c0) : memref<?xi64, #hal.descriptor_type<storage_buffer>>{%size}
  // Sibling reader: in-block dominating load of the parent. Must be reused
  // by the lift so both writes share an SSA root.
  %loaded_pool = iree_codegen.load_from_buffer %pool : memref<?xi64, #hal.descriptor_type<storage_buffer>> -> tensor<?xi64>
  %sub = memref.subview %pool[0] [1] [1] : memref<?xi64, #hal.descriptor_type<storage_buffer>> to memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  %v = tensor.from_elements %c1 : tensor<i64>
  // Sibling insert_slice into the loaded parent — flows to a different
  // binding so the load isn't dead.
  %v_sib = tensor.from_elements %c1 : tensor<i64>
  %inserted = tensor.insert_slice %v_sib into %loaded_pool[1] [1] [1] : tensor<i64> into tensor<?xi64>
  iree_codegen.store_to_buffer %v, %sub : tensor<i64> into memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  iree_codegen.store_to_buffer %inserted, %out : tensor<?xi64> into memref<?xi64, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: @lift_partial_store_dynamic_parent
//   CHECK-DAG:   %[[POOL:.+]] = hal.interface.binding.subspan{{.+}}binding(0){{.+}}: memref<?xi64
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan{{.+}}binding(1){{.+}}: memref<?xi64
// Reused in-block load of the dynamic-shape parent.
//       CHECK:   %[[T:.+]] = iree_codegen.load_from_buffer %[[POOL]] : memref<?xi64{{.+}} -> tensor<?xi64>
// Lifted partial store: insert_slice into %[[T]], then full store to %[[POOL]].
//       CHECK:   tensor.insert_slice {{.+}} into %[[T]]
//       CHECK:   iree_codegen.store_to_buffer {{.+}}, %[[POOL]] : tensor<?xi64> into memref<?xi64

// -----

// Verify a 3-level subview chain is fully unwound to the topmost binding
// after 3 iterations of the fixed-point loop. Each iteration needs its
// own in-block sibling reader (a `load_from_buffer` of that level's
// parent), so we pre-seed loads for all three levels — `%l2`, `%l1`,
// and `%pool` — and let the iteration consume them in turn.
#pipeline_layout_3lvl = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @lift_3level_subview_chain() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  %pool = hal.interface.binding.subspan layout(#pipeline_layout_3lvl) binding(0) alignment(64) offset(%c0) : memref<2x2x4xi64, #hal.descriptor_type<storage_buffer>>
  %l1 = memref.subview %pool[0, 0, 0] [1, 2, 4] [1, 1, 1] : memref<2x2x4xi64, #hal.descriptor_type<storage_buffer>> to memref<2x4xi64, strided<[4, 1]>, #hal.descriptor_type<storage_buffer>>
  %l2 = memref.subview %l1[0, 0] [1, 4] [1, 1] : memref<2x4xi64, strided<[4, 1]>, #hal.descriptor_type<storage_buffer>> to memref<4xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>>
  %l3 = memref.subview %l2[0] [1] [1] : memref<4xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>> to memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  // Sibling readers — one per chain level. The pass does not DCE, so
  // unused loads survive and are picked up by the lift's user-walk.
  %loaded_l2 = iree_codegen.load_from_buffer %l2 : memref<4xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>> -> tensor<4xi64>
  %loaded_l1 = iree_codegen.load_from_buffer %l1 : memref<2x4xi64, strided<[4, 1]>, #hal.descriptor_type<storage_buffer>> -> tensor<2x4xi64>
  %loaded_pool = iree_codegen.load_from_buffer %pool : memref<2x2x4xi64, #hal.descriptor_type<storage_buffer>> -> tensor<2x2x4xi64>
  %v = tensor.from_elements %c1 : tensor<i64>
  iree_codegen.store_to_buffer %v, %l3 : tensor<i64> into memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: @lift_3level_subview_chain
//   CHECK-DAG:   %[[POOL:.+]] = hal.interface.binding.subspan{{.+}}: memref<2x2x4xi64
//   CHECK-DAG:   %[[L1:.+]] = memref.subview %[[POOL]][0, 0, 0] [1, 2, 4] [1, 1, 1]
//   CHECK-DAG:   %[[L2:.+]] = memref.subview %[[L1]][0, 0] [1, 4] [1, 1]
// All three sibling readers (the in-block loads we pre-seeded) are reused.
//   CHECK-DAG:   %[[L2_T:.+]] = iree_codegen.load_from_buffer %[[L2]] : memref<4xi64{{.+}} -> tensor<4xi64>
//   CHECK-DAG:   %[[L1_T:.+]] = iree_codegen.load_from_buffer %[[L1]] : memref<2x4xi64{{.+}} -> tensor<2x4xi64>
//   CHECK-DAG:   %[[POOL_T:.+]] = iree_codegen.load_from_buffer %[[POOL]] : memref<2x2x4xi64{{.+}} -> tensor<2x2x4xi64>
// Iteration 1 peels innermost (L3 → L2): insert into the reused L2 load.
//       CHECK:   %[[I1:.+]] = tensor.insert_slice {{.+}} into %[[L2_T]][0] [1] [1] : tensor<i64> into tensor<4xi64>
// Iteration 2 peels middle (L2 → L1): insert previous result into L1 load.
//       CHECK:   %[[I2:.+]] = tensor.insert_slice %[[I1]] into %[[L1_T]][0, 0] [1, 4] [1, 1] : tensor<4xi64> into tensor<2x4xi64>
// Iteration 3 peels outermost (L1 → POOL): insert previous into POOL load.
//       CHECK:   %[[I3:.+]] = tensor.insert_slice %[[I2]] into %[[POOL_T]][0, 0, 0] [1, 2, 4] [1, 1, 1] : tensor<2x4xi64> into tensor<2x2x4xi64>
// Final store targets the topmost binding with the fully-composed tensor.
//       CHECK:   iree_codegen.store_to_buffer %[[I3]], %[[POOL]] : tensor<2x2x4xi64> into memref<2x2x4xi64

// -----

// Verify two partial stores at DIFFERENT chain depths to the same binding.
// Sibling readers — `load_from_buffer %pool` and `load_from_buffer
// %deep_outer` — pre-seeded so the lift fires for both stores. After
// full lift: both stores target the topmost binding, AND they share the
// pre-seeded `load_from_buffer %pool` (no second load of pool is
// introduced — the reuse logic finds the existing one on every
// iteration). This is exactly what OneShot's RaW analysis needs to
// detect overlap across the two writes.
#pipeline_layout_mixed = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @lift_two_stores_different_chain_depth() {
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %pool = hal.interface.binding.subspan layout(#pipeline_layout_mixed) binding(0) alignment(64) offset(%c0) : memref<2x4xi64, #hal.descriptor_type<storage_buffer>>
  // Shallow chain: 1 level (subview of pool).
  %shallow = memref.subview %pool[1, 0] [1, 4] [1, 1] : memref<2x4xi64, #hal.descriptor_type<storage_buffer>> to memref<4xi64, strided<[1], offset: 4>, #hal.descriptor_type<storage_buffer>>
  // Deep chain: 2 levels (subview of subview of pool).
  %deep_outer = memref.subview %pool[0, 0] [1, 4] [1, 1] : memref<2x4xi64, #hal.descriptor_type<storage_buffer>> to memref<4xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>>
  %deep_inner = memref.subview %deep_outer[0] [1] [1] : memref<4xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>> to memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  // Sibling readers: one at the pool level (used by both stores' final
  // lift) and one at the deep_outer level (used by deep store's first
  // lift iteration).
  %loaded_pool = iree_codegen.load_from_buffer %pool : memref<2x4xi64, #hal.descriptor_type<storage_buffer>> -> tensor<2x4xi64>
  %loaded_deep_outer = iree_codegen.load_from_buffer %deep_outer : memref<4xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>> -> tensor<4xi64>
  %v_shallow = tensor.from_elements %c1_i64, %c1_i64, %c1_i64, %c1_i64 : tensor<4xi64>
  %v_deep = tensor.from_elements %c2_i64 : tensor<i64>
  iree_codegen.store_to_buffer %v_shallow, %shallow : tensor<4xi64> into memref<4xi64, strided<[1], offset: 4>, #hal.descriptor_type<storage_buffer>>
  iree_codegen.store_to_buffer %v_deep, %deep_inner : tensor<i64> into memref<i64, strided<[]>, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: @lift_two_stores_different_chain_depth
//       CHECK:   %[[POOL:.+]] = hal.interface.binding.subspan{{.+}}: memref<2x4xi64
// Both stores end up targeting the topmost binding (no subviews on the
// store target side). The pool tensor must be loaded EXACTLY ONCE in
// the function — the reuse logic finds the pre-seeded load on every
// iteration of the fixed-point loop.
//   CHECK-NOT:   iree_codegen.store_to_buffer {{.+}}, %{{[^,]*}}subview
//       CHECK:   iree_codegen.load_from_buffer %[[POOL]] : memref<2x4xi64{{.+}} -> tensor<2x4xi64>
//   CHECK-NOT:   iree_codegen.load_from_buffer %[[POOL]] : memref<2x4xi64{{.+}} -> tensor<2x4xi64>
// Two stores to %POOL — one for each tensor.insert_slice chain.
//       CHECK:   iree_codegen.store_to_buffer {{.+}}, %[[POOL]]
//       CHECK:   iree_codegen.store_to_buffer {{.+}}, %[[POOL]]

// -----

// Verify the same-block restriction in the reuse search. A parent-block
// load_from_buffer of the binding MUST NOT be reused for a partial store
// inside an scf.for body, otherwise the load gets effectively hoisted out
// of the loop and per-iteration accumulation is lost (each iteration
// would start from the original pool snapshot instead of the current
// pool). With Option A semantics: an in-body sibling reader is required
// for the lift to fire — the test pre-seeds one inside the loop and
// verifies the lift picks it (not the parent-block load).
#pipeline_layout_neg = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @lift_does_not_reuse_parent_block_load() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i64 = arith.constant 1 : i64
  %pool = hal.interface.binding.subspan layout(#pipeline_layout_neg) binding(0) alignment(64) offset(%c0) : memref<4xi64, #hal.descriptor_type<storage_buffer>>
  // Pre-existing full load in the function-level block — MUST NOT be
  // reused by any lift inside the loop body.
  %loaded_outer = iree_codegen.load_from_buffer %pool : memref<4xi64, #hal.descriptor_type<storage_buffer>> -> tensor<4xi64>
  scf.for %i = %c0 to %c4 step %c1 {
    // In-body sibling reader: this is the one the lift must pick up.
    %loaded_inner = iree_codegen.load_from_buffer %pool : memref<4xi64, #hal.descriptor_type<storage_buffer>> -> tensor<4xi64>
    %sub = memref.subview %pool[%i] [1] [1] : memref<4xi64, #hal.descriptor_type<storage_buffer>> to memref<1xi64, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
    %v = tensor.from_elements %c1_i64 : tensor<1xi64>
    iree_codegen.store_to_buffer %v, %sub : tensor<1xi64> into memref<1xi64, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  }
  // Use the parent-block load somewhere so it isn't DCE'd.
  iree_codegen.store_to_buffer %loaded_outer, %pool : tensor<4xi64> into memref<4xi64, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: @lift_does_not_reuse_parent_block_load
//       CHECK:   %[[POOL:.+]] = hal.interface.binding.subspan{{.+}}: memref<4xi64
// Parent-block load (pre-existing, unchanged by the pass).
//       CHECK:   iree_codegen.load_from_buffer %[[POOL]]
//       CHECK:   scf.for
// In-body load inside the loop body — this is the one the lift uses.
//       CHECK:     iree_codegen.load_from_buffer %[[POOL]]
//       CHECK:     tensor.insert_slice
//       CHECK:     iree_codegen.store_to_buffer {{.+}}, %[[POOL]]

// -----

// Negative test: a partial `store_to_buffer` to a subview parent that has
// NO in-block dominating `load_from_buffer` MUST NOT be lifted. The
// conditional restriction is load-bearing — lifting unconditionally
// would introduce a spurious load that forces OneShot to bufferize the
// store out-of-place, which the prototype version did and which broke
// SPIRV/Vulkan codegen (extra workgroup-local alloc that SPIRV could
// not lower). Keep this test if anyone is tempted to "simplify" the
// lift's reuse search into an unconditional create-load fallback.
#pipeline_layout_no_reader = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @no_lift_when_no_sibling_reader() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  %pool = hal.interface.binding.subspan layout(#pipeline_layout_no_reader) binding(0) alignment(64) offset(%c0) : memref<4xi64, #hal.descriptor_type<storage_buffer>>
  %sub = memref.subview %pool[0] [1] [1] : memref<4xi64, #hal.descriptor_type<storage_buffer>> to memref<1xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>>
  %v = tensor.from_elements %c1 : tensor<1xi64>
  iree_codegen.store_to_buffer %v, %sub : tensor<1xi64> into memref<1xi64, strided<[1]>, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: @no_lift_when_no_sibling_reader
//       CHECK:   %[[POOL:.+]] = hal.interface.binding.subspan{{.+}}: memref<4xi64
// No load_from_buffer of the parent — the lift skipped this store.
//   CHECK-NOT:   iree_codegen.load_from_buffer
// No tensor.insert_slice — the original subview-store form is preserved.
//   CHECK-NOT:   tensor.insert_slice
// The store still targets the original subview, not the topmost binding.
//       CHECK:   %[[SUB:.+]] = memref.subview %[[POOL]]
//       CHECK:   iree_codegen.store_to_buffer {{.+}}, %[[SUB]]
