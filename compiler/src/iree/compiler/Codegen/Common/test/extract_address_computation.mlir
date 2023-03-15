// RUN: iree-opt -extract-address-computation %s --split-input-file | FileCheck %s
// RUN: iree-opt -extract-address-computation -expand-strided-metadata \
// RUN:   -loop-invariant-code-motion -decompose-affine-ops -loop-invariant-code-motion \
// RUN:   %s --split-input-file | FileCheck --check-prefix=INTEGRATION %s

// Simple test: check that we extract the address computation of a load into
// a dedicated subview.
// The resulting load will be loading from the subview and have only indices
// set to zero.

// CHECK-LABEL: @test(
// CHECK-SAME: %[[BASE:[^:]*]]: memref{{[^,]*}},
// CHECK-SAME: %[[DYN_OFFSET:.*]]: index)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET]], 0, 8] [1, 1, 1] [1, 1, 1] : memref<2x16x16xf32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: %[[LOADED_VAL:.*]] = memref.load %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: return %[[LOADED_VAL]] : f32

// For the integration test, check that the resulting address computation is:
// %offset * 16 * 16 + 0 * 16 + 8
// == %offset * 256 + 8

// INTEGRATION: #[[$C8_MAP:.*]] = affine_map<() -> (8)>
// INTEGRATION: #[[$x256_MAP:.*]] = affine_map<()[s0] -> (s0 * 256)>
// INTEGRATION: #[[$ADD_MAP:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// INTEGRATION-LABEL: func.func @test(
// INTEGRATION-SAME: %[[BASE:.*]]: memref<2x16x16xf32>,
// INTEGRATION-SAME: %[[OFF:.*]]: index) -> f32 {
// INTEGRATION:   %[[VAL_2:.*]] = arith.constant 0 : index
// INTEGRATION:   %[[MEMREF_BASE:.*]], %[[VAL_6:.*]], %[[VAL_5:.*]]:3, %[[VAL_6:.*]]:3 = memref.extract_strided_metadata %[[BASE]] : memref<2x16x16xf32> -> memref<f32>, index, index, index, index, index, index, index
// INTEGRATION:   %[[VAL_7:.*]] = affine.apply #[[$C8_MAP]]()
// INTEGRATION:   %[[VAL_8:.*]] = affine.apply #[[$x256_MAP]](){{\[}}%[[OFF]]]
// INTEGRATION:   %[[VAL_9:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[VAL_7]], %[[VAL_8]]]
// INTEGRATION:   %[[VAL_10:.*]] = memref.reinterpret_cast %[[MEMREF_BASE]] to offset: {{\[}}%[[VAL_9]]], sizes: [1, 1, 1], strides: [256, 16, 1] : memref<f32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// INTEGRATION:   %[[VAL_11:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// INTEGRATION:   return %[[VAL_11]] : f32
// INTEGRATION: }
func.func @test(%base : memref<2x16x16xf32>, %offset : index) -> f32 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %loaded_val = memref.load %base[%offset, %c0, %c8] : memref<2x16x16xf32>
  return %loaded_val : f32
}

// -----

// Simple test: check that we extract the address computation of a store into
// a dedicated subview.
// The resulting store will use the address from the subview and have only
// indices set to zero.

// CHECK-LABEL: @test_store(
// CHECK-SAME: %[[BASE:[^:]*]]: memref{{[^,]*}},
// CHECK-SAME: %[[DYN_OFFSET:.*]]: index)
// CHECK-DAG: %[[CF0:.*]] = arith.constant 0.0{{0*e\+00}} : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET]], 0, 8] [1, 1, 1] [1, 1, 1] : memref<2x16x16xf32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: memref.store %[[CF0]], %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// CHECK: return

// For the integration test, check that the resulting address computation is:
// %offset * 16 * 16 + 0 * 16 + 8
// == %offset * 256 + 8

// INTEGRATION: #[[$C8_MAP:.*]] = affine_map<() -> (8)>
// INTEGRATION: #[[$x256_MAP:.*]] = affine_map<()[s0] -> (s0 * 256)>
// INTEGRATION: #[[$ADD_MAP:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// INTEGRATION-LABEL: func.func @test_store(
// INTEGRATION-SAME: %[[BASE:.*]]: memref<2x16x16xf32>,
// INTEGRATION-SAME: %[[OFF:.*]]: index) {
// INTEGRATION-DAG:   %[[CF0:.*]] = arith.constant 0.0{{0*e\+00}} : f32
// INTEGRATION-DAG:   %[[VAL_2:.*]] = arith.constant 0 : index
// INTEGRATION:   %[[MEMREF_BASE:.*]], %[[VAL_6:.*]], %[[VAL_5:.*]]:3, %[[VAL_6:.*]]:3 = memref.extract_strided_metadata %[[BASE]] : memref<2x16x16xf32> -> memref<f32>, index, index, index, index, index, index, index
// INTEGRATION:   %[[VAL_7:.*]] = affine.apply #[[$C8_MAP]]()
// INTEGRATION:   %[[VAL_8:.*]] = affine.apply #[[$x256_MAP]](){{\[}}%[[OFF]]]
// INTEGRATION:   %[[VAL_9:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[VAL_7]], %[[VAL_8]]]
// INTEGRATION:   %[[VAL_10:.*]] = memref.reinterpret_cast %[[MEMREF_BASE]] to offset: {{\[}}%[[VAL_9]]], sizes: [1, 1, 1], strides: [256, 16, 1] : memref<f32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// INTEGRATION:   memref.store %[[CF0]], %[[VAL_10]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>>
// INTEGRATION:   return
// INTEGRATION: }
func.func @test_store(%base : memref<2x16x16xf32>, %offset : index) -> () {
  %cf0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  memref.store %cf0, %base[%offset, %c0, %c8] : memref<2x16x16xf32>
  return
}

// -----

// For this test, we made the source memref fully dynamic.
// The gist of the check remains the same as the simple test:
// The address computation is extracted into its own subview.
// The loops are used for the integration test.
// CHECK-LABEL: @testWithLoop(
// CHECK-SAME: %[[BASE:[^:]*]]: memref
// CHECK:  %[[SUM_ALL:.*]] = arith.constant 0.0{{0*e\+00}} : f32
// CHECK:  %[[C0:.*]] = arith.constant 0 : index
// CHECK:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[C2:.*]] = arith.constant 2 : index
// CHECK:  %[[UPPER_BOUND0:.*]] = memref.dim %[[BASE]], %[[C0]] : memref<?x?x?xf32,
// CHECK:  %[[UPPER_BOUND1:.*]] = memref.dim %[[BASE]], %[[C1]] : memref<?x?x?xf32,
// CHECK:  %[[UPPER_BOUND2:.*]] = memref.dim %[[BASE]], %[[C2]] : memref<?x?x?xf32,
// CHECK:  %[[SUM_RES2:.*]] = scf.for %[[IV2:.*]] = %[[C0]] to %[[UPPER_BOUND2]] step %[[C1]] iter_args(%[[SUM_ITER2:.*]] = %[[SUM_ALL]]) -> (f32) {
// CHECK:    %[[SUM_RES1:.*]] = scf.for %[[IV1:.*]] = %[[C0]] to %[[UPPER_BOUND1]] step %[[C1]] iter_args(%[[SUM_ITER1:.*]] = %[[SUM_ITER2]]) -> (f32) {
// CHECK:      %[[SUM_RES0:.*]] = scf.for %[[IV0:.*]] = %[[C0]] to %[[UPPER_BOUND0]] step %[[C1]] iter_args(%[[SUM_ITER0:.*]] = %[[SUM_ITER1]]) -> (f32) {
// CHECK:        %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[IV0]], %[[IV1]], %[[IV2]]] [1, 1, 1] [1, 1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:        %[[LOADED_VAL:.*]] = memref.load %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:        %[[RES:.*]] = arith.addf %[[LOADED_VAL]], %[[SUM_ITER2]] : f32
// CHECK:        scf.yield %[[RES]] : f32
// CHECK:      }
// CHECK:      scf.yield %[[SUM_RES0]] : f32
// CHECK:    }
// CHECK:    scf.yield %[[SUM_RES1]] : f32
// CHECK:  }
// CHECK:  return %[[SUM_RES2]] : f32

// For the integration test, check that the address computation is broken down
// in one computation per IV and the related subexpression is hoisted in the proper
// loop (dim2 at the start of the loop with iv2, dim1 with iv1, etc.)
//
// Note: the scf.for are purposely flipped (dim2 -> dim0 instead of dim0 -> dim2) to
// make the ordering from the decompose of affine ops more obvious.
// INTEGRATION: #[[$SINGLE_VALUE_MAP:.*]] = affine_map<()[s0] -> (s0)>
// INTEGRATION: #[[$MUL_MAP:.*]] = affine_map<()[s0, s1] -> (s1 * s0)>
// INTEGRATION: #[[$ADD_MAP:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// INTEGRATION-LABEL: func.func @testWithLoop(
// INTEGRATION-SAME: %[[BASE:.*]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> f32 {
// INTEGRATION:  %[[SUM_ALL:.*]] = arith.constant 0.000000e+00 : f32
// INTEGRATION:  %[[C0:.*]] = arith.constant 0 : index
// INTEGRATION:  %[[C1:.*]] = arith.constant 1 : index
// INTEGRATION:  %[[C2:.*]] = arith.constant 2 : index
// INTEGRATION:  %[[UPPER_BOUND0:.*]] = memref.dim %[[BASE]], %[[C0]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// INTEGRATION:  %[[UPPER_BOUND1:.*]] = memref.dim %[[BASE]], %[[C1]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// INTEGRATION:  %[[UPPER_BOUND2:.*]] = memref.dim %[[BASE]], %[[C2]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// INTEGRATION:  %[[MEMREF_BASE:.*]], %[[MEMREF_OFF:.*]], %[[MEMREF_SIZES:.*]]:3, %[[MEMREF_STRIDES:.*]]:3 = memref.extract_strided_metadata %[[BASE]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
// INTEGRATION:  %[[OFF:.*]] = affine.apply #[[$SINGLE_VALUE_MAP]](){{\[}}%[[MEMREF_OFF]]]
// INTEGRATION:  %[[SUM_RES2:.*]] = scf.for %[[IV2:.*]] = %[[C0]] to %[[UPPER_BOUND2]] step %[[C1]] iter_args(%[[SUM_ITER2:.*]] = %[[SUM_ALL]]) -> (f32) {
// INTEGRATION:    %[[STRIDE2:.*]] = affine.apply #[[$MUL_MAP]](){{\[}}%[[MEMREF_STRIDES]]#2, %[[IV2]]]
// INTEGRATION:    %[[OFF2:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[OFF]], %[[STRIDE2]]]
// INTEGRATION:    %[[SUM_RES1:.*]] = scf.for %[[IV1:.*]] = %[[C0]] to %[[UPPER_BOUND1]] step %[[C1]] iter_args(%[[SUM_ITER1:.*]] = %[[SUM_ITER2]]) -> (f32) {
// INTEGRATION:      %[[STRIDE1:.*]] = affine.apply #[[$MUL_MAP]](){{\[}}%[[MEMREF_STRIDES]]#1, %[[IV1]]]
// INTEGRATION:      %[[OFF1:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[OFF2]], %[[STRIDE1]]]
// INTEGRATION:      %[[SUM_RES0:.*]] = scf.for %[[IV0:.*]] = %[[C0]] to %[[UPPER_BOUND0]] step %[[C1]] iter_args(%[[SUM_ITER0:.*]] = %[[SUM_ITER1]]) -> (f32) {
// INTEGRATION:        %[[STRIDE0:.*]] = affine.apply #[[$MUL_MAP]](){{\[}}%[[MEMREF_STRIDES]]#0, %[[IV0]]]
// INTEGRATION:        %[[FINAL_OFF:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[OFF1]], %[[STRIDE0]]]
// INTEGRATION:        %[[SUBVIEW:.*]] = memref.reinterpret_cast %[[MEMREF_BASE]] to offset: {{\[}}%[[FINAL_OFF]]], sizes: [1, 1, 1], strides: {{\[}}%[[MEMREF_STRIDES]]#0, %[[MEMREF_STRIDES]]#1, %[[MEMREF_STRIDES]]#2] : memref<f32> to memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// INTEGRATION:        %[[LOADED_VAL:.*]] = memref.load %[[SUBVIEW]]{{\[}}%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// INTEGRATION:        %[[RES:.*]] = arith.addf %[[LOADED_VAL]], %[[SUM_ITER2]] : f32
// INTEGRATION:        scf.yield %[[RES]] : f32
// INTEGRATION:      }
// INTEGRATION:      scf.yield %[[SUM_RES0]] : f32
// INTEGRATION:    }
// INTEGRATION:    scf.yield %[[SUM_RES1]] : f32
// INTEGRATION:  }
// INTEGRATION:  return %[[SUM_RES2]] : f32
func.func @testWithLoop(%base : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>) -> f32 {
  %sum_all = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %upper_bound0 = memref.dim %base, %c0 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %upper_bound1 = memref.dim %base, %c1 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %upper_bound2 = memref.dim %base, %c2 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %sum_res2 = scf.for %iv2 = %c0 to %upper_bound2 step %c1 iter_args(%sum_iter2 = %sum_all) -> (f32) {
    %sum_res1 = scf.for %iv1 = %c0 to %upper_bound1 step %c1 iter_args(%sum_iter1 = %sum_iter2) -> (f32) {
      %sum_res0 = scf.for %iv0 = %c0 to %upper_bound0 step %c1 iter_args(%sum_iter0 = %sum_iter1) -> (f32) {
        %loaded_val = memref.load %base[%iv0, %iv1, %iv2] : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
        %res = arith.addf %loaded_val, %sum_iter2 : f32
        scf.yield %res : f32
      }
      scf.yield %sum_res0 : f32
    }
    scf.yield %sum_res1 : f32
  }
  return %sum_res2 : f32
}
