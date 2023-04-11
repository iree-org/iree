// RUN: iree-opt -extract-address-computation-gpu %s --split-input-file | FileCheck %s

// Simple test: check that we extract the address computation of a ldmatrix into
// a dedicated subview.
// The resulting ldmatrix will loaded from with subview and have only indices set
// to zero.
// Also the sizes of the view are adjusted to `original size - offset`.

// CHECK-DAG: #[[$FOUR_MINUS_OFF_MAP:.*]] = affine_map<()[s0] -> (-s0 + 4)>
// CHECK-DAG: #[[$THIRTY_TWO_MINUS_OFF_MAP:.*]] = affine_map<()[s0] -> (-s0 + 32)>
// CHECK-LABEL: @test_ldmatrix(
// CHECK-SAME: %[[BASE:[^:]*]]: memref<{{[^,]*}}, 3>,
// CHECK-SAME: %[[DYN_OFFSET0:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET1:[^:]*]]: index,
// CHECK-SAME: %[[DYN_OFFSET2:[^:]*]]: index)
// CHECK-DAG: %[[DYN_SIZE0:.*]] = affine.apply #[[$FOUR_MINUS_OFF_MAP]]()[%[[DYN_OFFSET0]]]
// CHECK-DAG: %[[DYN_SIZE1:.*]] = affine.apply #[[$THIRTY_TWO_MINUS_OFF_MAP]]()[%[[DYN_OFFSET1]]]
// CHECK-DAG: %[[DYN_SIZE2:.*]] = affine.apply #[[$THIRTY_TWO_MINUS_OFF_MAP]]()[%[[DYN_OFFSET2]]]
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET0]], %[[DYN_OFFSET1]], %[[DYN_OFFSET2]]] [%[[DYN_SIZE0]], %[[DYN_SIZE1]], %[[DYN_SIZE2]]] [1, 1, 1] : memref<4x32x32xf16, 3> to memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3>
// CHECK: %[[LOADED_VAL:.*]] = nvgpu.ldmatrix %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {numTiles = 4 : i32, transpose = false} : memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3> -> vector<4x2xf16>
// CHECK: return %[[LOADED_VAL]] : vector<4x2xf16>

func.func @test_ldmatrix(%base : memref<4x32x32xf16, 3>,
    %offset0 : index, %offset1: index, %offset2: index)
    -> vector<4x2xf16> {
  %loaded_val = nvgpu.ldmatrix
    %base[%offset0, %offset1, %offset2]
    {numTiles = 4 : i32, transpose = false}
      : memref<4x32x32xf16, 3> -> vector<4x2xf16>
  return %loaded_val : vector<4x2xf16>
}
