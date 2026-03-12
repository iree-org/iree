// RUN: iree-opt -pass-pipeline="builtin.module(any(iree-codegen-test-vector-layout-analysis))" --split-input-file %s --verify-diagnostics

// Basic propagation

#layout = #iree_map.pack_layout<((4, 32)) : ((0, 1))>

// 1D layout propagates through elementwise ops.
func.func @propagate_elementwise_1d(%arg0: memref<128xf16>, %b: vector<128xf16>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 32)) : ((0, 1))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<128xf16>
  %c = arith.mulf %rootl, %b : vector<128xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 32)) : ((0, 1))>}}
  %d = arith.addf %c, %b : vector<128xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 32)) : ((0, 1))>}}
  func.return %d : vector<128xf16>
}

// -----

#layout = #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>

// 2D layout propagates through elementwise ops.
func.func @propagate_elementwise_2d(%arg0: memref<8x32xf16>, %b: vector<8x32xf16>) -> vector<8x32xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf16>, vector<8x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<8x32xf16>
  %c = arith.mulf %rootl, %b : vector<8x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>}}
  %d = arith.addf %c, %b : vector<8x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>}}
  func.return %d : vector<8x32xf16>
}

// -----

// Transpose (permute)

#layout = #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>

// 2D transpose swaps both modes.
func.func @transpose_2d(%arg0: memref<8x32xf16>) -> vector<32x8xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf16>, vector<8x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<8x32xf16>
  %t = vector.transpose %rootl, [1, 0] : vector<8x32xf16> to vector<32x8xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 8), (2, 4)) : ((0, 4), (0, 1))>}}
  func.return %t : vector<32x8xf16>
}

// -----

#layout = #iree_map.pack_layout<((2, 4), (4, 8), 2) : ((0, 1), (0, 4), 0)>

// 3D transpose [1, 2, 0] permutes all three dimensions.
func.func @transpose_3d(%arg0: memref<8x32x2xf16>) -> vector<32x2x8xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<8x32x2xf16>, vector<8x32x2xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), (4, 8), 2) : ((0, 1), (0, 4), 0)>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<8x32x2xf16>
  %t = vector.transpose %rootl, [1, 2, 0] : vector<8x32x2xf16> to vector<32x2x8xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 8), 2, (2, 4)) : ((0, 4), 0, (0, 1))>}}
  func.return %t : vector<32x2x8xf16>
}

// -----

#layout = #iree_map.pack_layout<(4, (4, 8), 2) : (1, (0, 4), 32)>

// 3D transpose [2, 0, 1] with broadcast dims.
func.func @transpose_3d_with_broadcast(%arg0: memref<4x32x2xf16>) -> vector<2x4x32xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<4x32x2xf16>, vector<4x32x2xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<(4, (4, 8), 2) : (1, (0, 4), 32)>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<4x32x2xf16>
  %t = vector.transpose %rootl, [2, 0, 1] : vector<4x32x2xf16> to vector<2x4x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<(2, 4, (4, 8)) : (32, 1, (0, 4))>}}
  func.return %t : vector<2x4x32xf16>
}

// -----

// Shape cast — expand (1 dim → many dims)

#layout = #iree_map.pack_layout<((4, 32)) : ((0, 1))>

// Expand 1D to 2D: 128 → 4x32.
func.func @reshape_expand_1d_to_2d(%arg0: memref<128xf16>) -> vector<4x32xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 32)) : ((0, 1))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<128xf16>
  %reshape = vector.shape_cast %rootl : vector<128xf16> to vector<4x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<(4, 32) : (0, 1)>}}
  func.return %reshape : vector<4x32xf16>
}

// -----

#layout = #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>

// Expand second dim: 8x32 → 8x4x8.
func.func @reshape_expand_second_dim(%arg0: memref<8x32xf16>) -> vector<8x4x8xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf16>, vector<8x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<8x32xf16>
  %reshape = vector.shape_cast %rootl : vector<8x32xf16> to vector<8x4x8xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), 4, 8) : ((0, 1), 0, 4)>}}
  func.return %reshape : vector<8x4x8xf16>
}

// -----

// Shape cast — contract (many dims → 1 dim)

#layout = #iree_map.pack_layout<(4, (4, 8)) : (1, (0, 4))>

// Contract 2D to 1D: 4x32 → 128.
func.func @reshape_contract_2d_to_1d(%arg0: memref<4x32xf16>) -> vector<128xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x32xf16>, vector<4x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<(4, (4, 8)) : (1, (0, 4))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<4x32xf16>
  %reshape = vector.shape_cast %rootl : vector<4x32xf16> to vector<128xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 4, 8)) : ((1, 0, 4))>}}
  func.return %reshape : vector<128xf16>
}

// -----

// Shape cast with hierarchical modes

#layout = #iree_map.pack_layout<((8, 2, 2, 4)) : ((1, 0, 0, 8))>

// Expand 1D to 2D with hierarchical modes.
// Note: adjacent stride-0 leaves (2,2):(0,0) get coalesced to 4:0 during
// flatten+coalesce in the 1:1 reshape path.
func.func @reshape_expand_hierarchical(%arg0: memref<128xf16>) -> vector<4x32xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((8, 4, 4)) : ((1, 0, 8))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<128xf16>
  %reshape = vector.shape_cast %rootl : vector<128xf16> to vector<4x32xf16>
  // The outermost 8 is being split into (4, 2) : (2:1), the 2 innermost dim
  // goes to the 32 in the 4x32 reshaped dimension, which is why the outermost
  // 4 has a stride of 2.
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<(4, (2, 4, 4)) : (2, (1, 0, 8))>}}
  func.return %reshape : vector<4x32xf16>
}

// -----

// Transpose followed by shape_cast (combined)

#layout = #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>

// Transpose then reshape: 8x32 → 32x8 → 256.
func.func @transpose_then_contract(%arg0: memref<8x32xf16>) -> vector<256xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf16>, vector<8x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((2, 4), (4, 8)) : ((0, 1), (0, 4))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<8x32xf16>
  %t = vector.transpose %rootl, [1, 0] : vector<8x32xf16> to vector<32x8xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 8), (2, 4)) : ((0, 4), (0, 1))>}}
  %reshape = vector.shape_cast %t : vector<32x8xf16> to vector<256xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 8, 2, 4)) : ((0, 4, 0, 1))>}}
  func.return %reshape : vector<256xf16>
}

// -----

#layout = #iree_map.pack_layout<((4, 32)) : ((0, 1))>

// Expand then transpose: 128 → 4x32 → 32x4.
func.func @expand_then_transpose(%arg0: memref<128xf16>) -> vector<32x4xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<((4, 32)) : ((0, 1))>}}
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<128xf16>
  %reshape = vector.shape_cast %rootl : vector<128xf16> to vector<4x32xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<(4, 32) : (0, 1)>}}
  %t = vector.transpose %reshape, [1, 0] : vector<4x32xf16> to vector<32x4xf16>
  // expected-remark @above {{layout of result #0 is #iree_map.pack_layout<(32, 4) : (1, 0)>}}
  func.return %t : vector<32x4xf16>
}
