// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @to_simt_to_simd_fold
// CHECK-SAME: (%[[SIMD:.*]]: vector<64x64xf32>) -> vector<64x64xf32>
func.func @to_simt_to_simd_fold(%simd: vector<64x64xf32>) -> vector<64x64xf32> {
  // Both to_simt and to_simd should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simt = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simd_out = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // CHECK: return %[[SIMD]]
  func.return %simd_out : vector<64x64xf32>
}

// -----

// CHECK-LABEL: @to_simd_to_simt_fold
// CHECK-SAME: (%[[SIMT:.*]]: vector<4x4x4xf32>) -> vector<4x4x4xf32>
func.func @to_simd_to_simt_fold(%simt: vector<4x4x4xf32>) -> vector<4x4x4xf32> {
  // Both to_simt and to_simd should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simt_out = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  // CHECK: return %[[SIMT]]
  func.return %simt_out : vector<4x4x4xf32>
}

// -----

// CHECK-LABEL: @to_simd_to_simt_multi_use
// CHECK-SAME: (%[[SIMT:.*]]: vector<4x4x4xf32>)
func.func @to_simd_to_simt_multi_use(%simt: vector<4x4x4xf32>) -> (vector<4x4x4xf16>, vector<64x64xf32>) {
  // The to_simd operation should not be dce-ed after folding because it is returned.
  // CHECK: %[[SIMD:.*]] = iree_vector_ext.to_simd %[[SIMT]] : vector<4x4x4xf32> -> vector<64x64xf32>
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // The to_simt operation should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simt_out = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>

  // Check if the folding happened correctly.
  // CHECK: %[[TRUNCED:.*]] = arith.truncf %[[SIMT]]
  %trunced = arith.truncf %simt_out : vector<4x4x4xf32> to vector<4x4x4xf16>

  // CHECK: return %[[TRUNCED]], %[[SIMD]]
  func.return %trunced, %simd : vector<4x4x4xf16>, vector<64x64xf32>
}

// -----

func.func @transfer_gather_fold_broadcast(%indices: vector<64xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %broadcasted = vector.broadcast %indices : vector<64xindex> to vector<32x64xindex>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [None, %broadcasted: vector<32x64xindex>], %cst0
  { indexed_maps = [affine_map<(d0, d1) -> (d1, d0)>]}
  : tensor<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @transfer_gather_fold_broadcast
// CHECK: transfer_gather
// CHECK-SAME: indexed_maps = [#[[$MAP]]]

// -----

func.func @transfer_gather_fold_transpose(%indices: vector<64x32xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %transposed = vector.transpose %indices, [1, 0] : vector<64x32xindex> to vector<32x64xindex>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [None, %transposed: vector<32x64xindex>], %cst0
  {indexed_maps = [affine_map<(d0, d1) -> (d1, d0)>]}
  : tensor<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @transfer_gather_fold_transpose
// CHECK: transfer_gather
// CHECK-SAME: indexed_maps = [#[[$MAP]]]

// -----

func.func @transfer_gather_fold_step(%indices: vector<64x32xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x32xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %step = vector.step : vector<64xindex>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%step : vector<64xindex>, %indices: vector<64x32xindex>], %cst0
  {indexed_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)> ]}
  : tensor<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK-LABEL: @transfer_gather_fold_step
// CHECK-SAME: %[[ARG1:.*]]: vector<64x32xindex>
// CHECK: transfer_gather
// CHECK-SAME: [None, %[[ARG1]]

// -----

func.func @transfer_gather_fold_single_element(%scalar: vector<1xindex>,
  %indices: vector<64x1xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x1xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%scalar : vector<1xindex>, %indices: vector<64x1xindex>], %cst0
  {indexed_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)> ]}
  : tensor<4096x64xf16>, vector<64x1xf16>

  return %out : vector<64x1xf16>
}

// CHECK-LABEL: transfer_gather_fold_single_element
// CHECK-SAME: %{{.*}}: vector<1xindex>, %[[ARG1:.*]]: vector<64x1xindex>
// CHECK: transfer_gather
// CHECK-SAME: [None, %[[ARG1]]

// -----

func.func @transfer_gather_fold_contigious_load(%scalar: vector<64x1xindex>,
  %indices: vector<64x1xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x1xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [None, None], %cst0 {indexed_maps = []} : tensor<4096x64xf16>, vector<64x1xf16>

  return %out : vector<64x1xf16>
}

// CHECK-LABEL: @transfer_gather_fold_contigious_load
// CHECK: vector.transfer_read
// CHECK-NOT: transfer_gather

// -----
