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

// CHECK-LABEL: @transfer_gather_fold_single_element
// CHECK-SAME: %{{.*}}: vector<1xindex>, %[[ARG1:.*]]: vector<64x1xindex>
// CHECK: transfer_gather
// CHECK-SAME: [None, %[[ARG1]]

// -----

func.func @transfer_gather_fold_contiguous_load(%scalar: vector<64x1xindex>,
  %indices: vector<64x1xindex>,
  %source: tensor<4096x64xf16>)
  -> vector<64x1xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [None, None], %cst0 {indexed_maps = []} : tensor<4096x64xf16>, vector<64x1xf16>

  return %out : vector<64x1xf16>
}

// CHECK-LABEL: @transfer_gather_fold_contiguous_load
// CHECK: vector.transfer_read
// CHECK-NOT: transfer_gather

// -----

//===----------------------------------------------------------------------===//
// GatherOp fold tests
//===----------------------------------------------------------------------===//

func.func @gather_fold_broadcast(%indices: vector<64xindex>,
  %source: memref<4096x64xf16>)
  -> vector<64x32xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %broadcasted = vector.broadcast %indices : vector<64xindex> to vector<32x64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%broadcasted: vector<32x64xindex>], %cst0
  { indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                     affine_map<(d0, d1)[s0] -> (d1, d0)>]}
  : memref<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0)>
// CHECK-LABEL: @gather_fold_broadcast
// CHECK-NOT: vector.broadcast
// CHECK: gather
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]]]

// -----

func.func @gather_fold_transpose(%indices: vector<64x32xindex>,
  %source: memref<4096x64xf16>)
  -> vector<64x32xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %transposed = vector.transpose %indices, [1, 0] : vector<64x32xindex> to vector<32x64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%transposed: vector<32x64xindex>], %cst0
  {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                    affine_map<(d0, d1)[s0] -> (d1, d0)>]}
  : memref<4096x64xf16>, vector<64x32xf16>


  return %out : vector<64x32xf16>
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0, d1)>
// CHECK-LABEL: @gather_fold_transpose
// CHECK-NOT: vector.transpose
// CHECK: gather
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]]]

// -----

func.func @gather_fold_step(%indices: vector<64x32xindex>,
  %source: memref<4096x64xf16>)
  -> vector<64x32xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %step = vector.step : vector<64xindex>
  %expanded = vector.broadcast %step : vector<64xindex> to vector<32x64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%expanded, %indices : vector<32x64xindex>, vector<64x32xindex>], %cst0
  {indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s1, s0)>,
                    affine_map<(d0, d1)[s0, s1] -> (d1, d0)>,
                    affine_map<(d0, d1)[s0, s1] -> (d0, d1)>]}
  : memref<4096x64xf16>, vector<64x32xf16>

  return %out : vector<64x32xf16>
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0, d1)>
// CHECK-LABEL: @gather_fold_step
// CHECK-NOT: vector.step
// CHECK-NOT: vector.broadcast
// CHECK: gather
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]]]

// -----

func.func @gather_fold_mask_broadcast(%indices: vector<64xindex>,
  %source: memref<4096x64xf16>,
  %mask: vector<64xi1>)
  -> vector<64x32xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %mask_broadcasted = vector.broadcast %mask : vector<64xi1> to vector<32x64xi1>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%indices: vector<64xindex>], %cst0, %mask_broadcasted
  { indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                     affine_map<(d0, d1)[s0] -> (d0)>,
                     affine_map<(d0, d1)[s0] -> (d1, d0)>]}
  : memref<4096x64xf16>, vector<64x32xf16>, vector<32x64xi1>

  return %out : vector<64x32xf16>
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0)>
// CHECK-LABEL: @gather_fold_mask_broadcast
// CHECK-NOT: vector.broadcast
// CHECK: gather
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]]]

// -----

func.func @gather_fold_single_element(%indices: vector<1xindex>,
  %indices2: vector<32xindex>,
  %source: memref<4096x64xf16>)
  -> vector<1x32xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%indices, %indices2 : vector<1xindex>, vector<32xindex>], %cst0
  { indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                     affine_map<(d0, d1)[s0, s1] -> (d0)>,
                     affine_map<(d0, d1)[s0, s1] -> (d1)>]}
  : memref<4096x64xf16>, vector<1x32xf16>

  return %out : vector<1x32xf16>
}

// CHECK-LABEL: @gather_fold_single_element
// CHECK-SAME: %[[INDICES:.*]]: vector<1xindex>, %[[INDICES2:.*]]: vector<32xindex>, %[[SOURCE:.*]]: memref<4096x64xf16>
// CHECK: %[[VEC:.+]] = vector.extract %{{.+}}[0] : index from vector<1xindex>
// CHECK: iree_vector_ext.gather %[[SOURCE]][%[[VEC]], %c0] [%[[INDICES2]] : vector<32xindex>]

// -----

func.func @gather_fold_contiguous_with_identity_mask(
  %source: memref<64x64xf16>,
  %mask: vector<64x64xi1>)
  -> vector<64x64xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %indices = vector.step : vector<64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%indices: vector<64xindex>], %cst0, %mask
  {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                    affine_map<(d0, d1)[s0] -> (d1)>,
                    affine_map<(d0, d1)[s0] -> (d0, d1)>]}
  : memref<64x64xf16>, vector<64x64xf16>, vector<64x64xi1>

  return %out : vector<64x64xf16>
}

// CHECK-LABEL: @gather_fold_contiguous_with_identity_mask
// CHECK-SAME: %[[SOURCE:.*]]: memref<64x64xf16>, %[[MASK:.*]]: vector<64x64xi1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NOT: vector.broadcast
// CHECK-NOT: vector.transpose
// CHECK: %[[READ:.*]] = vector.transfer_read %[[SOURCE]][%[[C0]], %[[C0]]], %[[PAD]], %[[MASK]]
// CHECK-SAME: {in_bounds = [true, true]} : memref<64x64xf16>, vector<64x64xf16>
// CHECK: return %[[READ]]

// -----

func.func @gather_fold_contiguous_with_transposed_mask(
  %source: memref<64x64xf16>,
  %mask: vector<64x64xi1>)
  -> vector<64x64xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %indices = vector.step : vector<64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%indices: vector<64xindex>], %cst0, %mask
  {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                    affine_map<(d0, d1)[s0] -> (d1)>,
                    affine_map<(d0, d1)[s0] -> (d1, d0)>]}
  : memref<64x64xf16>, vector<64x64xf16>, vector<64x64xi1>

  return %out : vector<64x64xf16>
}

// CHECK-LABEL: @gather_fold_contiguous_with_transposed_mask
// CHECK-SAME: %[[SOURCE:.*]]: memref<64x64xf16>, %[[MASK:.*]]: vector<64x64xi1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NOT: vector.broadcast
// CHECK: %[[TRANSPOSE:.*]] = vector.transpose %[[MASK]], [1, 0] : vector<64x64xi1> to vector<64x64xi1>
// CHECK: %[[READ:.*]] = vector.transfer_read %[[SOURCE]][%[[C0]], %[[C0]]], %[[PAD]], %[[TRANSPOSE]]
// CHECK-SAME: {in_bounds = [true, true]} : memref<64x64xf16>, vector<64x64xf16>
// CHECK: return %[[READ]]

// -----

func.func @gather_fold_contiguous_mask_broadcast_major_dim(
  %source: memref<4096x64xf16>,
  %mask: vector<64xi1>)
  -> vector<64x64xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %indices = vector.step : vector<64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%indices: vector<64xindex>], %cst0, %mask
  {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                    affine_map<(d0, d1)[s0] -> (d1)>,
                    affine_map<(d0, d1)[s0] -> (d0)>]}
  : memref<4096x64xf16>, vector<64x64xf16>, vector<64xi1>

  return %out : vector<64x64xf16>
}

// CHECK-LABEL: @gather_fold_contiguous_mask_broadcast_major_dim
// CHECK-SAME: %[[SOURCE:.*]]: memref<4096x64xf16>, %[[MASK:.*]]: vector<64xi1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[MASK]] : vector<64xi1> to vector<64x64xi1>
// CHECK: %[[TRANSPOSE:.*]] = vector.transpose %[[BCAST]], [1, 0] : vector<64x64xi1> to vector<64x64xi1>
// CHECK: %[[READ:.*]] = vector.transfer_read %[[SOURCE]][%[[C0]], %[[C0]]], %[[PAD]], %[[TRANSPOSE]]
// CHECK-SAME: {in_bounds = [true, true]} : memref<4096x64xf16>, vector<64x64xf16>
// CHECK: return %[[READ]]

// -----

func.func @gather_fold_contiguous_mask_broadcast_minor_dim(
  %source: memref<64x4096xf16>,
  %mask: vector<64xi1>)
  -> vector<64x64xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %indices = vector.step : vector<64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%indices: vector<64xindex>], %cst0, %mask
  {indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                    affine_map<(d0, d1)[s0] -> (d0)>,
                    affine_map<(d0, d1)[s0] -> (d1)>]}
  : memref<64x4096xf16>, vector<64x64xf16>, vector<64xi1>

  return %out : vector<64x64xf16>
}

// CHECK-LABEL: @gather_fold_contiguous_mask_broadcast_minor_dim
// CHECK-SAME: %[[SOURCE:.*]]: memref<64x4096xf16>, %[[MASK:.*]]: vector<64xi1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[MASK]] : vector<64xi1> to vector<64x64xi1>
// CHECK-NOT: vector.transpose
// CHECK: %[[READ:.*]] = vector.transfer_read %[[SOURCE]][%[[C0]], %[[C0]]], %[[PAD]], %[[BCAST]]
// CHECK-SAME: {in_bounds = [true, true]} : memref<64x4096xf16>, vector<64x64xf16>
// CHECK: return %[[READ]]

// -----

//===----------------------------------------------------------------------===//
// ScatterOp fold tests
//===----------------------------------------------------------------------===//

func.func @scatter_fold_broadcast(%indices: vector<64xindex>,
  %source: memref<4096x64xf16>, %value: vector<64x32xf16>) {

  %c0 = arith.constant 0 : index

  %broadcasted = vector.broadcast %indices : vector<64xindex> to vector<32x64xindex>

  iree_vector_ext.scatter %value, %source[%c0, %c0]
  [%broadcasted: vector<32x64xindex>]
  {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                    affine_map<(d0, d1)[s0] -> (d1, d0)>]}
  : memref<4096x64xf16>, vector<64x32xf16>

  return
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0)>
// CHECK-LABEL: @scatter_fold_broadcast
// CHECK-NOT: vector.broadcast
// CHECK: scatter
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]]]

// -----

func.func @scatter_fold_transpose(%indices: vector<64x32xindex>,
  %source: memref<4096x64xf16>, %value: vector<64x32xf16>) {

  %c0 = arith.constant 0 : index

  %transposed = vector.transpose %indices, [1, 0] : vector<64x32xindex> to vector<32x64xindex>

  iree_vector_ext.scatter %value, %source[%c0, %c0]
  [%transposed: vector<32x64xindex>]
  {indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                    affine_map<(d0, d1)[s0] -> (d1, d0)>]}
  : memref<4096x64xf16>, vector<64x32xf16>

  return
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0, d1)>
// CHECK-LABEL: @scatter_fold_transpose
// CHECK-NOT: vector.transpose
// CHECK: scatter
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]]]

// -----

func.func @scatter_fold_step(%indices: vector<64x32xindex>,
  %source: memref<4096x64xf16>, %value: vector<64x32xf16>) {

  %c0 = arith.constant 0 : index

  %step = vector.step : vector<64xindex>
  %expanded = vector.broadcast %step : vector<64xindex> to vector<32x64xindex>

  iree_vector_ext.scatter %value, %source[%c0, %c0]
  [%expanded, %indices : vector<32x64xindex>, vector<64x32xindex>]
  {indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (s1, s0)>,
                    affine_map<(d0, d1)[s0, s1] -> (d1, d0)>,
                    affine_map<(d0, d1)[s0, s1] -> (d0, d1)>]}
  : memref<4096x64xf16>, vector<64x32xf16>

  return
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0, d1)>
// CHECK-LABEL: @scatter_fold_step
// CHECK-NOT: vector.step
// CHECK-NOT: vector.broadcast
// CHECK: scatter
// CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP1]]]

// -----

//===----------------------------------------------------------------------===//
// ScatterOp -> transfer_write canonicalization tests
//===----------------------------------------------------------------------===//

func.func @scatter_fold_contiguous_to_write(
  %source: memref<64x64xf16>,
  %value: vector<64x64xf16>) {

  %c0 = arith.constant 0 : index

  iree_vector_ext.scatter %value, %source[%c0, %c0]
  {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>]}
  : memref<64x64xf16>, vector<64x64xf16>

  return
}

// CHECK-LABEL: @scatter_fold_contiguous_to_write
// CHECK-SAME: %[[SOURCE:.*]]: memref<64x64xf16>, %[[VALUE:.*]]: vector<64x64xf16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NOT: iree_vector_ext.scatter
// CHECK: vector.transfer_write %[[VALUE]], %[[SOURCE]][%[[C0]], %[[C0]]]
// CHECK-SAME: {in_bounds = [true, true]} : vector<64x64xf16>, memref<64x64xf16>
// CHECK: return

// -----

func.func @scatter_fold_contiguous_with_mask(
  %source: memref<64x64xf16>,
  %value: vector<64x64xf16>,
  %mask: vector<64x64xi1>) {

  %c0 = arith.constant 0 : index

  iree_vector_ext.scatter %value, %source[%c0, %c0], %mask
  {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                    affine_map<(d0, d1) -> (d0, d1)>]}
  : memref<64x64xf16>, vector<64x64xf16>, vector<64x64xi1>

  return
}

// CHECK-LABEL: @scatter_fold_contiguous_with_mask
// CHECK-SAME: %[[SOURCE:.*]]: memref<64x64xf16>, %[[VALUE:.*]]: vector<64x64xf16>, %[[MASK:.*]]: vector<64x64xi1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NOT: iree_vector_ext.scatter
// CHECK: vector.transfer_write %[[VALUE]], %[[SOURCE]][%[[C0]], %[[C0]]], %[[MASK]]
// CHECK-SAME: {in_bounds = [true, true]} : vector<64x64xf16>, memref<64x64xf16>
// CHECK: return

// -----

func.func @scatter_fold_contiguous_with_transposed_mask(
  %source: memref<64x64xf16>,
  %value: vector<64x64xf16>,
  %mask: vector<64x64xi1>) {

  %c0 = arith.constant 0 : index

  iree_vector_ext.scatter %value, %source[%c0, %c0], %mask
  {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                    affine_map<(d0, d1) -> (d1, d0)>]}
  : memref<64x64xf16>, vector<64x64xf16>, vector<64x64xi1>

  return
}

// CHECK-LABEL: @scatter_fold_contiguous_with_transposed_mask
// CHECK-SAME: %[[SOURCE:.*]]: memref<64x64xf16>, %[[VALUE:.*]]: vector<64x64xf16>, %[[MASK:.*]]: vector<64x64xi1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[TRANSPOSE:.*]] = vector.transpose %[[MASK]], [1, 0]
// CHECK-NOT: iree_vector_ext.scatter
// CHECK: vector.transfer_write %[[VALUE]], %[[SOURCE]][%[[C0]], %[[C0]]], %[[TRANSPOSE]]
// CHECK-SAME: {in_bounds = [true, true]} : vector<64x64xf16>, memref<64x64xf16>
// CHECK: return
