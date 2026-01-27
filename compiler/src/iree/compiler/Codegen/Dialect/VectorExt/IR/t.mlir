// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

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

func.func @gather_fold_contigious_to_read(%source: memref<4096x64xf16>,
  %mask: vector<64xi1>)
  -> vector<64x64xf16> {
  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index

  %indices = vector.step : vector<64xindex>

  %out = iree_vector_ext.gather %source[%c0, %c0]
  [%indices: vector<64xindex>], %cst0, %mask
  { indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                     affine_map<(d0, d1)[s0] -> (d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>]}
  : memref<4096x64xf16>, vector<64x64xf16>, vector<64xi1>

  return %out : vector<64x64xf16>
}
