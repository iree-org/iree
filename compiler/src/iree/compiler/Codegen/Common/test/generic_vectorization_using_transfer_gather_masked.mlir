// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true vectorize-to-transfer-gather=true}))" --split-input-file %s | FileCheck %s

// Tests for vectorization of gather-like linalg.generic operations into
// iree_vector_ext.transfer_gather ops, using the vectorize-to-transfer-gather
// option.

!storage = tensor<8192x8xf16>
!ind     = tensor<128xi64>
!x       = tensor<128x8xf16>

#gather = {
    indexing_maps = [affine_map<(page, vec) -> (page)>,
                     affine_map<(page, vec) -> (page, vec)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @paged_gather_read(%storage : !storage, %ind: !ind) -> !x {
  %x = tensor.empty() : !x
  %x_g = linalg.generic #gather
         ins(%ind : !ind)
         outs(%x : !x) {
  ^bb0(%page: i64, %out: f16):
    %pageidx = arith.index_cast %page : i64 to index
    %vec   = linalg.index 1 : index
    %extracted = tensor.extract %storage[%pageidx, %vec] : !storage
    linalg.yield %extracted : f16
  } -> !x
  return %x_g : !x
}
// CHECK-LABEL: @paged_gather_read
// CHECK-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>, %[[ARG1:.+]]: tensor<128xi64>
// CHECK: %[[INDEX_LOAD:.+]] = vector.transfer_read %[[ARG1]]
// CHECK: %[[INDEX_CAST:.+]] = arith.index_cast %[[INDEX_LOAD]] : vector<128xi64> to vector<128xindex>
// CHECK: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK-SAME: [%[[INDEX_CAST]] : vector<128xindex>]
// CHECK: vector.transfer_write %[[GATHER]], %{{.*}}

// -----

!storage = tensor<8192x8xf16>
!x       = tensor<128x8xf16>

#gather = {
    indexing_maps = [affine_map<(page, vec) -> (page, vec)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @contiguous_gather_read(%storage : !storage) -> !x {
  %x = tensor.empty() : !x
  %x_g = linalg.generic #gather
         outs(%x : !x) {
  ^bb0(%out: f16):
    %pageidx = linalg.index 0 : index
    %vec   = linalg.index 1 : index
    %extracted = tensor.extract %storage[%pageidx, %vec] : !storage
    linalg.yield %extracted : f16
  } -> !x
  return %x_g : !x
}
// CHECK-LABEL: @contiguous_gather_read
// CHECK-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>
// CHECK: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK: vector.transfer_write %[[GATHER]], %{{.*}}

// -----

!storage = tensor<8192x8xf16>
!ind     = tensor<128xi64>
!x       = tensor<128x8xf16>

#gather = {
    indexing_maps = [affine_map<(page, vec) -> (page)>,
                     affine_map<(page, vec) -> (page, vec)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @strided_paged_gather_read(%storage : !storage, %ind: !ind) -> !x {
  %x = tensor.empty() : !x
  %c2 = arith.constant 2 : index
  %x_g = linalg.generic #gather
         ins(%ind : !ind)
         outs(%x : !x) {
  ^bb0(%page: i64, %out: f16):
    %pageidx = arith.index_cast %page : i64 to index
    %vec   = linalg.index 1 : index
    %strided_vec = arith.muli %vec, %c2 : index
    %extracted = tensor.extract %storage[%pageidx, %strided_vec] : !storage
    linalg.yield %extracted : f16
  } -> !x
  return %x_g : !x
}
// The strided index (arith.muli) is treated as a gathered dimension.
// CHECK-LABEL: @strided_paged_gather_read
// CHECK: %[[GATHER:.+]] = iree_vector_ext.transfer_gather
// CHECK: vector.transfer_write %[[GATHER]], %{{.*}}

// -----

!storage = tensor<8192x8xf16>
!ind0     = tensor<128xi64>
!ind1     = tensor<8xi64>
!x       = tensor<128x8xf16>

#gather = {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @full_gather_read(%storage : !storage, %ind0: !ind0, %ind1 : !ind1) -> !x {
  %x = tensor.empty() : !x
  %x_g = linalg.generic #gather
         ins(%ind0, %ind1 : !ind0, !ind1)
         outs(%x : !x) {
   ^bb0(%id0: i64, %id1 : i64, %out: f16):
    %idx0 = arith.index_cast %id0 : i64 to index
    %idx1 = arith.index_cast %id1 : i64 to index
    %extracted = tensor.extract %storage[%idx0, %idx1] : !storage
    linalg.yield %extracted : f16
  } -> !x
  return %x_g : !x
}
// CHECK-LABEL: @full_gather_read
// CHECK-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>, %[[ARG1:.+]]: tensor<128xi64>, %[[ARG2:.+]]: tensor<8xi64>
// CHECK-DAG: %[[IDX0:.+]] = vector.transfer_read %[[ARG1]]
// CHECK-DAG: %[[IDX1:.+]] = vector.transfer_read %[[ARG2]]
// CHECK-DAG: %[[CAST0:.+]] = arith.index_cast %[[IDX0]] : vector<128xi64> to vector<128xindex>
// CHECK-DAG: %[[CAST1:.+]] = arith.index_cast %[[IDX1]] : vector<8xi64> to vector<8xindex>
// CHECK-DAG: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK-SAME: [%[[CAST0]], %[[CAST1]] : vector<128xindex>, vector<8xindex>]
// CHECK: vector.transfer_write %[[GATHER]], %{{.*}}

// -----

!storage = tensor<8192x8xf16>
!ind0     = tensor<128xi64>
!ind1     = tensor<8xi64>
!x       = tensor<128x8xf16>

#gather = {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @multi_extract(%storage : !storage, %storage2: !storage, %ind0: !ind0, %ind1 : !ind1) -> ( !x, !x ) {
  %x = tensor.empty() : !x
  %x_g, %x_g1 = linalg.generic #gather
         ins(%ind0, %ind1 : !ind0, !ind1)
         outs(%x, %x : !x, !x) {
   ^bb0(%id0: i64, %id1 : i64, %out: f16, %out2: f16):
    %idx0 = arith.index_cast %id0 : i64 to index
    %idx1 = arith.index_cast %id1 : i64 to index
    %extracted = tensor.extract %storage[%idx0, %idx1] : !storage
    %idx2 = arith.index_cast %id0 : i64 to index
    %idx3 = arith.index_cast %id1 : i64 to index
    %extracted1 = tensor.extract %storage2[%idx2, %idx3] : !storage
    linalg.yield %extracted, %extracted1 : f16, f16
  } -> (!x, !x)
  return %x_g, %x_g1 : !x, !x
}
// CHECK-LABEL: @multi_extract
// CHECK-COUNT-2: transfer_gather

// -----

func.func @implicit_gather_like_generic_stride_2(%arg0: tensor<1x1x31xf32>, %arg1: tensor<1x1x1x1x16xf32>) -> tensor<1x1x1x1x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4 * 2 + d0)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<1x1x31xf32>) outs(%arg1 : tensor<1x1x1x1x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1x1x1x16xf32>
  return %0 : tensor<1x1x1x1x16xf32>
}
// CHECK:       #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4)[s0] -> (0, 0, s0)>
// CHECK:       #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4)[s0] -> (d4)>
// CHECK-LABEL: func.func @implicit_gather_like_generic_stride_2
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x1x31xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x1x1x1x16xf32>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[DENSE:.+]] = arith.constant dense<2> : vector<16xindex>
// CHECK-DAG:     %[[PASSTHRU:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[STEP:.+]] = vector.step : vector<16xindex>
// CHECK:         %[[INDICES:.+]] = arith.muli %[[STEP]], %[[DENSE]] : vector<16xindex>
// CHECK:         %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[IN]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      [%[[INDICES]] : vector<16xindex>], %[[PASSTHRU]]
// CHECK-SAME:      {indexing_maps = [#[[$MAP0]], #[[$MAP1]]]}
// CHECK:         %[[RESULT:.+]] = vector.transfer_write %[[GATHER]], %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK:         return %[[RESULT]]

// -----

func.func @implicit_gather_strided_leading_dims(%arg0: tensor<1x1x3xf32>, %arg1: tensor<1x1x1x1x3xf32>) -> tensor<1x1x1x1x3xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d2, d1 * 2 + d3, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<1x1x3xf32>) outs(%arg1 : tensor<1x1x1x1x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1x1x1x3xf32>
  return %0 : tensor<1x1x1x1x3xf32>
}
// CHECK:       #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (0, 0, d4)>
// CHECK-LABEL: func.func @implicit_gather_strided_leading_dims
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x1x3xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x1x1x1x3xf32>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[PASSTHRU:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[IN]][%[[C0]], %[[C0]], %[[C0]]],
// CHECK-SAME:      %[[PASSTHRU]] {indexing_maps = [#[[$MAP0]]]}
// CHECK:         %[[RESULT:.+]] = vector.transfer_write %[[GATHER]], %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK:         return %[[RESULT]]

// -----

// Dynamic paged gather: vector sizes inferred from affine.min upper bound.

#gather = {
    indexing_maps = [affine_map<(page, vec) -> (page)>,
                     affine_map<(page, vec) -> (page, vec)>],
    iterator_types = ["parallel", "parallel"]
}
func.func @dynamic_paged_gather_read(
    %storage : tensor<8192x8xf16>, %ind: tensor<?xi64>, %n: index) -> tensor<?x8xf16> {
  %sz = affine.min affine_map<(d0) -> (d0, 128)>(%n)
  %ind_slice = tensor.extract_slice %ind[0] [%sz] [1] : tensor<?xi64> to tensor<?xi64>
  %empty = tensor.empty(%sz) : tensor<?x8xf16>
  %gathered = linalg.generic #gather
      ins(%ind_slice : tensor<?xi64>)
      outs(%empty : tensor<?x8xf16>) {
  ^bb0(%page: i64, %o: f16):
    %pageidx = arith.index_cast %page : i64 to index
    %vec = linalg.index 1 : index
    %extracted = tensor.extract %storage[%pageidx, %vec] : tensor<8192x8xf16>
    linalg.yield %extracted : f16
  } -> tensor<?x8xf16>
  return %gathered : tensor<?x8xf16>
}
// CHECK-LABEL: func.func @dynamic_paged_gather_read
// CHECK-SAME:    %[[STORAGE:[a-zA-Z0-9]+]]: tensor<8192x8xf16>
// CHECK-SAME:    %[[IND:[a-zA-Z0-9]+]]: tensor<?xi64>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[SZ:.+]] = affine.min
// CHECK:         %[[IND_SLICE:.+]] = tensor.extract_slice %[[IND]]
// CHECK:         %[[MASK_1D:.+]] = vector.create_mask %[[SZ]] : vector<128xi1>
// CHECK:         %[[IND_VEC:.+]] = vector.transfer_read %[[IND_SLICE]][%[[C0]]], {{.*}}, %[[MASK_1D]]
// CHECK:         %[[IDX_CAST:.+]] = arith.index_cast %[[IND_VEC]] : vector<128xi64> to vector<128xindex>
// CHECK:         iree_vector_ext.transfer_gather %[[STORAGE]]
// CHECK-SAME:      [%[[IDX_CAST]] : vector<128xindex>]
// CHECK:         vector.transfer_write

// -----

// Dynamic contiguous gather: vector sizes inferred from affine.min upper bound.

#gather = {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
}
func.func @dynamic_contiguous_gather_read(
    %storage : tensor<8192x8xf16>, %n: index) -> tensor<?x8xf16> {
  %sz = affine.min affine_map<(d0) -> (d0, 64)>(%n)
  %empty = tensor.empty(%sz) : tensor<?x8xf16>
  %gathered = linalg.generic #gather
      outs(%empty : tensor<?x8xf16>) {
  ^bb0(%o: f16):
    %row = linalg.index 0 : index
    %vec = linalg.index 1 : index
    %extracted = tensor.extract %storage[%row, %vec] : tensor<8192x8xf16>
    linalg.yield %extracted : f16
  } -> tensor<?x8xf16>
  return %gathered : tensor<?x8xf16>
}
// CHECK-LABEL: func.func @dynamic_contiguous_gather_read
// CHECK-SAME:    %[[STORAGE:[a-zA-Z0-9]+]]: tensor<8192x8xf16>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[STORAGE]]
// CHECK-SAME:      : tensor<8192x8xf16>, vector<64x8xf16>
// CHECK:         vector.transfer_write %[[GATHER]]

// -----

// Multiple chained gathers: the index vectors computed for the first two
// gathers (and the clamp ops derived from their results) must be reused
// directly as vector SSA values by subsequent gathers, without materializing
// tensor.empty<...xindex> intermediaries and write-read chains.

#m_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#m_2d = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @three_gathers_no_index_tensor(
    %in0: tensor<1x8x8xf32>,
    %in1: tensor<1x8xf32>, %in2: tensor<1x8xf32>,
    %out_init: tensor<1x8x8xf32>,
    %indir_table: tensor<50x32x25x2xi32>,
    %lut: tensor<50x40x40xi8>,
    %arg0: index, %arg2: index, %arg4: index, %arg6: index)
    -> tensor<1x8x8xf32> {
  %c0    = arith.constant 0   : index
  %c1    = arith.constant 1   : index
  %c39   = arith.constant 39  : index
  %c0_i8 = arith.constant 0   : i8
  %cst   = arith.constant 0.0 : f32
  %0 = linalg.generic {
        indexing_maps = [#m_3d, #m_2d, #m_2d, #m_3d],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%in0, %in1, %in2 : tensor<1x8x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>)
        outs(%out_init : tensor<1x8x8xf32>) {
  ^bb0(%in: f32, %a: f32, %b: f32, %out: f32):
    %m1 = arith.divf %a, %b : f32
    %p  = arith.cmpf une, %in, %cst : f32
    %v  = arith.select %p, %m1, %cst : f32
    %i0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg0, %arg2)
    %d1 = linalg.index 1 : index
    %i1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg4)[%d1]
    %d2 = linalg.index 2 : index
    %i2 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg6)[%d2]
    // Gather #1 and #2: identical outer indices, last index differs.
    %ea = tensor.extract %indir_table[%i0, %i1, %i2, %c0] : tensor<50x32x25x2xi32>
    %ea_idx = arith.index_cast %ea : i32 to index
    %eb = tensor.extract %indir_table[%i0, %i1, %i2, %c1] : tensor<50x32x25x2xi32>
    %eb_idx = arith.index_cast %eb : i32 to index
    %ea_max = arith.maxsi %ea_idx, %c0  : index
    %ea_min = arith.minui %ea_max, %c39 : index
    %eb_max = arith.maxsi %eb_idx, %c0  : index
    %eb_min = arith.minui %eb_max, %c39 : index
    // Gather #3: depends on results of #1 and #2.
    %ec = tensor.extract %lut[%i0, %ea_min, %eb_min] : tensor<50x40x40xi8>
    %gate = arith.cmpi ugt, %ec, %c0_i8 : i8
    %r = arith.select %gate, %v, %cst : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8xf32>
  return %0 : tensor<1x8x8xf32>
}
// Verify three transfer_gather ops are produced. Index vectors from the first
// two gathers (and the clamp ops on their results) feed directly into the
// third gather as vector SSA values — no tensor.empty<...xindex> or
// write-read chains.
//
// CHECK-LABEL: func.func @three_gathers_no_index_tensor
// CHECK-SAME:    %[[IN0:[a-zA-Z0-9]+]]: tensor<1x8x8xf32>
// CHECK-SAME:    %[[IN1:[a-zA-Z0-9]+]]: tensor<1x8xf32>
// CHECK-SAME:    %[[IN2:[a-zA-Z0-9]+]]: tensor<1x8xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x8x8xf32>
// CHECK-SAME:    %[[TABLE:[a-zA-Z0-9]+]]: tensor<50x32x25x2xi32>
// CHECK-SAME:    %[[LUT:[a-zA-Z0-9]+]]: tensor<50x40x40xi8>
//
//     No index-typed tensors should appear anywhere in the output.
// CHECK-NOT:     tensor<{{.*}}xindex>
//
//     Gather #1 from %indir_table — index vecs consumed directly.
// CHECK:         %[[G1:.+]] = iree_vector_ext.transfer_gather %[[TABLE]]
// CHECK-SAME:      : tensor<50x32x25x2xi32>, vector<1x8x8xi32>
// CHECK:         %[[G1_IDX:.+]] = arith.index_cast %[[G1]] : vector<1x8x8xi32> to vector<1x8x8xindex>
//
//     Gather #2 from %indir_table — reuses the same index vecs as #1.
// CHECK:         %[[G2:.+]] = iree_vector_ext.transfer_gather %[[TABLE]]
// CHECK-SAME:      : tensor<50x32x25x2xi32>, vector<1x8x8xi32>
// CHECK:         %[[G2_IDX:.+]] = arith.index_cast %[[G2]] : vector<1x8x8xi32> to vector<1x8x8xindex>
//
//     Clamp results of gather #1 and #2 — pure vector ops, no tensor roundtrip.
//     G1_IDX -> maxsi -> minui = CLAMP_A, G2_IDX -> maxsi -> minui = CLAMP_B.
// CHECK:         %[[G1_MAX:.+]] = arith.maxsi %[[G1_IDX]], {{.*}} : vector<1x8x8xindex>
// CHECK:         %[[CLAMP_A:.+]] = arith.minui %[[G1_MAX]], {{.*}} : vector<1x8x8xindex>
// CHECK:         %[[G2_MAX:.+]] = arith.maxsi %[[G2_IDX]], {{.*}} : vector<1x8x8xindex>
// CHECK:         %[[CLAMP_B:.+]] = arith.minui %[[G2_MAX]], {{.*}} : vector<1x8x8xindex>
//
//     Gather #3 from %lut — takes clamped results directly as index vectors.
// CHECK:         %[[G3:.+]] = iree_vector_ext.transfer_gather %[[LUT]]
// CHECK-SAME:      [{{.*}}, %[[CLAMP_A]], %[[CLAMP_B]] : {{.*}}]
// CHECK-SAME:      : tensor<50x40x40xi8>, vector<1x8x8xi8>
//
//     Final select + write.
// CHECK:         %[[GATE:.+]] = arith.cmpi ugt, %[[G3]]
// CHECK:         %[[RES:.+]] = arith.select %[[GATE]]
// CHECK:         vector.transfer_write %[[RES]], %[[OUT]]
// CHECK-NOT:     tensor<{{.*}}xindex>
