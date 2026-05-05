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
