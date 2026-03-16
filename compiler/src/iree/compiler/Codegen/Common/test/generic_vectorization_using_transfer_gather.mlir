// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{vectorize-to-transfer-gather=true}))" --split-input-file %s | FileCheck %s --check-prefix=CHECK-GATHER

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

// CHECK-GATHER-LABEL: @paged_gather_read
// CHECK-GATHER-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>, %[[ARG1:.+]]: tensor<128xi64>
// CHECK-GATHER: %[[INDEX_LOAD:.+]] = vector.transfer_read %[[ARG1]]
// CHECK-GATHER: %[[INDEX_CAST:.+]] = arith.index_cast %[[INDEX_LOAD]] : vector<128xi64> to vector<128xindex>
// CHECK-GATHER: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK-GATHER-SAME: [%[[INDEX_CAST]] : vector<128xindex>]
// CHECK-GATHER: vector.transfer_write %[[GATHER]], %{{.*}}

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

// CHECK-GATHER-LABEL: @contiguous_gather_read
// CHECK-GATHER-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>
// CHECK-GATHER: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK-GATHER: vector.transfer_write %[[GATHER]], %{{.*}}

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
// CHECK-GATHER-LABEL: @strided_paged_gather_read
// CHECK-GATHER: %[[GATHER:.+]] = iree_vector_ext.transfer_gather
// CHECK-GATHER: vector.transfer_write %[[GATHER]], %{{.*}}

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

// CHECK-GATHER-LABEL: @full_gather_read
// CHECK-GATHER-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>, %[[ARG1:.+]]: tensor<128xi64>, %[[ARG2:.+]]: tensor<8xi64>
// CHECK-GATHER-DAG: %[[IDX0:.+]] = vector.transfer_read %[[ARG1]]
// CHECK-GATHER-DAG: %[[IDX1:.+]] = vector.transfer_read %[[ARG2]]
// CHECK-GATHER-DAG: %[[CAST0:.+]] = arith.index_cast %[[IDX0]] : vector<128xi64> to vector<128xindex>
// CHECK-GATHER-DAG: %[[CAST1:.+]] = arith.index_cast %[[IDX1]] : vector<8xi64> to vector<8xindex>
// CHECK-GATHER-DAG: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK-GATHER-SAME: [%[[CAST0]], %[[CAST1]] : vector<128xindex>, vector<8xindex>]
// CHECK-GATHER: vector.transfer_write %[[GATHER]], %{{.*}}

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

// CHECK-GATHER-LABEL: @multi_extract
// CHECK-GATHER-COUNT-2: transfer_gather

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
// CHECK-GATHER:       #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4)[s0] -> (0, 0, s0)>
// CHECK-GATHER:       #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4)[s0] -> (d4)>
// CHECK-GATHER-LABEL: func.func @implicit_gather_like_generic_stride_2
// CHECK-GATHER-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x1x31xf32>
// CHECK-GATHER-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x1x1x1x16xf32>
// CHECK-GATHER-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-GATHER-DAG:     %[[DENSE:.+]] = arith.constant dense<2> : vector<16xindex>
// CHECK-GATHER-DAG:     %[[PASSTHRU:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-GATHER-DAG:     %[[STEP:.+]] = vector.step : vector<16xindex>
// CHECK-GATHER:         %[[INDICES:.+]] = arith.muli %[[STEP]], %[[DENSE]] : vector<16xindex>
// CHECK-GATHER:         %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[IN]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-GATHER-SAME:      [%[[INDICES]] : vector<16xindex>], %[[PASSTHRU]]
// CHECK-GATHER-SAME:      {indexing_maps = [#[[$MAP0]], #[[$MAP1]]]}
// CHECK-GATHER:         %[[RESULT:.+]] = vector.transfer_write %[[GATHER]], %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-GATHER:         return %[[RESULT]]

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
// CHECK-GATHER:       #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (0, 0, d4)>
// CHECK-GATHER-LABEL: func.func @implicit_gather_strided_leading_dims
// CHECK-GATHER-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x1x3xf32>
// CHECK-GATHER-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x1x1x1x3xf32>
// CHECK-GATHER-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-GATHER-DAG:     %[[PASSTHRU:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-GATHER:         %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[IN]][%[[C0]], %[[C0]], %[[C0]]],
// CHECK-GATHER-SAME:      %[[PASSTHRU]] {indexing_maps = [#[[$MAP0]]]}
// CHECK-GATHER:         %[[RESULT:.+]] = vector.transfer_write %[[GATHER]], %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-GATHER:         return %[[RESULT]]
