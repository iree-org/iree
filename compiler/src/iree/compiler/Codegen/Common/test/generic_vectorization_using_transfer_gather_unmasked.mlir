// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false vectorize-to-transfer-gather=true}))" --split-input-file %s | FileCheck %s

// Static-shape companion to generic_vectorization_using_transfer_gather_masked.mlir,
// run with masking disabled. Pins down that gather-vectorization works when
// the pipeline is configured without masking. Dynamic cases are omitted: the
// vectorizer bails on them with masking=false.

#gather = {
    indexing_maps = [affine_map<(page, vec) -> (page)>,
                     affine_map<(page, vec) -> (page, vec)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @paged_gather_read(%storage : tensor<8192x8xf16>, %ind: tensor<128xi64>) -> tensor<128x8xf16> {
  %x = tensor.empty() : tensor<128x8xf16>
  %x_g = linalg.generic #gather
         ins(%ind : tensor<128xi64>)
         outs(%x : tensor<128x8xf16>) {
  ^bb0(%page: i64, %out: f16):
    %pageidx = arith.index_cast %page : i64 to index
    %vec   = linalg.index 1 : index
    %extracted = tensor.extract %storage[%pageidx, %vec] : tensor<8192x8xf16>
    linalg.yield %extracted : f16
  } -> tensor<128x8xf16>
  return %x_g : tensor<128x8xf16>
}
// CHECK-LABEL: @paged_gather_read
// CHECK-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>, %[[ARG1:.+]]: tensor<128xi64>
// CHECK: %[[INDEX_LOAD:.+]] = vector.transfer_read %[[ARG1]]
// CHECK: %[[INDEX_CAST:.+]] = arith.index_cast %[[INDEX_LOAD]] : vector<128xi64> to vector<128xindex>
// CHECK: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK-SAME: [%[[INDEX_CAST]] : vector<128xindex>]
// CHECK: vector.transfer_write %[[GATHER]], %{{.*}}

// -----

// Degenerate case: preExtract is empty (only linalg.index ops, which are
// consumed by the extract directly). preOp has no ops to vectorize and no
// outputs to harvest — the gather falls through to the contiguous path.

#gather = {
    indexing_maps = [affine_map<(page, vec) -> (page, vec)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @contiguous_gather_read(%storage : tensor<8192x8xf16>) -> tensor<128x8xf16> {
  %x = tensor.empty() : tensor<128x8xf16>
  %x_g = linalg.generic #gather
         outs(%x : tensor<128x8xf16>) {
  ^bb0(%out: f16):
    %pageidx = linalg.index 0 : index
    %vec   = linalg.index 1 : index
    %extracted = tensor.extract %storage[%pageidx, %vec] : tensor<8192x8xf16>
    linalg.yield %extracted : f16
  } -> tensor<128x8xf16>
  return %x_g : tensor<128x8xf16>
}
// CHECK-LABEL: @contiguous_gather_read
// CHECK-SAME: %[[ARG0:.+]]: tensor<8192x8xf16>
// CHECK: %[[GATHER:.+]] = iree_vector_ext.transfer_gather %[[ARG0]]
// CHECK: vector.transfer_write %[[GATHER]], %{{.*}}

// -----

#gather = {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
}

func.func @full_gather_read(%storage : tensor<8192x8xf16>, %ind0: tensor<128xi64>, %ind1 : tensor<8xi64>) -> tensor<128x8xf16> {
  %x = tensor.empty() : tensor<128x8xf16>
  %x_g = linalg.generic #gather
         ins(%ind0, %ind1 : tensor<128xi64>, tensor<8xi64>)
         outs(%x : tensor<128x8xf16>) {
   ^bb0(%id0: i64, %id1 : i64, %out: f16):
    %idx0 = arith.index_cast %id0 : i64 to index
    %idx1 = arith.index_cast %id1 : i64 to index
    %extracted = tensor.extract %storage[%idx0, %idx1] : tensor<8192x8xf16>
    linalg.yield %extracted : f16
  } -> tensor<128x8xf16>
  return %x_g : tensor<128x8xf16>
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
