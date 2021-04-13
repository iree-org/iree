// RUN: iree-opt -split-input-file -iree-flow-pre-partitioning-conversion %s | IreeFileCheck %s

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @linalg_generic
func @linalg_generic(%arg0 : tensor<1xi32>) {
  %init = linalg.init_tensor [1] : tensor<1xi32>
  %generic = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
    ins(%arg0 : tensor<1xi32>) outs(%init : tensor<1xi32>) {
    ^bb0(%arg1: i32, %s: i32):
      %idx = index_cast %arg1 : i32 to index
      // CHECK: tensor.extract
      %extract = tensor.extract %arg0[%idx] : tensor<1xi32>
      linalg.yield %extract : i32
    } -> tensor<1xi32>
  return
}

// -----

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @linalg_indexed_generic
func @linalg_indexed_generic(%arg0 : tensor<1xi32>) {
  %init = linalg.init_tensor [1] : tensor<1xi32>
  %generic = linalg.indexed_generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
    ins(%arg0 : tensor<1xi32>) outs(%init : tensor<1xi32>) {
    ^bb0(%arg1: index, %arg2: i32, %s: i32):
      %idx = index_cast %arg2 : i32 to index
      // CHECK: tensor.extract
      %extract = tensor.extract %arg0[%idx] : tensor<1xi32>
      linalg.yield %extract : i32
    } -> tensor<1xi32>
  return
}

// -----

func @tensor_extract(%arg0 : tensor<1xi32>, %arg1 : index) {
  // CHECK: flow.tensor.load
  %extract = tensor.extract %arg0[%arg1] : tensor<1xi32>
  return
}
