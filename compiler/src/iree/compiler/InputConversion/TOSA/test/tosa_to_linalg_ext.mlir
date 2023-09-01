// RUN: iree-opt --split-input-file --iree-tosa-to-linalg-ext --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @scatter_static
func.func @scatter_static(%arg0 : tensor<1x4x5xf32>, %arg1 : tensor<1x2xi32>, %arg2 : tensor<1x2x5xf32>) ->  tensor<1x4x5xf32> {
  // CHECK: %[[EXPANDIDX:.+]] = tensor.expand_shape %arg1
  // CHECK-SAME{literal}: [[0], [1, 2]] : tensor<1x2xi32> into tensor<1x2x1xi32>
  // CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<1x2x1xi32>
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<1x2x1xi32>)
  // CHECK-DAG: %[[CONCAT:.+]] = tosa.concat %[[FILL]], %[[EXPANDIDX]] {axis = 2 : i32}
  // CHECK: %[[COLLAPSE_IDX:.+]] = tensor.collapse_shape %[[CONCAT]]
  // CHECK-SAME{literal}: [[0, 1], [2]] : tensor<1x2x2xi32> into tensor<2x2xi32>
  // CHECK: %[[COLLAPSE_UPD:.+]] = tensor.collapse_shape %arg2
  // CHECK-SAME{literal}: [[0, 1], [2]] : tensor<1x2x5xf32> into tensor<2x5xf32>
  // CHECK: %[[SCATTER:.+]] = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
  // CHECK-SAME: ins(%[[COLLAPSE_UPD]], %[[COLLAPSE_IDX]] : tensor<2x5xf32>, tensor<2x2xi32>)
  // CHECK-SAME: outs(%arg0 : tensor<1x4x5xf32>)
  // CHECK: ^bb0(%[[UPD:.+]]: f32, %{{.+}}: f32):
  // CHECK:   iree_linalg_ext.yield %[[UPD]]
  // CHECK: } -> tensor<1x4x5xf32>
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<1x4x5xf32>, tensor<1x2xi32>, tensor<1x2x5xf32>)  -> (tensor<1x4x5xf32>)

  // CHECK: return %[[SCATTER]]
  return %0 : tensor<1x4x5xf32>
}

// -----

// CHECK-LABEL: @scatter_static_batched
func.func @scatter_static_batched(%arg0 : tensor<2x4x5xf32>, %arg1 : tensor<2x2xi32>, %arg2 : tensor<2x2x5xf32>) ->  tensor<2x4x5xf32> {
  // CHECK: %[[EXPANDIDX:.+]] = tensor.expand_shape %arg1
  // CHECK-SAME{literal}: [[0], [1, 2]] : tensor<2x2xi32> into tensor<2x2x1xi32>
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x2x1xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[EXPANDIDX]] : tensor<2x2x1xi32>)
  // CHECK-SAME: outs(%[[EMPTY:.+]] : tensor<2x2x1xi32>) {
  // CHECK:   %[[IDX:.+]] = linalg.index 0 : index
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]] : index to i32
  // CHECK:   linalg.yield %[[CAST]] : i32
  // CHECK: %[[CONCAT:.+]] = tosa.concat %[[GENERIC]], %[[EXPANDIDX]] {axis = 2 : i32}
  // CHECK: %[[COLLAPSE_IDX:.+]] = tensor.collapse_shape %[[CONCAT]]
  // CHECK-SAME{literal}: [[0, 1], [2]] : tensor<2x2x2xi32> into tensor<4x2xi32>
  // CHECK: %[[COLLAPSE_UPD:.+]] = tensor.collapse_shape %arg2
  // CHECK-SAME{literal}: [[0, 1], [2]] : tensor<2x2x5xf32> into tensor<4x5xf32>
  // CHECK: %[[SCATTER:.+]] = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
  // CHECK-SAME: ins(%[[COLLAPSE_UPD]], %[[COLLAPSE_IDX]] : tensor<4x5xf32>, tensor<4x2xi32>)
  // CHECK-SAME: outs(%arg0 : tensor<2x4x5xf32>)
  // CHECK: ^bb0(%[[UPD:.+]]: f32, %{{.+}}: f32):
  // CHECK:   iree_linalg_ext.yield %[[UPD]]
  // CHECK: } -> tensor<2x4x5xf32>
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<2x4x5xf32>, tensor<2x2xi32>, tensor<2x2x5xf32>)  -> (tensor<2x4x5xf32>)

  // CHECK: return %[[SCATTER]]
  return %0 : tensor<2x4x5xf32>
}

// -----

// CHECK-LABEL: @scatter_dynamic
func.func @scatter_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xi32>, %arg2 : tensor<?x?x?xf32>) ->  tensor<?x?x?xf32> {
  // CHECK-DAG: %[[EXPAND:.+]] = tensor.expand_shape %arg1
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[DIM0:.+]] = tensor.dim %[[EXPAND]], %[[C0]] : tensor<?x?x1xi32>
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %[[EXPAND]], %[[C1]] : tensor<?x?x1xi32>
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?x1xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[EXPAND]] : tensor<?x?x1xi32>) outs(%[[EMPTY]] : tensor<?x?x1xi32>) {
  // CHECK: %[[CONCAT:.+]] = tosa.concat %[[GENERIC]], %[[EXPAND]] {axis = 2 : i32}
  // CHECK: %[[COLLAPSE_IDX:.+]] = tensor.collapse_shape %[[CONCAT]]
  // CHECK-SAME{literal}: [[0, 1], [2]] : tensor<?x?x2xi32> into tensor<?x2xi32>
  // CHECK: %[[COLLAPSE_UPD:.+]] = tensor.collapse_shape %arg2
  // CHECK-SAME{literal}: [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
  // CHECK: %[[SCATTER:.+]] = iree_linalg_ext.scatter
  // CHECK-SAME: ins(%[[COLLAPSE_UPD]], %[[COLLAPSE_IDX]] : tensor<?x?xf32>, tensor<?x2xi32>) outs(%arg0 : tensor<?x?x?xf32>)
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x?x?xf32>, tensor<?x?xi32>, tensor<?x?x?xf32>)  -> (tensor<?x?x?xf32>)

  // CHECK: return %[[SCATTER]]
  return %0 : tensor<?x?x?xf32>
}

