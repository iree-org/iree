// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-buffers -iree-linalg-vector-transforms %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: matmul_vectorization(%[[ARG0:.+]]: memref<4x64xf32>, %[[ARG1:.+]]: memref<64x3xf32>, %[[ARG2:.+]]: memref<4x3xf32>)
func @matmul_vectorization(%arg0: memref<4x64xf32>, %arg1: memref<64x3xf32>, %arg3: memref<4x3xf32>) attributes {iree.dispatch_fn_name = ""} {
  %0 = iree.load_input(%arg0: memref<4x64xf32>) : tensor<4x64xf32>
  %1 = iree.load_input(%arg1: memref<64x3xf32>) : tensor<64x3xf32>
  %result = "xla_hlo.dot"(%0, %1) : (tensor<4x64xf32>, tensor<64x3xf32>) -> tensor<4x3xf32>
  iree.store_output(%result: tensor<4x3xf32>, %arg3: memref<4x3xf32>)
  return
}
// CHECK: %[[ARG2_VEC:.+]] = vector.type_cast %[[ARG2]]
// CHECK: %[[RES_VEC:.+]] = vector.contract
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: vector<4x64xf32>, vector<64x3xf32> into vector<4x3xf32>
// CHECK: store %[[RES_VEC]], %[[ARG2_VEC]]
