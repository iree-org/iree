// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-buffers -iree-linalg-vector-transforms %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: matmul_vectorization
// CHECK: %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4x3xf32>
module {
  func @matmul_vectorization() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<4x64xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<64x3xf32>
    %result = "xla_hlo.dot"(%0, %1) : (tensor<4x64xf32>, tensor<64x3xf32>) -> tensor<4x3xf32>
    hal.interface.store.tensor %result, @legacy_io::@ret0, offset = %c0 : tensor<4x3xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK: %[[ARG2_VEC:.+]] = vector.type_cast %[[ARG2]]
// CHECK: %[[RES_VEC:.+]] = vector.contract
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: vector<4x64xf32>, vector<64x3xf32> into vector<4x3xf32>
// CHECK: store %[[RES_VEC]], %[[ARG2_VEC]]
