// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module {
  //  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
  //  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
  //  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
  //      CHECK: @concatenate
  //      CHECK: linalg.indexed_generic {
  // CHECK-SAME:   args_in = 2
  // CHECK-SAME:   args_out = 1
  // CHECK-SAME:   indexing_maps
  // CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
  // CHECK-SAME:   iterator_types = ["parallel", "reduction", "reduction", "reduction"]
  func @concatenate() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x3xi32>
    %2 = "mhlo.concatenate"(%0, %1) {
      dimension = 1
    } : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<2x5xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
//      CHECK: @concatenate
//      CHECK: linalg.indexed_generic {
// CHECK-SAME:   args_in = 2
// CHECK-SAME:   args_out = 1
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction", "reduction", "reduction"]
module {
  func @concatenate() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<3x2xi32>
    %2 = "mhlo.concatenate"(%0, %1) {
      dimension = 0
    } : (tensor<2x2xi32>, tensor<3x2xi32>) -> tensor<5x2xi32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<5x2xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
