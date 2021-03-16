// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

//    CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
//    CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  //      CHECK: func @torch_select_index
  //  CHECK-DAG: %[[INPUT:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x1x5xi32>
  //  CHECK-DAG: %[[INDEX:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2xi32>
  //      CHECK: linalg.indexed_generic {
  // CHECK-SAME:   indexing_maps
  // CHECK-SAME:   #[[MAP0]], #[[MAP1]]
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
  // CHECK-SAME: ins(%[[INDEX]] : tensor<2xi32>)
  //      CHECK: ^{{.+}}(
  // CHECK-SAME:   %[[I:.+]]: index, %[[J:.+]]: index, %[[K:.+]]: index
  // CHECK-SAME:   %[[VAL:.+]]: i32, %{{.+}}: i32):
  //      CHECK:   %[[CAST:.+]] = index_cast %[[VAL]] : i32 to index
  //      CHECK:   %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[CAST]], %[[J]], %[[K]]] : tensor<5x1x5xi32>
  //      CHECK:   linalg.yield %[[VAL2]] : i32
  func @torch_select_index() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x1x5xi32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2xi32>
    %2 = "mhlo.torch_index_select"(%0, %1) {
      dim = 0 : i64,
      batch_dims = 0 : i64
    } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<2x1x5xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

//    CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> ()>
//    CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
module {
  //  CHECK-DAG: %[[INPUT:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<4x8xf32>
  //  CHECK-DAG: %[[INDEX:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<i32>
  //      CHECK: %[[T0:.+]] = linalg.init_tensor [8] : tensor<8xf32>
  //      CHECK: linalg.indexed_generic {
  // CHECK-SAME:   indexing_maps
  // CHECK-SAME:   #[[MAP0]], #[[MAP1]]
  // CHECK-SAME:   iterator_types = ["parallel"]
  // CHECK-SAME:   ins(%[[INDEX]] : tensor<i32>) outs(%[[T0]] : tensor<8xf32>)
  //      CHECK:   ^{{.+}}(
  // CHECK-SAME:     %[[I:[a-zA-Z0-9_]+]]: index, %[[VAL:[a-zA-Z0-9_]+]]: i32, %{{.+}}: f32):
  //      CHECK:     %[[CAST:.+]] = index_cast %[[VAL]] : i32 to index
  //      CHECK:     %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[CAST]], %[[I]]] : tensor<4x8xf32>
  //      CHECK:     linalg.yield %[[VAL2]] : f32
  func @torch_select_index_scalar() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<4x8xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<i32>
    %2 = "mhlo.torch_index_select"(%0, %1) {
      batch_dims = 0 : i64,
      dim = 0 : i64
    } : (tensor<4x8xf32>, tensor<i32>) -> tensor<8xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<8xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

//    CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//    CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  //  CHECK-DAG: %[[INPUT:.+]] = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<4x7x8x2xf32>
  //  CHECK-DAG: %[[INDEX:.+]] = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<4x1xi32>
  //      CHECK: linalg.indexed_generic {
  // CHECK-SAME:   indexing_maps
  // CHECK-SAME:   #[[MAP0]], #[[MAP1]]
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK-SAME: ins(%[[INDEX]] : tensor<4x1xi32>)
  // CHECK-NEXT: ^{{.+}}(
  // CHECK-SAME:   %[[I:[a-zA-Z0-9_]+]]: index, %[[J:[a-zA-Z0-9_]+]]: index,
  // CHECK-SAME:   %[[K:[a-zA-Z0-9_]+]]: index, %[[L:.+]]: index,
  // CHECK-SAME:   %[[VAL:.+]]: i32, %{{.+}}: f32):
  //      CHECK:   %[[CAST:.+]] = index_cast %[[VAL]] : i32 to index
  //      CHECK:   %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[I]], %[[J]], %[[CAST]], %[[L]]] : tensor<4x7x8x2xf32>
  //      CHECK:   linalg.yield %[[VAL2]] : f32
  func @torch_select_index_batch() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<4x7x8x2xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<4x1xi32>
    %2 = "mhlo.torch_index_select"(%0, %1) {
      dim = 2 : i64,
      batch_dims = 1 : i64
    } : (tensor<4x7x8x2xf32>, tensor<4x1xi32>) -> tensor<4x7x1x2xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<4x7x1x2xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

func @torch_index_select_dynamic(%input: tensor<?x?x?x?xf32>,
                                 %index: tensor<?x?xi32>) -> tensor<?x?x?x?xf32>{
  %0 = "mhlo.torch_index_select"(%input, %index) {
    batch_dims = 1 : i64,
    dim = 2 : i64
  } : (tensor<?x?x?x?xf32>, tensor<?x?xi32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

//      CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//      CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @torch_index_select_dynamic
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK:   %[[C0:.+]] = constant 0 : index
//      CHECK:   %[[D0:.+]] = memref.dim %[[INPUT]], %[[C0]]
//      CHECK:   %[[C1:.+]] = constant 1 : index
//      CHECK:   %[[D1:.+]] = memref.dim %[[INPUT]], %[[C1]]
//      CHECK:   %[[C1:.+]] = constant 1 : index
//      CHECK:   %[[D2:.+]] = memref.dim %[[INDEX]], %[[C1]]
//      CHECK:   %[[C3:.+]] = constant 3 : index
//      CHECK:   %[[D3:.+]] = memref.dim %[[INPUT]], %[[C3]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]], %[[D3]]]
//      CHECK:   %[[RESULT:.+]] = linalg.indexed_generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[INDEX]] : tensor<?x?xi32>)
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?x?x?xf32>)
//      CHECK:     ^{{.+}}(
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:       %[[ARG1:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:       %[[ARG3:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:       %[[ARG4:[a-zA-Z0-9_]+]]: i32, %{{[a-zA-Z0-9_]+}}: f32)
//      CHECK:       %[[POS:.+]] = index_cast %[[ARG4]]
//      CHECK:       %[[YIELD:.+]] = tensor.extract %[[INPUT]][%[[ARG0]], %[[ARG1]], %[[POS]], %[[ARG3]]]
//      CHECK:       linalg.yield %[[YIELD]]
