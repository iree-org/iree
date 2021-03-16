// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-pipeline %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
module {
  //      CHECK: func @reduce_add
  //  CHECK-DAG: %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<5xf32>
  //  CHECK-DAG: %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<5x4xf32>
  //  CHECK-DAG: %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
  //      CHECK: %[[INIT:.+]] = memref.load %[[ARG1]][] : memref<f32>
  //      CHECK: linalg.fill(%[[ARG2]], %[[INIT]])
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: #[[MAP0]], #[[MAP1]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME:   ins(%[[ARG0]] : memref<5x4xf32>
  // CHECK-SAME:   outs(%[[ARG2]] : memref<5xf32>
  // CHECK-NEXT: ^{{.+}}(%[[SRC:.+]]: f32, %[[DST:.+]]: f32):
  // CHECK-NEXT:   %[[RES:.+]] = addf %[[SRC]], %[[DST]] : f32
  // CHECK-NEXT:   linalg.yield %[[RES]] : f32
  // CHECK-NEXT: }
  func @reduce_add() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x4xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce"(%0, %1) ({
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<5xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

module {
  //      CHECK:   %[[COND:.+]] = cmpf olt, %{{.+}}, %{{.+}} : f32
  // CHECK-NEXT:   select %[[COND]], %{{.+}}, %{{.+}} : f32
  func @reduce_minimum() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x4xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce"(%0, %1) ({
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.minimum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<5xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

module {
  //      CHECK:   %[[COND:.+]] = cmpf ogt, %{{.+}}, %{{.+}} : f32
  // CHECK-NEXT:   select %[[COND]], %{{.+}}, %{{.+}} : f32
  func @reduce_maximum() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x4xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce"(%0, %1) ({
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<5xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
module {
  //      CHECK: func @reduce_dim0
  //  CHECK-DAG: %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4xf32>
  //  CHECK-DAG: %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<5x4xf32>
  //  CHECK-DAG: %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
  //      CHECK: %[[INIT:.+]] = memref.load %[[ARG1]][] : memref<f32>
  //      CHECK: linalg.fill(%[[ARG2]], %[[INIT]])
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: #[[MAP0]], #[[MAP1]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME:   ins(%[[ARG0]] : memref<5x4xf32>
  // CHECK-SAME:   outs(%[[ARG2]] : memref<4xf32>
  // CHECK-NEXT: ^{{.+}}(%[[SRC:.+]]: f32, %[[DST:.+]]: f32):
  // CHECK-NEXT:   %[[RES:.+]] = addf %[[SRC]], %[[DST]] : f32
  // CHECK-NEXT:   linalg.yield %[[RES]] : f32
  // CHECK-NEXT: }
  func @reduce_dim0() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x4xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce"(%0, %1) ({
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<4xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
module {
  //      CHECK: func @reduce_init_const
  //  CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2xf32>
  //  CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x10xf32>
  //  CHECK-DAG: %[[CST:.+]] = constant 0xFF800000 : f32
  //      CHECK: linalg.fill(%[[OUT]], %[[CST]])
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: #[[MAP0]], #[[MAP1]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME:   ins(%[[ARG0]] : memref<2x10xf32>
  // CHECK-SAME:   outs(%[[ARG2]] : memref<2xf32>
  // CHECK-NEXT: ^{{.+}}(%[[SRC:.+]]: f32, %[[DST:.+]]: f32):
  // CHECK-NEXT:   %[[RES:.+]] = addf %[[SRC]], %[[DST]] : f32
  // CHECK-NEXT:   linalg.yield %[[RES]] : f32
  // CHECK-NEXT: }
  func @reduce_init_const() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x10xf32>
    %cst = constant dense<0xFF800000> : tensor<f32>
    %1 = "mhlo.reduce"(%0, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>): // no predecessors
      %2 = mhlo.add %arg2, %arg3 {name = "maximum.21"} : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x10xf32>, tensor<f32>) -> tensor<2xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<2xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0)>
module {
  //      CHECK: func @reduce_multi_dimensions
  //      CHECK: %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4xf32>
  //      CHECK: %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<5x4x3xf32>
  //      CHECK: %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
  //      CHECK: %[[INIT:.+]] = memref.load %[[ARG1]][] : memref<f32>
  //      CHECK: linalg.fill(%[[ARG2]], %[[INIT]])
  //      CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: #[[MAP0]], #[[MAP1]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]}
  // CHECK-SAME: ins(%[[ARG0]] : memref<5x4x3xf32>)
  // CHECK-SAME: outs(%[[ARG2]] : memref<4xf32>)
  // CHECK-NEXT: ^{{.+}}(%[[SRC:.+]]: f32, %[[DST:.+]]: f32):
  // CHECK-NEXT:   %[[RES:.+]] = addf %[[SRC]], %[[DST]] : f32
  // CHECK-NEXT:   linalg.yield %[[RES]] : f32
  // CHECK-NEXT: }
  func @reduce_multi_dimensions() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x4x3xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce"(%0, %1) ({
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%3) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<5x4x3xf32>, tensor<f32>) -> tensor<4xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
