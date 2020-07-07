// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
module {
  //      CHECK: func @reduction_entry
  //  CHECK-DAG: %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<5xf32>
  //  CHECK-DAG: %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<5x4xf32>
  //  CHECK-DAG: %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
  //      CHECK: linalg.indexed_generic {args_in = 2 : i64, args_out = 1 : i64,
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: #[[MAP0]], #[[MAP1]], #[[MAP2]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]] {
  // CHECK-NEXT: ^{{.+}}(%{{.+}}, %[[IDX:.+]]: index, %[[SRC:.+]]: f32, %[[INIT:.+]]: f32, %[[DST:.+]]: f32):
  //      CHECK:   %[[OPERAND:.+]] = select %{{.+}}, %[[INIT]], %[[DST]] : f32
  // CHECK-NEXT:   %[[RES:.+]] = addf %[[SRC]], %[[OPERAND]] : f32
  // CHECK-NEXT:   linalg.yield %[[RES]] : f32
  // CHECK-NEXT: }: memref<5x4xf32>, memref<f32>, memref<5xf32>
  func @reduction_entry() {
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
  //      CHECK:   %[[COND:.+]] = cmpf "olt", %{{.+}}, %{{.+}} : f32
  // CHECK-NEXT:   select %[[COND]], %{{.+}}, %{{.+}} : f32
  func @reduction_entry() {
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
  //      CHECK:   %[[COND:.+]] = cmpf "ogt", %{{.+}}, %{{.+}} : f32
  // CHECK-NEXT:   select %[[COND]], %{{.+}}, %{{.+}} : f32
  func @reduction_entry() {
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

module {
  //      CHECK:   %[[COND:.+]] = cmpf "ogt", %{{.+}}, %{{.+}} : f32
  // CHECK-NEXT:   select %[[COND]], %{{.+}}, %{{.+}} : f32
  func @reduction_entry() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<5x4xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<f32>
    %2 = "mhlo.reduce"(%0, %1) ({
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = mhlo.maximum %arg3, %arg4 : tensor<f32>
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

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
module {
  //      CHECK: func @reduction_entry
  //      CHECK: %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4xf32>
  //      CHECK: %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<5x4xf32>
  //      CHECK: %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
  //      CHECK: linalg.indexed_generic {args_in = 2 : i64, args_out = 1 : i64,
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: #[[MAP0]], #[[MAP1]], #[[MAP2]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]}
  // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]] {
  // CHECK-NEXT: ^{{.+}}(%{{.+}}, %[[IDX:.+]]: index, %[[SRC:.+]]: f32, %[[INIT:.+]]: f32, %[[DST:.+]]: f32):
  //      CHECK:   %[[OPERAND:.+]] = select %{{.+}}, %[[INIT]], %[[DST]] : f32
  // CHECK-NEXT:   %[[RES:.+]] = addf %[[SRC]], %[[OPERAND]] : f32
  // CHECK-NEXT:   linalg.yield %[[RES]] : f32
  // CHECK-NEXT: }: memref<5x4xf32>, memref<f32>, memref<4xf32>
  func @reduction_entry() {
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

module {
  // CHECK-LABEL: func @reduce_init_const
  func @reduce_init_const() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x10xf32>
    // CHECK: %[[CST:.+]] = constant 0xFF800000 : f32
    // CHECK: linalg.indexed_generic
    // CHECK-SAME: args_in = 1
    // CHECK-SAME: args_out = 1
    // CHECK: ^{{.+}}(%{{.+}}: index, %[[DIM:.+]]: index, %{{.+}}: f32, %[[OUTPUT:.+]]: f32):
    // CHECK: select %{{.+}}, %[[CST]], %[[OUTPUT]] : f32
    %cst = constant dense<0xFF800000> : tensor<f32>
    %1 = "mhlo.reduce"(%0, %cst) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>): // no predecessors
      %2 = mhlo.add %arg2, %arg3 {name = "maximum.21"} : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<1xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0)>
module {
  //      CHECK: func @reduction_multi_dimensions
  //      CHECK: %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4xf32>
  //      CHECK: %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<5x4x3xf32>
  //      CHECK: %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<f32>
  //      CHECK: linalg.indexed_generic {args_in = 2 : i64, args_out = 1 : i64,
  // CHECK-SAME: indexing_maps
  // CHECK-SAME: #[[MAP0]], #[[MAP1]], #[[MAP2]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]}
  // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]] {
  // CHECK-NEXT: ^{{.+}}(%{{.+}}, %[[IDX:.+]]: index, %[[SRC:.+]]: f32, %[[INIT:.+]]: f32, %[[DST:.+]]: f32):
  //      CHECK:   %[[TRUE:.+]] = constant true
  //      CHECK:   %[[CMP1:.+]] = cmpi
  //      CHECK:   %[[COND1:.+]] = and %[[TRUE]], %[[CMP1]]
  //      CHECK:   %[[CMP2:.+]] = cmpi
  //      CHECK:   %[[COND2:.+]] = and %[[COND1]], %[[CMP2]]
  // CHECK-NEXT:   %[[OPERAND:.+]] = select %[[COND2]], %[[INIT]], %[[DST]] : f32
  // CHECK-NEXT:   %[[RES:.+]] = addf %[[SRC]], %[[OPERAND]] : f32
  // CHECK-NEXT:   linalg.yield %[[RES]] : f32
  // CHECK-NEXT: }: memref<5x4x3xf32>, memref<f32>, memref<4xf32>
  func @reduction_multi_dimensions() {
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
