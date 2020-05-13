// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK-LABEL: func @element_wise
  // CHECK: %[[ARG0:[a-zA-Z0-9$._-]+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x2xf32>
  // CHECK: %[[ARG1:[a-zA-Z0-9$._-]+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<2x2xf32>
  // CHECK: %[[ARG2:[a-zA-Z0-9$._-]+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x2xf32>
  func @element_wise() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0 : tensor<2x2xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]]
    %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %1 {
    // CHECK: ^{{[a-zA-Z0-9$._-]+}}
    // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]: f32
    // CHECK-SAME: %[[ARG4:[a-zA-Z0-9$._-]+]]: f32
    // CHECK-SAME: %[[ARG5:[a-zA-Z0-9$._-]+]]: f32
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      // CHECK: addf %[[ARG3]], %[[ARG4]]
      %3 = addf %arg3, %arg4 : f32
      linalg.yield %3 : f32
    }: tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0 : tensor<2x2xf32>
    // CHECK: return
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK-LABEL: func @indexed_generic
  // CHECK: %[[ARG0:[a-zA-Z0-9$._-]+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x2xi32>
  // CHECK: %[[ARG1:[a-zA-Z0-9$._-]+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x2xi32>
  func @indexed_generic() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2xi32>
    // CHECK: linalg.indexed_generic
    // CHECK-SAME: %[[ARG0]], %[[ARG1]]
    %1 = linalg.indexed_generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} %0 {
    // CHECK: ^{{[a-zA-Z0-9$._-]+}}
    // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: index
    // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]: index
    // CHECK-SAME: %[[ARG4:[a-zA-Z0-9$._-]+]]: i32
    ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
      // CHECK: %[[A:.+]] = index_cast %[[ARG2]] : index to i32
      // CHECK: %[[B:.+]] = index_cast %[[ARG3]] : index to i32
      // CHECK: %[[C:.+]] = addi %[[ARG4]], %[[A]] : i32
      // CHECK: %[[D:.+]] = addi %[[C]], %[[B]] : i32
      // CHECK: linalg.yield %[[D]] : i32
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = addi %4, %3 : i32
      linalg.yield %5 : i32
    }: tensor<2x2xi32> -> tensor<2x2xi32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<2x2xi32>
    // CHECK: return
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
