// RUN: iree-opt -split-input-file -iree-codegen-split-dispatch-function -verify-diagnostics %s | IreeFileCheck %s

// CHECK: module attributes {vkspv.entry_point_schedule = ["kernel_dispatch_0", "kernel_dispatch_1"]}
module {
  // CHECK: func @kernel_dispatch_1()
  // CHECK:   %[[DIM:.+]] = hal.interface.load.constant
  // CHECK:   %[[SHAPE1:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK:   %[[SHAPE2:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK:   %[[IN1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x2x2x512xf32>
  // CHECK:   %[[TS1:.+]] = shapex.tie_shape %[[IN1]], %[[SHAPE1]]
  // CHECK:   %[[IN2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
  // CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
  // CHECK:   %[[TS2:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE2]]
  // CHECK:   linalg.conv(%[[IN2]], %[[TS1]], %[[TS2]])
  // CHECK:   return

  // CHECK: func @kernel_dispatch_0()
  // CHECK:   %[[ZERO:.+]] = constant
  // CHECK:   %[[DIM:.+]] = hal.interface.load.constant
  // CHECK:   %[[SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
  // CHECK:   %[[TS:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE]]
  // CHECK:   linalg.fill(%[[TS]], %[[ZERO]])
  // CHECK:   return

  func @kernel() {
    %cst = constant 0.000000e+00 : f32
    %dim = hal.interface.load.constant offset = 0 : index
    %shape1 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,2,2,512]>
    %shape2 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,1,1,512]>
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x2x2x512xf32>
    %ts1 = shapex.tie_shape %0, %shape1 : memref<?x2x2x512xf32>, !shapex.ranked_shape<[?,2,2,512]>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
    %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
    %ts2 = shapex.tie_shape %2, %shape2 : memref<?x1x1x512xf32>, !shapex.ranked_shape<[?,1,1,512]>
    linalg.fill(%ts2, %cst) : memref<?x1x1x512xf32>, f32
    linalg.conv(%1, %ts1, %ts2) {dilations = [1, 1], padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, strides = [2, 2]} : memref<3x3x512x1xf32>, memref<?x2x2x512xf32>, memref<?x1x1x512xf32>
    return
  }
  hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// -----

// CHECK: module attributes {vkspv.entry_point_schedule = ["kernel_dispatch_0", "kernel_dispatch_1", "kernel_dispatch_2"]}
module {
  // CHECK: func @kernel_dispatch_2()
  // CHECK:   %[[DIM:.+]] = hal.interface.load.constant
  // CHECK:   %[[SHAPE1:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK:   %[[SHAPE2:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK:   %[[IN1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x2x2x512xf32>
  // CHECK:   %[[TS1:.+]] = shapex.tie_shape %[[IN1]], %[[SHAPE1]]
  // CHECK:   %[[IN2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
  // CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
  // CHECK:   %[[TS2:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE2]]
  // CHECK:   linalg.conv(%[[IN2]], %[[TS1]], %[[TS2]])
  // CHECK:   return

  // CHECK: func @kernel_dispatch_1() {
  // CHECK:   %[[C0:.+]] = constant 0 : index
  // CHECK:   %[[C1:.+]] = constant 1 : index
  // CHECK:   scf.parallel (%{{.*}}) = (%[[C0]]) to (%[[C1]]) step (%[[C1]])
  // CHECK:     scf.yield
  // CHECK:   return

  // CHECK: func @kernel_dispatch_0()
  // CHECK:   %[[ZERO:.+]] = constant
  // CHECK:   %[[DIM:.+]] = hal.interface.load.constant
  // CHECK:   %[[SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
  // CHECK:   %[[TS:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE]]
  // CHECK:   linalg.fill(%[[TS]], %[[ZERO]])
  // CHECK:   return

  func @kernel() {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %dim = hal.interface.load.constant offset = 0 : index
    %shape1 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,2,2,512]>
    %shape2 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,1,1,512]>
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x2x2x512xf32>
    %ts1 = shapex.tie_shape %0, %shape1 : memref<?x2x2x512xf32>, !shapex.ranked_shape<[?,2,2,512]>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
    %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
    %ts2 = shapex.tie_shape %2, %shape2 : memref<?x1x1x512xf32>, !shapex.ranked_shape<[?,1,1,512]>
    linalg.fill(%ts2, %cst) : memref<?x1x1x512xf32>, f32
    scf.parallel (%iv) = (%c0) to (%c1) step (%c1) {
      scf.yield
    }
    linalg.conv(%1, %ts1, %ts2) {dilations = [1, 1], padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, strides = [2, 2]} : memref<3x3x512x1xf32>, memref<?x2x2x512xf32>, memref<?x1x1x512xf32>
    return
  }
  hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}


// -----

// Nothing to do if there is just one Linalg op.

// CHECK-NOT: vkspv.entry_point_schedule
module {
  // CHECK-LABEL: @kernel()
  func @kernel() {
    %cst = constant 0.000000e+00 : f32
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x2x2x512xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
    %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x1x1x512xf32>
    linalg.conv(%1, %0, %2) {dilations = [1, 1], padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, strides = [2, 2]} : memref<3x3x512x1xf32>, memref<1x2x2x512xf32>, memref<1x1x1x512xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// -----

// Do not split when Linalg and non-Linalg ops are interleaving each other.

module {
  // expected-error @+1 {{cannot separate Linalg/Parallel ops into multiple kernels}}
  func @kernel() {
    %cst = constant 0.000000e+00 : f32
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x2x2x512xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
    %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x1x1x512xf32>
    linalg.fill(%2, %cst) : memref<1x1x1x512xf32>, f32
    "some_op"() : () -> ()
    linalg.conv(%1, %0, %2) {dilations = [1, 1], padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, strides = [2, 2]} : memref<3x3x512x1xf32>, memref<1x2x2x512xf32>, memref<1x1x1x512xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0 * 12 + d1 + 53)>

module {
  func @subview_interleaved() {
    %cst = constant 0.000000e+00 : f32
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<18x12xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<12x4xf32>
    linalg.fill(%0, %cst) : memref<18x12xf32>, f32
    %2 = subview %0[4, 5] [18, 12] [1, 1]  : memref<18x12xf32> to memref<18x12xf32, #map0>
    linalg.copy(%1, %2) : memref<12x4xf32>, memref<18x12xf32, #map0>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

//      CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 + 53)>
//      CHECK: module attributes {vkspv.entry_point_schedule =
// CHECK-SAME:   ["subview_interleaved_dispatch_0",
// CHECK-SAME:    "subview_interleaved_dispatch_1"]}
//      CHECK: func @subview_interleaved_dispatch_1()
//  CHECK-DAG:   %[[DST:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<18x12xf32>
//  CHECK-DAG:   %[[SRC:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<12x4xf32>
//      CHECK:   %[[SUB:.+]] = subview %[[DST]][4, 5] [18, 12] [1, 1]  : memref<18x12xf32> to memref<18x12xf32, #[[MAP0]]>
//      CHECK:   linalg.copy(%[[SRC]], %[[SUB]]) : memref<12x4xf32>, memref<18x12xf32, #[[MAP0]]>
//      CHECK:   return
//      CHECK: func @subview_interleaved_dispatch_0()
//      CHECK:   %[[CST:.+]] = constant
//      CHECK:   %[[DST2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<18x12xf32>
//      CHECK:   linalg.fill(%[[DST2]], %[[CST]]) : memref<18x12xf32>, f32
//      CHECK:   return

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

module {
  func @reshape_interleaved() {
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1} : memref<1x2x4xf32>
    %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x4xf32>
    linalg.generic {args_in = 1 : i64, args_out = 1 : i64,
                    indexing_maps = [#map0, #map0],
                    iterator_types = ["parallel", "parallel"]} %2, %0 {
    ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
      %4 = tanh %arg0 : f32
      linalg.yield %4 : f32
    }: memref<2x4xf32>, memref<2x4xf32>
    %3 = linalg.reshape %0 [#map1, #map2] : memref<2x4xf32> into memref<1x2x4xf32>
    linalg.copy(%3, %1) : memref<1x2x4xf32>, memref<1x2x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: module attributes {vkspv.entry_point_schedule =
// CHECK-SAME:   ["reshape_interleaved_dispatch_0",
// CHECK-SAME:    "reshape_interleaved_dispatch_1"]}
//      CHECK: func @reshape_interleaved_dispatch_1()
//      CHECK:   %[[SRC1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4xf32>
//      CHECK:   %[[DST:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1} : memref<1x2x4xf32>
//      CHECK:   %[[SRC2:.+]] = linalg.reshape %[[SRC1]] [#[[MAP0]], #[[MAP1]]] : memref<2x4xf32> into memref<1x2x4xf32>
//      CHECK:   linalg.copy(%[[SRC2]], %[[DST]]) : memref<1x2x4xf32>, memref<1x2x4xf32>
//      CHECK:   return
//      CHECK: func @reshape_interleaved_dispatch_0()
//      CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4xf32>
//      CHECK:   %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x4xf32>
//      CHECK:   linalg.generic {{.*}} %[[IN]], %[[OUT]]
