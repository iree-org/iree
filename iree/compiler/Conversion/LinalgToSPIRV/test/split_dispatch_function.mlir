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
