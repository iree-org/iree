// RUN: iree-opt -allow-unregistered-dialect -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-split-dispatch-function))" -verify-diagnostics %s | IreeFileCheck %s

hal.executable @kernel_fusable_fill_conv1d_ops attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel_fusable_fill_conv1d_ops attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x3x512xf32>, !flow.dispatch.input<3x512x1xf32>,
        !flow.dispatch.output<?x1x512xf32>) -> ()}
    module {
      //     CHECK: func @kernel_fusable_fill_conv1d_ops
      //     CHECK:   linalg.fill
      // CHECK-NOT:   return
      //     CHECK:   linalg.conv_1d_input_nwc_filter_wcf
      //     CHECK:   return

      func @kernel_fusable_fill_conv1d_ops() {
        %cst = constant 0.000000e+00 : f32
        %dim = hal.interface.load.constant offset = 0 : index
        %shape1 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,3,512]>
        %shape2 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,1,512]>
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x3x512xf32>
        %ts1 = shapex.tie_shape %0, %shape1 : memref<?x3x512xf32>, !shapex.ranked_shape<[?,3,512]>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x512x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x512xf32>
        %ts2 = shapex.tie_shape %2, %shape2 : memref<?x1x512xf32>, !shapex.ranked_shape<[?,1,512]>
        linalg.fill(%ts2, %cst) : memref<?x1x512xf32>, f32
        linalg.conv_1d_input_nwc_filter_wcf {
          dilations = dense<1> : tensor<1xi64>,
          strides = dense<2> : tensor<1xi64>}
           ins(%ts1, %1 : memref<?x3x512xf32>, memref<3x512x1xf32>)
          outs(%ts2 : memref<?x1x512xf32>)
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
// -----

hal.executable @kernel_fusable_fill_conv2d_ops attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel_fusable_fill_conv2d_ops attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x3x3x512xf32>, !flow.dispatch.input<3x3x512x1xf32>,
        !flow.dispatch.output<?x1x1x512xf32>) -> ()}
    module {
      //     CHECK: func @kernel_fusable_fill_conv2d_ops
      //     CHECK:   linalg.fill
      // CHECK-NOT:   return
      //     CHECK:   linalg.conv_2d_input_nhwc_filter_hwcf
      //     CHECK:   return

      func @kernel_fusable_fill_conv2d_ops() {
        %cst = constant 0.000000e+00 : f32
        %dim = hal.interface.load.constant offset = 0 : index
        %shape1 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,3,3,512]>
        %shape2 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,1,1,512]>
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x3x3x512xf32>
        %ts1 = shapex.tie_shape %0, %shape1 : memref<?x3x3x512xf32>, !shapex.ranked_shape<[?,3,3,512]>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
        %ts2 = shapex.tie_shape %2, %shape2 : memref<?x1x1x512xf32>, !shapex.ranked_shape<[?,1,1,512]>
        linalg.fill(%ts2, %cst) : memref<?x1x1x512xf32>, f32
        linalg.conv_2d_input_nhwc_filter_hwcf {
          dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
           ins(%ts1, %1 : memref<?x3x3x512xf32>, memref<3x3x512x1xf32>)
          outs(%ts2 : memref<?x1x1x512xf32>)
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

hal.executable @kernel_fusable_fill_conv3d_ops attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel_fusable_fill_conv3d_ops attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x3x3x3x512xf32>, !flow.dispatch.input<3x3x3x512x1xf32>,
        !flow.dispatch.output<?x1x1x1x512xf32>) -> ()}
    module {
      //     CHECK: func @kernel_fusable_fill_conv3d_ops
      //     CHECK:   linalg.fill
      // CHECK-NOT:   return
      //     CHECK:   linalg.conv_3d_input_ndhwc_filter_dhwcf
      //     CHECK:   return

      func @kernel_fusable_fill_conv3d_ops() {
        %cst = constant 0.000000e+00 : f32
        %dim = hal.interface.load.constant offset = 0 : index
        %shape1 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,3,3,3,512]>
        %shape2 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,1,1,1,512]>
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x3x3x3x512xf32>
        %ts1 = shapex.tie_shape %0, %shape1 : memref<?x3x3x3x512xf32>, !shapex.ranked_shape<[?,3,3,3,512]>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x3x512x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x1x512xf32>
        %ts2 = shapex.tie_shape %2, %shape2 : memref<?x1x1x1x512xf32>, !shapex.ranked_shape<[?,1,1,1,512]>
        linalg.fill(%ts2, %cst) : memref<?x1x1x1x512xf32>, f32
        linalg.conv_3d_input_ndhwc_filter_dhwcf {
          dilations = dense<1> : tensor<3xi64>,
          strides = dense<2> : tensor<3xi64>}
           ins(%ts1, %1 : memref<?x3x3x3x512xf32>, memref<3x3x3x512x1xf32>)
          outs(%ts2 : memref<?x1x1x1x512xf32>)
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

hal.executable @kernel_fusable_fill_matmul_ops attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel_fusable_fill_matmul_ops attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x512xf32>, !flow.dispatch.input<512x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
    module {
      //     CHECK: func @kernel_fusable_fill_matmul_ops
      //     CHECK:   linalg.fill
      // CHECK-NOT:   return
      //     CHECK:   linalg.matmul
      //     CHECK:   return

      func @kernel_fusable_fill_matmul_ops() {
        %cst = constant 0.000000e+00 : f32
        %dimM = hal.interface.load.constant offset = 0 : index
        %dimN = hal.interface.load.constant offset = 1 : index
        %shape1 = shapex.make_ranked_shape %dimM : (index) -> !shapex.ranked_shape<[?,512]>
        %shape2 = shapex.make_ranked_shape %dimN : (index) -> !shapex.ranked_shape<[512,?]>
        %shape3 = shapex.make_ranked_shape %dimM, %dimN : (index, index) -> !shapex.ranked_shape<[?,?]>
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x512xf32>
        %ts1 = shapex.tie_shape %0, %shape1 : memref<?x512xf32>, !shapex.ranked_shape<[?,512]>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<512x?xf32>
        %ts2 = shapex.tie_shape %1, %shape2 : memref<512x?xf32>, !shapex.ranked_shape<[512, ?]>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x?xf32>
        %ts3 = shapex.tie_shape %2, %shape3 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
        linalg.fill(%ts3, %cst) : memref<?x?xf32>, f32
        linalg.matmul ins(%ts1, %ts2 : memref<?x512xf32>, memref<512x?xf32>)
                      outs(%ts3 : memref<?x?xf32>)
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

hal.executable @kernel_fusable_pooling attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel_fusable_pooling attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?x?x?xf32>,
        !flow.dispatch.output<?x?x?x?xf32>) -> ()}
    module {
      //     CHECK: func @kernel_fusable_pooling()
      //     CHECK:   linalg.fill
      // CHECK-NOT:   return
      //     CHECK:   linalg.pooling_nhwc_sum
      //     CHECK:   return
      func @kernel_fusable_pooling() {
        %cst = constant 0.000000e+00 : f32
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x?xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<?x?x?x?xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x?x?x?xf32>
        linalg.fill(%2, %cst) : memref<?x?x?x?xf32>, f32
        linalg.pooling_nhwc_sum
          {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
          ins(%1, %0: memref<?x?x?x?xf32>, memref<?x?xf32>)
          outs(%2: memref<?x?x?x?xf32>)
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

hal.executable @kernel attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x3x3x512xf32>, !flow.dispatch.input<3x3x512x1xf32>,
        !flow.dispatch.output<?x1x1x512xf32>) -> ()}
    // CHECK: hal.executable.entry_point @kernel_dispatch_0
    // CHECK: hal.executable.entry_point @kernel_dispatch_1
    // CHECK: module attributes {hal.entry_point_schedule = [@kernel_dispatch_0, @kernel_dispatch_1]}
    module {
      // CHECK: func @kernel_dispatch_1()
      // CHECK:   %[[ZERO:.+]] = constant
      // CHECK:   %[[DIM:.+]] = hal.interface.load.constant
      // CHECK:   %[[SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
      // CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
      // CHECK:   %[[TS:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE]]
      // CHECK:   linalg.fill(%[[TS]], %[[ZERO]])
      // CHECK:   return

      // CHECK: func @kernel_dispatch_0()
      // CHECK:   %[[DIM:.+]] = hal.interface.load.constant
      // CHECK:   %[[SHAPE1:.+]] = shapex.make_ranked_shape %[[DIM]]
      // CHECK:   %[[SHAPE2:.+]] = shapex.make_ranked_shape %[[DIM]]
      // CHECK:   %[[IN1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x3x3x512xf32>
      // CHECK:   %[[TS1:.+]] = shapex.tie_shape %[[IN1]], %[[SHAPE1]]
      // CHECK:   %[[IN2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
      // CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
      // CHECK:   %[[TS2:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE2]]
      // CHECK:   linalg.conv_2d_input_nhwc_filter_hwcf
      // CHECK-SAME: ins(%[[TS1]], %[[IN2]] : memref<?x3x3x512xf32>, memref<3x3x512x1xf32>)
      // CHECK-SAME: outs(%[[TS2]] : memref<?x1x1x512xf32>)
      // CHECK:   return

      func @kernel() {
        %cst = constant 0.000000e+00 : f32
        %dim = hal.interface.load.constant offset = 0 : index
        %shape1 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,3,3,512]>
        %shape2 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,1,1,512]>
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x3x3x512xf32>
        %ts1 = shapex.tie_shape %0, %shape1 : memref<?x3x3x512xf32>, !shapex.ranked_shape<[?,3,3,512]>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
        %ts2 = shapex.tie_shape %2, %shape2 : memref<?x1x1x512xf32>, !shapex.ranked_shape<[?,1,1,512]>
        linalg.conv_2d_input_nhwc_filter_hwcf {
          dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
           ins(%ts1, %1 : memref<?x3x3x512xf32>, memref<3x3x512x1xf32>)
          outs(%ts2 : memref<?x1x1x512xf32>)
        linalg.fill(%ts2, %cst) : memref<?x1x1x512xf32>, f32
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

hal.executable @kernel attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x3x3x512xf32>, !flow.dispatch.input<3x3x512x1xf32>,
        !flow.dispatch.output<?x1x1x512xf32>) -> ()}
    // CHECK: hal.executable.entry_point @kernel_dispatch_0
    // CHECK: hal.executable.entry_point @kernel_dispatch_1
    // CHECK: hal.executable.entry_point @kernel_dispatch_2
    // CHECK: module attributes {hal.entry_point_schedule = [@kernel_dispatch_0, @kernel_dispatch_1, @kernel_dispatch_2]}
    module {
    //      CHECK: func @kernel_dispatch_2()
    //      CHECK:   %[[DIM:.+]] = hal.interface.load.constant
    //      CHECK:   %[[SHAPE1:.+]] = shapex.make_ranked_shape %[[DIM]]
    //      CHECK:   %[[SHAPE2:.+]] = shapex.make_ranked_shape %[[DIM]]
    //      CHECK:   %[[IN1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x3x3x512xf32>
    //      CHECK:   %[[TS1:.+]] = shapex.tie_shape %[[IN1]], %[[SHAPE1]]
    //      CHECK:   %[[IN2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
    //      CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
    //      CHECK:   %[[TS2:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE2]]
    //      CHECK:   linalg.conv_2d_input_nhwc_filter_hwcf
    // CHECK-SAME:     ins(%[[TS1]], %[[IN2]] : memref<?x3x3x512xf32>, memref<3x3x512x1xf32>)
    // CHECK-SAME:     outs(%[[TS2]] : memref<?x1x1x512xf32>)
    //      CHECK:   return

    //      CHECK: func @kernel_dispatch_1()
    //      CHECK:   %[[C0:.+]] = constant 0 : index
    //      CHECK:   %[[C1:.+]] = constant 1 : index
    //      CHECK:   scf.parallel (%{{.*}}) = (%[[C0]]) to (%[[C1]]) step (%[[C1]])
    //      CHECK:     scf.yield
    //      CHECK:   return

    //      CHECK: func @kernel_dispatch_0()
    //      CHECK:   %[[ZERO:.+]] = constant
    //      CHECK:   %[[DIM:.+]] = hal.interface.load.constant
    //      CHECK:   %[[SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
    //      CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
    //      CHECK:   %[[TS:.+]] = shapex.tie_shape %[[OUT]], %[[SHAPE]]
    //      CHECK:   linalg.fill(%[[TS]], %[[ZERO]])
    //      CHECK:   return

      func @kernel() {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %dim = hal.interface.load.constant offset = 0 : index
        %shape1 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,3,3,512]>
        %shape2 = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,1,1,512]>
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x3x3x512xf32>
        %ts1 = shapex.tie_shape %0, %shape1 : memref<?x3x3x512xf32>, !shapex.ranked_shape<[?,3,3,512]>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x1x1x512xf32>
        %ts2 = shapex.tie_shape %2, %shape2 : memref<?x1x1x512xf32>, !shapex.ranked_shape<[?,1,1,512]>
        linalg.fill(%ts2, %cst) : memref<?x1x1x512xf32>, f32
        scf.parallel (%iv) = (%c0) to (%c1) step (%c1) {
          scf.yield
        }
        linalg.conv_2d_input_nhwc_filter_hwcf {
          dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
           ins(%ts1, %1 : memref<?x3x3x512xf32>, memref<3x3x512x1xf32>)
          outs(%ts2 : memref<?x1x1x512xf32>)
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

// Nothing to do if there is just one Linalg op.

hal.executable @kernel attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<1x3x3x512xf32>, !flow.dispatch.input<3x3x512x1xf32>,
        !flow.dispatch.output<1x1x1x512xf32>) -> ()}
    // CHECK-NOT: hal.entry_point_schedule
    module {
      // CHECK-LABEL: @kernel()
      func @kernel() attributes {hal.num_workgroups_fn = @kernel__num_workgroups__} {
        %cst = constant 0.000000e+00 : f32
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x3x3x512xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x1x1x512xf32>
        linalg.conv_2d_input_nhwc_filter_hwcf {
          dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
           ins(%0, %1 : memref<1x3x3x512xf32>, memref<3x3x512x1xf32>)
          outs(%2 : memref<1x1x1x512xf32>)
        return
      }
      // CHECK-LABEL: @kernel__num_workgroups__
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}



// -----

// Do not split when Linalg and non-Linalg ops are interleaving each other.

hal.executable @kernel attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x3x512xf32>, !flow.dispatch.input<3x512x1xf32>,
        !flow.dispatch.output<?x1x512xf32>) -> ()}
    module {
      // expected-error @+1 {{cannot separate Linalg/Parallel ops into multiple kernels}}
      func @kernel() {
        %cst = constant 0.000000e+00 : f32
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x3x3x512xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x512x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x1x1x512xf32>
        linalg.fill(%2, %cst) : memref<1x1x1x512xf32>, f32
        "some_op"() : () -> ()
        linalg.conv_2d_input_nhwc_filter_hwcf {
          dilations = dense<1> : tensor<2xi64>,
          strides = dense<2> : tensor<2xi64>}
           ins(%0, %1 : memref<1x3x3x512xf32>, memref<3x3x512x1xf32>)
          outs(%2 : memref<1x1x1x512xf32>)
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----
#map0 = affine_map<(d0, d1) -> (d0 * 12 + d1 + 53)>

hal.executable @subview_interleaved attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @subview_interleaved attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<18x12xf32>, !flow.dispatch.output<12x4xf32>) -> ()}
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
  }
}

//      CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 + 53)>
//  CHECK-DAG: hal.executable.entry_point @subview_interleaved_dispatch_0
//  CHECK-DAG: hal.executable.entry_point @subview_interleaved_dispatch_1
//      CHECK: module attributes {hal.entry_point_schedule =
// CHECK-SAME:   [@subview_interleaved_dispatch_0, @subview_interleaved_dispatch_1]}
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

hal.executable @reshape_interleaved attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @reshape_interleaved attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<2x4xf32>, !flow.dispatch.output<1x2x4xf32>,
        !flow.dispatch.output<2x4xf32>) -> ()}
    module {
      func @reshape_interleaved() {
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1} : memref<1x2x4xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x4xf32>
        linalg.generic {indexing_maps = [#map0, #map0],
                        iterator_types = ["parallel", "parallel"]}
          ins(%2 : memref<2x4xf32>)
         outs(%0 : memref<2x4xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %4 = math.tanh %arg0 : f32
          linalg.yield %4 : f32
        }
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
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: hal.executable.entry_point @reshape_interleaved_dispatch_0
//  CHECK-DAG: hal.executable.entry_point @reshape_interleaved_dispatch_1
//      CHECK: module attributes {hal.entry_point_schedule =
// CHECK-SAME:   [@reshape_interleaved_dispatch_0, @reshape_interleaved_dispatch_1]}
//      CHECK: func @reshape_interleaved_dispatch_1()
//      CHECK:   %[[SRC1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4xf32>
//      CHECK:   %[[DST:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1} : memref<1x2x4xf32>
//      CHECK:   %[[SRC2:.+]] = linalg.reshape %[[SRC1]] [#[[MAP0]], #[[MAP1]]] : memref<2x4xf32> into memref<1x2x4xf32>
//      CHECK:   linalg.copy(%[[SRC2]], %[[DST]]) : memref<1x2x4xf32>, memref<1x2x4xf32>
//      CHECK:   return
//      CHECK: func @reshape_interleaved_dispatch_0()
//      CHECK:   %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4xf32>
//      CHECK:   %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x4xf32>
//      CHECK:   linalg.generic
// CHECK-SAME:     ins(%[[IN]] :
// CHECK-SAME:    outs(%[[OUT]] :

// -----

hal.executable @predict_ex_dispatch_0 attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @predict_ex_dispatch_0 attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<1x512x1xf32>, !flow.dispatch.input<4x8x16xf32>,
        !flow.dispatch.output<4x8x16xf32>, !flow.dispatch.output<4x8x16xf32>) -> ()}
    module {
      func @predict_ex_dispatch_0() {
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x512x1xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1} : memref<4x8x16xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x512x1xf32>
        linalg.copy(%2, %0) : memref<1x512x1xf32>, memref<1x512x1xf32>
        %3 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<4x8x16xf32>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (-d0 + 3, d1, d2)>,
                                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                        iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%3 : memref<4x8x16xf32>)
         outs(%1 : memref<4x8x16xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          linalg.yield %arg0 : f32
        }
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//  CHECK-DAG: hal.executable.entry_point @predict_ex_dispatch_0_dispatch_0
//  CHECK-DAG: hal.executable.entry_point @predict_ex_dispatch_0_dispatch_1
//      CHECK: module attributes {hal.entry_point_schedule =
// CHECK-SAME:   [@predict_ex_dispatch_0_dispatch_0, @predict_ex_dispatch_0_dispatch_1]}
//      CHECK: func @predict_ex_dispatch_0_dispatch_1
// CHECK-NEXT:   iree.placeholder
// CHECK-SAME:     binding = @legacy_io::@ret1
// CHECK-NEXT:   iree.placeholder
// CHECK-SAME:     binding = @legacy_io::@arg1
// CHECK-NEXT:   linalg.generic
//      CHECK:     linalg.yield
//  CHECK-NOT:   linalg
//      CHECK:   return
//      CHECK: func @predict_ex_dispatch_0_dispatch_0
// CHECK-NEXT:   iree.placeholder
// CHECK-SAME:     binding = @legacy_io::@ret0
// CHECK-NEXT:   iree.placeholder
// CHECK-SAME:     binding = @legacy_io::@arg0
// CHECK-NEXT:   linalg.copy
//  CHECK-NOT:   linalg
//      CHECK:   return

// -----

hal.executable @kernel_fusable_fill_matmul_generic_ops attributes {sym_visiblity = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @kernel_fusable_fill_matmul_generic_ops attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x512xf32>, !flow.dispatch.input<512x?xf32>,
        !flow.dispatch.input<?x?xf32>, !flow.dispatch.output<?x?xf32>) -> ()}
    module {
      //     CHECK: func @kernel_fusable_fill_matmul_generic_ops
      //     CHECK:   linalg.fill
      // CHECK-NOT:   return
      //     CHECK:   linalg.matmul
      // CHECK-NOT:   return
      //     CHECK:   linalg.generic
      //     CHECK:   return

      func @kernel_fusable_fill_matmul_generic_ops() {
        %cst = constant 0.000000e+00 : f32
        %dimM = hal.interface.load.constant offset = 0 : index
        %dimN = hal.interface.load.constant offset = 1 : index
        %shape1 = shapex.make_ranked_shape %dimM : (index) -> !shapex.ranked_shape<[?,512]>
        %shape2 = shapex.make_ranked_shape %dimN : (index) -> !shapex.ranked_shape<[512,?]>
        %shape3 = shapex.make_ranked_shape %dimM, %dimN : (index, index) -> !shapex.ranked_shape<[?,?]>
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x512xf32>
        %ts0 = shapex.tie_shape %0, %shape1 : memref<?x512xf32>, !shapex.ranked_shape<[?,512]>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<512x?xf32>
        %ts1 = shapex.tie_shape %1, %shape2 : memref<512x?xf32>, !shapex.ranked_shape<[512, ?]>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg2} : memref<?x?xf32>
        %ts2 = shapex.tie_shape %2, %shape3 : memref<?x?xf32>, !shapex.ranked_shape<[?, ?]>
        %3 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x?xf32>
        %ts3 = shapex.tie_shape %3, %shape3 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
        %4 = alloc(%dimM, %dimN) : memref<?x?xf32>
        %ts4 = shapex.tie_shape %4, %shape3 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
        linalg.fill(%ts4, %cst) : memref<?x?xf32>, f32
        linalg.matmul ins(%ts0, %ts1 : memref<?x512xf32>, memref<512x?xf32>)
                      outs(%ts4 : memref<?x?xf32>)
        linalg.generic
          {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>,
                            affine_map<(d0, d1) -> (d0, d1)>],
           iterator_types = ["parallel", "parallel"]}
          ins(%ts2, %ts4 : memref<?x?xf32>, memref<?x?xf32>)
          outs(%ts3 : memref<?x?xf32>) {
          ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
            %5 = addf %arg0, %arg1 : f32
            linalg.yield %5 : f32
        }
        return
      }
      hal.interface @legacy_io attributes {push_constants = 1 : i32, sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @arg2, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
