// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-set-num-workgroups,builtin.module(builtin.func(iree-spirv-tile,iree-spirv-vectorize))))' %s | IreeFileCheck %s

#config = #iree_codegen.lowering.config<tile_sizes = [[0, 4, 4, 16], [0, 4, 1, 4], [0, 0, 0, 0, 1, 1, 4]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [16, 4, 4]>

hal.executable private @conv_static_shape_f32 {
  hal.interface public @io {
    hal.interface.binding public @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @conv_static_shape_f32 attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [4: index, 4: index, 1: index],
      translation.info = #translation
    }
    builtin.module  {
      func @conv_static_shape_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c112 = arith.constant 112 : index
        %c16 = arith.constant 16 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:1x225x225x8xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:3x3x8x16xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:1x112x112x16xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c16 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg1)[%workgroup_size_y]
              %13 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, 0], sizes = [1, %10, %12, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x8xf32> -> tensor<1x?x?x8xf32>
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg2)[%workgroup_size_x]
              %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, %14], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x8x16xf32> -> tensor<3x3x8x?xf32>
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %18 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg0)[%workgroup_size_z]
              %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg1)[%workgroup_size_y]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 16, s0)>(%arg2)[%workgroup_size_x]
              %21 = linalg.init_tensor [1, %18, %19, %20] : tensor<1x?x?x?xf32>
              %22 = linalg.fill(%cst, %21) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %23 = linalg.conv_2d_nhwc_hwcf {lowering.config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
                ins(%13, %15 : tensor<1x?x?x8xf32>, tensor<3x3x8x?xf32>)
                outs(%22 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %23, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %16, %17, %14], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x16xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @arg0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding public @arg1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding public @ret0, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

// CHECK-LABEL: func @conv_static_shape_f32()

// No vector transfer write ops generated for the linalg.fill op: it's cancelled with read ops.
// CHECK-NOT: vector.transfer

// Check tiling loop along filter height/width and input channel
//      CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:     -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>)
//      CHECK:   scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:       -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>)
//      CHECK:     scf.for %{{.*}} = %c0 to %c8 step %c4
// CHECK-SAME:         -> (vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>, vector<1x4xf32>)

// CHECK-COUNT-16: vector.fma

// CHECK-COUNT-3: scf.yield

// For linalg.conv_2d_nhwc_hwcf
// CHECK-COUNT-4: vector.transfer_write

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[0, 4, 4, 16], [0, 1, 1, 4], [0, 0, 0, 0, 1, 1]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [16, 4, 4]>

hal.executable private @depthwise_conv_static_shape_f32 {
  hal.interface public @io {
    hal.interface.binding public @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @depthwise_conv_static_shape_f32 attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [4: index, 4: index, 4: index],
      translation.info = #translation
    }
    builtin.module  {
      func @depthwise_conv_static_shape_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c56 = arith.constant 56 : index
        %c96 = arith.constant 96 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:1x113x113x96xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:3x3x96xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:1x56x56x96xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c56 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c56 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c96 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 113)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 113)>(%arg1)[%workgroup_size_y]
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg2)[%workgroup_size_x]
              %14 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, %arg2], sizes = [1, %10, %12, %13], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x113x113x96xf32> -> tensor<1x?x?x?xf32>
              %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [3, 3, %13], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x96xf32> -> tensor<3x3x?xf32>
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 56)>(%arg0)[%workgroup_size_z]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 56)>(%arg1)[%workgroup_size_y]
              %18 = affine.min affine_map<(d0)[s0] -> (-d0 + 56, s0)>(%arg0)[%workgroup_size_z]
              %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 56, s0)>(%arg1)[%workgroup_size_y]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 96, s0)>(%arg2)[%workgroup_size_x]
              %21 = linalg.init_tensor [1, %18, %19, %20] : tensor<1x?x?x?xf32>
              %22 = linalg.fill(%cst, %21) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %23 = linalg.depthwise_conv_2d_nhwc_hwc {lowering.config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
                ins(%14, %15 : tensor<1x?x?x?xf32>, tensor<3x3x?xf32>)
                outs(%22 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %23, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %16, %17, %13], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x56x56x96xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @arg0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding public @arg1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding public @ret0, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

// CHECK-LABEL: func @depthwise_conv_static_shape_f32()

// No vector transfer write ops generated for the linalg.fill op: it's cancelled with read ops.
// CHECK-NOT: vector.transfer

// check tiling loop along filter height/width and input channel
//      CHECK:    scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:        -> (vector<1x1x1x4xf32>)
//      CHECK:      scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:          -> (vector<1x1x1x4xf32>)

// CHECK: vector.fma

// CHECK-COUNT-2: scf.yield

// For linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: vector.transfer_write
