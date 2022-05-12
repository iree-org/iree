// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-set-num-workgroups,builtin.module(func.func(iree-spirv-create-fast-slow-path,iree-spirv-tile,iree-spirv-vectorize))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 16], [0, 4, 1, 4], [0, 0, 0, 0, 1, 1, 4]]>
#translation = #iree_codegen.translation_info<SPIRVVectorize, workload_per_wg = [16, 4, 4]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_static_shape_f32 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @conv_static_shape_f32 layout(#executable_layout) {
      workgroup_size = [4: index, 4: index, 1: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @conv_static_shape_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c112 = arith.constant 112 : index
        %c16 = arith.constant 16 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x225x225x8xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:3x3x8x16xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1x112x112x16xf32>
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
              %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %23 = linalg.conv_2d_nhwc_hwcf {lowering_config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
                ins(%13, %15 : tensor<1x?x?x8xf32>, tensor<3x3x8x?xf32>)
                outs(%22 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %23, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %16, %17, %14], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x16xf32>
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @conv_static_shape_f32()

// No vector transfer write ops generated for the linalg.fill op: it's cancelled with read ops.
// CHECK-NOT: vector.transfer

// Check tiling loop along filter height/width and input channel
//      CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:     -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
//      CHECK:   scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:       -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
//      CHECK:     scf.for %{{.*}} = %c0 to %c8 step %c4
// CHECK-SAME:         -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)

// CHECK-COUNT-16: vector.fma

// CHECK-COUNT-3: scf.yield

// For linalg.conv_2d_nhwc_hwcf
// CHECK-COUNT-4: vector.transfer_write

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 16], [0, 1, 1, 4], [0, 0, 0, 0, 1, 1]]>
#translation = #iree_codegen.translation_info<SPIRVVectorize, workload_per_wg = [16, 4, 4]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @depthwise_conv_static_shape_f32 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @depthwise_conv_static_shape_f32 layout(#executable_layout) {
      workgroup_size = [4: index, 4: index, 4: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @depthwise_conv_static_shape_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c56 = arith.constant 56 : index
        %c96 = arith.constant 96 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x113x113x96xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:3x3x96xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1x56x56x96xf32>
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
              %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %23 = linalg.depthwise_conv_2d_nhwc_hwc {lowering_config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
                ins(%14, %15 : tensor<1x?x?x?xf32>, tensor<3x3x?xf32>)
                outs(%22 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %23, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %16, %17, %13], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x56x56x96xf32>
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @depthwise_conv_static_shape_f32()

// No vector transfer write ops generated for the linalg.fill op: it's cancelled with read ops.
// CHECK-NOT: vector.transfer

// check tiling loop along filter height/width and input channel
//      CHECK:    scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:        -> (vector<4xf32>)
//      CHECK:      scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:          -> (vector<4xf32>)

// CHECK: vector.fma

// CHECK-COUNT-2: scf.yield

// For linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: vector.transfer_write

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1, 4]]>
#translation = #iree_codegen.translation_info<SPIRVVectorize, workload_per_wg = [32, 4, 1]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>

hal.executable private @low_padded_conv {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @low_padded_conv layout(#executable_layout) {
      workgroup_size = [8: index, 2: index, 1: index],
      translation_info = #translation
    }
    builtin.module {
      func.func @low_padded_conv() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c112 = arith.constant 112 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x224x224x3xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:3x3x3x32xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x112x112x32xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
        %4 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %5 to %c112 step %6 {
          %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %7 to %c112 step %8 {
            %9 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %10 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %9 to %c32 step %10 {
              %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %16 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %13, %14, %15], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x112x112x32xf32> -> tensor<1x?x?x?xf32>
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %20 = tensor.extract_slice %4[0, %arg0, %arg1, %arg2] [1, %17, %18, %19] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x?x?x?xf32>
              %21 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%11, %arg0)
              %22 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%12, %arg1)
              %23 = affine.min affine_map<(d0) -> (d0 * 2, 224)>(%arg0)
              %24 = affine.min affine_map<(d0, d1) -> (d0 + d1 * 2, 224)>(%21, %arg0)
              %25 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%24, %23)
              %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%21, %24, %23)
              %27 = affine.min affine_map<(d0) -> (d0 * 2, 224)>(%arg1)
              %28 = affine.min affine_map<(d0, d1) -> (d0 + d1 * 2, 224)>(%22, %arg1)
              %29 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%28, %27)
              %30 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%22, %28, %27)
              %31 = flow.dispatch.tensor.load %0, offsets = [0, %23, %27, 0], sizes = [1, %25, %29, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x224x224x3xf32> -> tensor<1x?x?x3xf32>
              %32 = tensor.pad %31 low[0, 0, 0, 0] high[0, %26, %30, 0] {
              ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
                tensor.yield %cst : f32
              } : tensor<1x?x?x3xf32> to tensor<1x?x?x3xf32>
              %33 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %34 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 3, %33], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x3x32xf32> -> tensor<3x3x3x?xf32>
              %35 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg0)[%workgroup_size_z]
              %36 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg1)[%workgroup_size_y]
              %37 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %38 = linalg.init_tensor [1, %35, %36, %37] : tensor<1x?x?x?xf32>
              %39 = linalg.fill ins(%cst : f32) outs(%38 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %40 = linalg.conv_2d_nhwc_hwcf {lowering_config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%32, %34 : tensor<1x?x?x3xf32>, tensor<3x3x3x?xf32>) outs(%39 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %41 = linalg.generic {lowering_config = #config, indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %16 : tensor<1x?x?x?xf32>, tensor<1x?x?x?xf32>) outs(%20 : tensor<1x?x?x?xf32>) {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
                %42 = arith.subf %arg3, %arg4 : f32
                linalg.yield %42 : f32
              } -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %41, %3, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %17, %18, %19], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @low_padded_conv()

// Loop nest for workgroup tiling and distribution
// CHECK-COUNT-3: scf.for

// Switch between fast and slow path
//         CHECK: scf.if

// Fast path:
// Loop nest for thread tiling and reduction tiling
// CHECK-COUNT-4: scf.for
// Vector code
// CHECK-COUNT-6: vector.fma

//         CHECK: } else {

// Slow path:
// Loop nest for thread tiling and reduction tiling
// CHECK-COUNT-4: scf.for
//         CHECK: scf.if
//    CHECK-NEXT:   vector.transfer_read
//         CHECK: scf.if
//    CHECK-NEXT:   vector.transfer_read
// CHECK-COUNT-6: vector.fma

// -----

#config =  #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1]]>
#translation = #iree_codegen.translation_info<SPIRVVectorize, workload_per_wg = [32, 4, 1]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable private @low_high_padded_depthwise_conv {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @low_high_padded_depthwise_conv layout(#executable_layout) {
      workgroup_size = [8: index, 2: index, 1: index],
      translation_info = #translation
    }
    builtin.module {
      func.func @low_high_padded_depthwise_conv() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c112 = arith.constant 112 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x112x112x32xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:3x3x32xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:32xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
        %4 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %5 to %c112 step %6 {
          %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %7 to %c112 step %8 {
            %9 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %10 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %9 to %c32 step %10 {
              %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %12 = flow.dispatch.tensor.load %2, offsets = [%arg2], sizes = [%11], strides = [1] : !flow.dispatch.tensor<readonly:32xf32> -> tensor<?xf32>
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %18 = tensor.extract_slice %4[0, %arg0, %arg1, %arg2] [1, %15, %16, %17] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x?x?x?xf32>
              %19 = affine.min affine_map<(d0, d1) -> (d1 + 2, -d0 + 114)>(%arg0, %13)
              %20 = affine.min affine_map<(d0, d1) -> (d1 + 2, -d0 + 114)>(%arg1, %14)
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %22 = affine.max affine_map<(d0) -> (0, -d0 + 1)>(%arg0)
              %23 = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%arg0)
              %24 = affine.min affine_map<(d0) -> (d0, 112)>(%23)
              %25 = affine.max affine_map<(d0, d1) -> (d0 + d1 - 1, 0)>(%19, %arg0)
              %26 = affine.min affine_map<(d0) -> (d0, 112)>(%25)
              %27 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%26, %24)
              %28 = affine.apply affine_map<(d0, d1, d2, d3) -> (-d0 + d1 - d2 + d3)>(%22, %19, %26, %24)
              %29 = affine.max affine_map<(d0) -> (0, -d0 + 1)>(%arg1)
              %30 = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%arg1)
              %31 = affine.min affine_map<(d0) -> (d0, 112)>(%30)
              %32 = affine.max affine_map<(d0, d1) -> (d0 + d1 - 1, 0)>(%20, %arg1)
              %33 = affine.min affine_map<(d0) -> (d0, 112)>(%32)
              %34 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%33, %31)
              %35 = affine.apply affine_map<(d0, d1, d2, d3) -> (-d0 + d1 - d2 + d3)>(%29, %20, %33, %31)
              %36 = affine.min affine_map<(d0) -> (d0, 32)>(%arg2)
              %37 = affine.min affine_map<(d0, d1) -> (d0 + d1, 32)>(%arg2, %21)
              %38 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%37, %36)
              %39 = flow.dispatch.tensor.load %0, offsets = [0, %24, %31, %36], sizes = [1, %27, %34, %38], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x112x112x32xf32> -> tensor<1x?x?x?xf32>
              %40 = tensor.pad %39 low[0, %22, %29, 0] high[0, %28, %35, 0] {
              ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
                tensor.yield %cst : f32
              } : tensor<1x?x?x?xf32> to tensor<1x?x?x?xf32>
              %41 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %42 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [3, 3, %41], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x32xf32> -> tensor<3x3x?xf32>
              %43 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg0)[%workgroup_size_z]
              %44 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg1)[%workgroup_size_y]
              %45 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %46 = linalg.init_tensor [1, %43, %44, %45] : tensor<1x?x?x?xf32>
              %47 = linalg.fill ins(%cst : f32) outs(%46 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %48 = linalg.depthwise_conv_2d_nhwc_hwc {lowering_config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%40, %42 : tensor<1x?x?x?xf32>, tensor<3x3x?xf32>) outs(%47 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %49 = linalg.generic {lowering_config = #config, indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %48 : tensor<?xf32>, tensor<1x?x?x?xf32>) outs(%18 : tensor<1x?x?x?xf32>) {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
                %50 = arith.addf %arg3, %arg4 : f32
                linalg.yield %50 : f32
              } -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %49, %3, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %15, %16, %17], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
      }
    }
  }
  return
      }
    }
  }
}

// CHECK-LABEL: func.func @low_high_padded_depthwise_conv()

// Loop nest for workgroup tiling and distribution
// CHECK-COUNT-3: scf.for

// Switch between fast and slow path
//         CHECK: scf.if

// Fast path:
// Loop nest for thread tiling and reduction tiling
// CHECK-COUNT-4: scf.for
// Vector code
// CHECK-COUNT-2: vector.transfer_read
//         CHECK: vector.fma
//         CHECK: vector.transfer_read
//         CHECK: vector.fma

//         CHECK: } else {

// Slow path:
// Loop nest for thread tiling and reduction tiling
// CHECK-COUNT-4: scf.for
//         CHECK: scf.if
//    CHECK-NEXT:   vector.transfer_read
//         CHECK: scf.if
//    CHECK-NEXT:   vector.transfer_read
//         CHECK: vector.transfer_read
// CHECK-COUNT-2: vector.fma
