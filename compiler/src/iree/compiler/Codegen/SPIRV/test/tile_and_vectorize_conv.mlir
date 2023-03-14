// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-create-fast-slow-path,iree-spirv-tile,canonicalize,cse,iree-spirv-vectorize,canonicalize,cse)))))' \
// RUN:   %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 16], [0, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @nhwc_conv_static_shape_f32 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.export @nhwc_conv_static_shape_f32 layout(#pipeline_layout) attributes {
      workgroup_size = [4: index, 4: index, 1: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @nhwc_conv_static_shape_f32() {
        %c112 = arith.constant 112 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_z]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_z]
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
            %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
            scf.for %arg2 = %7 to %c16 step %8 {
              %9 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>> -> tensor<1x4x4x16xf32>
              %10 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = flow.dispatch.tensor.load %0, offsets = [0, %10, %11, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>> -> tensor<1x9x9x8xf32>
              %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
              %14 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<2> : tensor<2xi64>}
                      ins(%12, %13 : tensor<1x9x9x8xf32>, tensor<3x3x8x16xf32>)
                      outs(%14 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              flow.dispatch.tensor.store %15, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : tensor<1x4x4x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @nhwc_conv_static_shape_f32()

// No vector transfer write ops generated for the linalg.fill op: initial values are forwarded to loops.
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

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 8, 32], [0, 1, 4, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @nhwc_nhwc_depthwise_conv_static_shape_f32 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.export @nhwc_nhwc_depthwise_conv_static_shape_f32 layout(#pipeline_layout) attributes {
      workgroup_size = [4: index, 4: index, 4: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @nhwc_nhwc_depthwise_conv_static_shape_f32() {
        %c56 = arith.constant 56 : index
        %c96 = arith.constant 96 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x113x113x96xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x96xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x56x56x96xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c56 step %workgroup_count_z {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
          scf.for %arg1 = %3 to %c56 step %4 {
            %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %5 to %c96 step %6 {
              %7 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 1, 8, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x56x56x96xf32>> -> tensor<1x1x8x32xf32>
              %8 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %10 = flow.dispatch.tensor.load %0, offsets = [0, %8, %9, %arg2], sizes = [1, 3, 17, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x113x113x96xf32>> -> tensor<1x3x17x32xf32>
              %11 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [3, 3, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x96xf32>> -> tensor<3x3x32xf32>
              %12 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x1x8x32xf32>) -> tensor<1x1x8x32xf32>
              %13 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<2> : tensor<2xi64>}
                      ins(%10, %11 : tensor<1x3x17x32xf32>, tensor<3x3x32xf32>) outs(%12 : tensor<1x1x8x32xf32>) -> tensor<1x1x8x32xf32>
              flow.dispatch.tensor.store %13, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 1, 8, 32], strides = [1, 1, 1, 1] : tensor<1x1x8x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x56x56x96xf32>>
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @nhwc_nhwc_depthwise_conv_static_shape_f32()

// No vector transfer write ops generated for the linalg.fill op: initial values are forwarded to loops.
// CHECK-NOT: vector.transfer

// check tiling loop along filter height/width and input channel
//      CHECK:    scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:        -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
//      CHECK:      scf.for %{{.+}} = %c0 to %c3 step %c1
// CHECK-SAME:          -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)

// CHECK-COUNT-5: vector.transfer_read
// CHECK-COUNT-4: vector.fma

// CHECK-COUNT-2: scf.yield

// For linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-COUNT-4: vector.transfer_write

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>

hal.executable private @low_padded_conv {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.export @low_padded_conv layout(#pipeline_layout) attributes {
      workgroup_size = [8: index, 2: index, 1: index],
      translation_info = #translation
    }
    builtin.module {
      func.func @low_padded_conv() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c112 = arith.constant 112 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x224x224x3xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
        %4 = tensor.empty() : tensor<1x112x112x32xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c112 step %workgroup_count_z {
          %7 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
          %8 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
          scf.for %arg1 = %7 to %c112 step %8 {
            %9 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %10 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %9 to %c32 step %10 {
              %12 = affine.min affine_map<(d0) -> (4, -d0 + 112)>(%arg1)[]
              %14 = affine.min affine_map<(d0) -> (4, -d0 + 112)>(%arg1)[]
              %15 = affine.min affine_map<(d0) -> (32, -d0 + 32)>(%arg2)[]
              %16 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 1, %14, %15], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>> -> tensor<1x1x?x?xf32>
              %18 = affine.min affine_map<(d0) -> (4, -d0 + 112)>(%arg1)[]
              %19 = affine.min affine_map<(d0) -> (32, -d0 + 32)>(%arg2)[]
              %20 = tensor.extract_slice %4[0, %arg0, %arg1, %arg2] [1, 1, %18, %19] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x1x?x?xf32>
              %21 = affine.min affine_map<(d0) -> (3, d0 * -2 + 225)>(%arg0)
              %22 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%12, %arg1)
              %23 = affine.min affine_map<(d0) -> (d0 * 2, 224)>(%arg0)
              %24 = affine.min affine_map<(d0, d1) -> (d0 + d1 * 2, 224)>(%21, %arg0)
              %25 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%24, %23)
              %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%21, %24, %23)
              %27 = affine.min affine_map<(d0) -> (d0 * 2, 224)>(%arg1)
              %28 = affine.min affine_map<(d0, d1) -> (d0 + d1 * 2, 224)>(%22, %arg1)
              %29 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%28, %27)
              %30 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%22, %28, %27)
              %31 = flow.dispatch.tensor.load %0, offsets = [0, %23, %27, 0], sizes = [1, %25, %29, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x224x224x3xf32>> -> tensor<1x?x?x3xf32>
              %32 = tensor.pad %31 low[0, 0, 0, 0] high[0, %26, %30, 0] {
              ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
                tensor.yield %cst : f32
              } : tensor<1x?x?x3xf32> to tensor<1x?x?x3xf32>
              %33 = affine.min affine_map<(d0) -> (-d0 + 32, 32)>(%arg2)[]
              %34 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 3, %33], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>> -> tensor<3x3x3x?xf32>
              %36 = affine.min affine_map<(d0) -> (-d0 + 112, 4)>(%arg1)[]
              %37 = affine.min affine_map<(d0) -> (-d0 + 32, 32)>(%arg2)[]
              %38 = tensor.empty(%36, %37) : tensor<1x1x?x?xf32>
              %39 = linalg.fill ins(%cst : f32) outs(%38 : tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
              %40 = linalg.conv_2d_nhwc_hwcf {lowering_config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%32, %34 : tensor<1x?x?x3xf32>, tensor<3x3x3x?xf32>) outs(%39 : tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %16 : tensor<1x1x?x?xf32>, tensor<1x1x?x?xf32>) outs(%20 : tensor<1x1x?x?xf32>) {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
                %42 = arith.subf %arg3, %arg4 : f32
                linalg.yield %42 : f32
              } -> tensor<1x1x?x?xf32>
              flow.dispatch.tensor.store %41, %3, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 1, %18, %19], strides = [1, 1, 1, 1] : tensor<1x1x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
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
// CHECK-COUNT-9: vector.transfer_read
// CHECK-COUNT-6: vector.fma
// Fused elementwise ops
// CHECK-COUNT-2: vector.transfer_read
// CHECK-COUNT-2: arith.subf

//         CHECK: } else {

// Slow path:
// Loop nest for thread tiling and reduction tiling
// CHECK-COUNT-4: scf.for
//         CHECK: scf.if
// CHECK-COUNT-3:   vector.transfer_read
//         CHECK: scf.if
// CHECK-COUNT-3:   vector.transfer_read
// CHECK-COUNT-3: vector.transfer_read
// CHECK-COUNT-6: vector.fma
// Fused elementwise ops
// CHECK-COUNT-2: vector.transfer_read
// CHECK-COUNT-2: arith.subf

// -----

#config =  #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable private @low_high_padded_nhwc_depthwise_conv {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.export @low_high_padded_nhwc_depthwise_conv layout(#pipeline_layout) attributes {
      workgroup_size = [8: index, 2: index, 1: index],
      translation_info = #translation
    }
    builtin.module {
      func.func @low_high_padded_nhwc_depthwise_conv() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c112 = arith.constant 112 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x32xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
        %4 = tensor.empty() : tensor<1x112x112x32xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c112 step %workgroup_count_z {
          %7 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
          %8 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
          scf.for %arg1 = %7 to %c112 step %8 {
            %9 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %10 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %9 to %c32 step %10 {
              %11 = affine.min affine_map<(d0) -> (32, -d0 + 32)>(%arg2)[]
              %12 = flow.dispatch.tensor.load %2, offsets = [%arg2], sizes = [%11], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xf32>> -> tensor<?xf32>
              %14 = affine.min affine_map<(d0) -> (4, -d0 + 112)>(%arg1)[]
              %16 = affine.min affine_map<(d0) -> (4, -d0 + 112)>(%arg1)[]
              %17 = affine.min affine_map<(d0) -> (16, -d0 + 32)>(%arg2)[]
              %18 = tensor.extract_slice %4[0, %arg0, %arg1, %arg2] [1, 1, %16, %17] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x1x?x?xf32>
              %19 = affine.min affine_map<(d0) -> (3, -d0 + 114)>(%arg0)
              %20 = affine.min affine_map<(d0, d1) -> (d1 + 2, -d0 + 114)>(%arg1, %14)
              %21 = affine.min affine_map<(d0) -> (-d0 + 32, 32)>(%arg2)[]
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
              %39 = flow.dispatch.tensor.load %0, offsets = [0, %24, %31, %36], sizes = [1, %27, %34, %38], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>> -> tensor<1x?x?x?xf32>
              %40 = tensor.pad %39 low[0, %22, %29, 0] high[0, %28, %35, 0] {
              ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
                tensor.yield %cst : f32
              } : tensor<1x?x?x?xf32> to tensor<1x?x?x?xf32>
              %41 = affine.min affine_map<(d0) -> (-d0 + 32, 32)>(%arg2)[]
              %42 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [3, 3, %41], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x32xf32>> -> tensor<3x3x?xf32>
              %44 = affine.min affine_map<(d0) -> (-d0 + 112, 4)>(%arg1)[]
              %45 = affine.min affine_map<(d0) -> (-d0 + 32, 32)>(%arg2)[]
              %46 = tensor.empty(%44, %45) : tensor<1x1x?x?xf32>
              %47 = linalg.fill ins(%cst : f32) outs(%46 : tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
              %48 = linalg.depthwise_conv_2d_nhwc_hwc {lowering_config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%40, %42 : tensor<1x?x?x?xf32>, tensor<3x3x?xf32>) outs(%47 : tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
              %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %48 : tensor<?xf32>, tensor<1x1x?x?xf32>) outs(%18 : tensor<1x1x?x?xf32>) {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
                %50 = arith.addf %arg3, %arg4 : f32
                linalg.yield %50 : f32
              } -> tensor<1x1x?x?xf32>
              flow.dispatch.tensor.store %49, %3, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 1, %16, %17], strides = [1, 1, 1, 1] : tensor<1x1x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
      }
    }
  }
  return
      }
    }
  }
}

// CHECK-LABEL: func.func @low_high_padded_nhwc_depthwise_conv()

// Loop nest for workgroup tiling and distribution
// CHECK-COUNT-3: scf.for

// Switch between fast and slow path
//         CHECK: scf.if

// Fast path:
// Loop nest for thread tiling and reduction tiling
// CHECK-COUNT-4: scf.for
// Vector code
// CHECK-COUNT-3: vector.transfer_read
// CHECK-COUNT-2: vector.fma

//         CHECK: } else {

// Slow path:
// Loop nest for thread tiling and reduction tiling
// CHECK-COUNT-4: scf.for
//         CHECK: scf.if
//         CHECK:   vector.transfer_read
//         CHECK: scf.if
//         CHECK:   vector.transfer_read
//         CHECK: vector.transfer_read
// CHECK-COUNT-2: vector.fma

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 16, 8, 8], [0, 8, 1, 4], [0, 0, 0, 0, 4, 1, 1], [0, 0, 1, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable private @nchw_conv_static_shape_f32 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.export @nchw_conv_static_shape_f32 layout(#pipeline_layout) attributes {
      workgroup_size = [4: index, 4: index, 1: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @nchw_conv_static_shape_f32() {
        %c1280 = arith.constant 1280 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x1280x10x10xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1280x1280x3x3xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_z]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_z]
        scf.for %arg0 = %3 to %c1280 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
          scf.for %arg1 = %5 to %c8 step %6 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_x]
            %8 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_x]
            scf.for %arg2 = %7 to %c8 step %8 {
              %9 = flow.dispatch.tensor.load %0, offsets = [0, 0, %arg1, %arg2], sizes = [2, 1280, 10, 10], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280x10x10xf32>> -> tensor<2x1280x10x10xf32>
              %10 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, 0, 0], sizes = [16, 1280, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280x3x3xf32>> -> tensor<16x1280x3x3xf32>
              %11 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [2, 16, 8, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>> -> tensor<2x16x8x8xf32>
              %12 = linalg.conv_2d_nchw_fchw
                      {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>}
                      ins(%9, %10 : tensor<2x1280x10x10xf32>, tensor<16x1280x3x3xf32>)
                      outs(%11 : tensor<2x16x8x8xf32>) -> tensor<2x16x8x8xf32>
              flow.dispatch.tensor.store %12, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [2, 16, 8, 8], strides = [1, 1, 1, 1] : tensor<2x16x8x8xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>>
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @nchw_conv_static_shape_f32()

// No vector transfer write ops generated for the linalg.fill op: initial values are forwarded to loops.
// CHECK-NOT: vector.transfer

// Check tiling loop along input channel and filter height/width
// TODO: enable vector hoisting
//      CHECK: scf.for %{{.*}} = %c0 to %c1280 step %c4
// CHECK-SAME:     -> (tensor<2x8x1x4xf32>)
//      CHECK:   scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:       -> (tensor<2x8x1x4xf32>)
//      CHECK:     scf.for %{{.*}} = %c0 to %c3 step %c1
// CHECK-SAME:         -> (vector<4xf32>{{(, vector<4xf32>)+}})

// CHECK-COUNT-64: vector.fma

// For linalg.conv_2d_nchw_fchw
// CHECK-COUNT-16: vector.transfer_write

//  CHECK-COUNT-3: scf.yield %{{.+}} : tensor<2x16x8x8xf32>
