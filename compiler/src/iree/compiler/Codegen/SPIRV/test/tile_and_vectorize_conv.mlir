// Note: upstream hoisting on tensors does not canonicalize scf.for anymore, run canonicalization to make this test happy

// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-create-fast-slow-path,iree-spirv-tile,canonicalize,cse,iree-spirv-vectorize,canonicalize,cse)))))' %s | FileCheck %s

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
