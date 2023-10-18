// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile,canonicalize,cse,iree-codegen-generic-vectorization,iree-spirv-initial-vector-lowering,iree-codegen-hoist-redundant-vector-transfers,iree-spirv-final-vector-lowering,canonicalize,cse)))))' \
// RUN:   %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 2, 2, 8], [0, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @pooling_nhwc_sum_f32 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.export @pooling_nhwc_sum_f32 layout(#pipeline_layout) attributes {
      workgroup_size = [2: index, 2: index, 2: index],
      translation_info = #translation
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module  {
      func.func @pooling_nhwc_sum_f32() {
        %c2 = arith.constant 2 : index
        %c24 = arith.constant 24 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x24x24x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x2x2x8xf32>>
        %2 = tensor.empty() : tensor<12x12xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 24)>()[%workgroup_id_z]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 24)>()[%workgroup_id_y]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_x]
        %6 = flow.dispatch.tensor.load %0, offsets = [0, %3, %4, %5], sizes = [1, %c24, %c24, %c8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x24x24x8xf32>> -> tensor<1x?x?x?xf32>
        %7 = tensor.empty() : tensor<1x2x2x8xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
        %9 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<12> : vector<2xi64>}
          ins(%6, %2 : tensor<1x?x?x?xf32>, tensor<12x12xf32>)
          outs(%8 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
        %cast = tensor.cast %9 : tensor<1x2x2x8xf32> to tensor<1x?x?x?xf32>
        %10 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_id_z]
        %11 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_id_y]
        %12 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_x]
        flow.dispatch.tensor.store %cast, %1, offsets = [0, %10, %11, %12], sizes = [1, %c2, %c2, %c8], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x2x2x8xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @pooling_nhwc_sum_f32()

// No vector transfer write ops generated for the linalg.fill op: initial values are forwarded to loops.
// CHECK-NOT: vector.transfer

// Check tiling loop along filter height/width
//      CHECK: scf.for %{{.*}} = %c0 to %c12 step %c1
//      CHECK:   scf.for %{{.*}} = %c0 to %c12 step %c1

// CHECK: vector.transfer_read
// CHECK: arith.addf %{{.+}}, %{{.+}} : vector<4xf32>

// CHECK-OUNT-2: scf.yield

// For linalg.conv_2d_nhwc_hwcf
// CHECK: vector.transfer_write
