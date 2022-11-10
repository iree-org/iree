// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile,iree-spirv-vectorize, cse)))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 8, 64], [1, 8, 4], [0, 0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @fused_fill_batch_matmul {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.export @fused_fill_batch_matmul layout(#pipeline_layout) attributes {
      workgroup_size = [16: index, 1: index, 1: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @fused_fill_batch_matmul() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c4 = arith.constant 4 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x1024x1024xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x1024x1024xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4x1024x1024xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c4 step %workgroup_count_z {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
            %8 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
            scf.for %arg2 = %7 to %c1024 step %8 {
              %10 = affine.min affine_map<(d0) -> (8, -d0 + 1024)>(%arg1)[]
              %11 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [1, %10, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1024x1024xf32>> -> tensor<1x?x1024xf32>
              %12 = affine.min affine_map<(d0) -> (64, -d0 + 1024)>(%arg2)[]
              %13 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [1, 1024, %12], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1024x1024xf32>> -> tensor<1x1024x?xf32>
              %15 = affine.min affine_map<(d0) -> (-d0 + 1024, 8)>(%arg1)[]
              %16 = affine.min affine_map<(d0) -> (-d0 + 1024, 64)>(%arg2)[]
              %17 = tensor.empty(%15, %16) : tensor<1x?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
              %19 = linalg.batch_matmul {lowering_config = #config} ins(%11, %13 : tensor<1x?x1024xf32>, tensor<1x1024x?xf32>) outs(%18 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
              flow.dispatch.tensor.store %19, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, %10, %12], strides = [1, 1, 1] : tensor<1x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x1024x1024xf32>>
            }
          }
        }
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @fused_fill_batch_matmul
//      CHECK-NOT:   vector.transfer
//          CHECK:   %{{.+}}:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write
//          CHECK:    return
