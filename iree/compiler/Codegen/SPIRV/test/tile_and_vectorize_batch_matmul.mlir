// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-set-num-workgroups,builtin.module(func.func(iree-spirv-tile,iree-spirv-vectorize))))' -cse %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 8, 64], [1, 8, 4], [0, 0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVVectorize, workload_per_wg = [64, 8, 1]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @fused_fill_batch_matmul {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @fused_fill_batch_matmul layout(#executable_layout) {
      workgroup_size = [16: index, 1: index, 1: index],
      translation_info = #translation
    }
    builtin.module  {
      func @fused_fill_batch_matmul() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c4 = arith.constant 4 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:4x1024x1024xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:4x1024x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:4x1024x1024xf32>
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
        scf.for %arg0 = %3 to %c4 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c1024 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_y]
              %11 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [%9, %10, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:4x1024x1024xf32> -> tensor<?x?x1024xf32>
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg2)[%workgroup_size_x]
              %13 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [%9, 1024, %12], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:4x1024x1024xf32> -> tensor<?x1024x?xf32>
              %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 4, s0)>(%arg0)[%workgroup_size_z]
              %15 = affine.min affine_map<(d0)[s0] -> (-d0 + 1024, s0)>(%arg1)[%workgroup_size_y]
              %16 = affine.min affine_map<(d0)[s0] -> (-d0 + 1024, s0)>(%arg2)[%workgroup_size_x]
              %17 = linalg.init_tensor [%14, %15, %16] : tensor<?x?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
              %19 = linalg.batch_matmul {lowering_config = #config} ins(%11, %13 : tensor<?x?x1024xf32>, tensor<?x1024x?xf32>) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
              flow.dispatch.tensor.store %19, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%9, %10, %12], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:4x1024x1024xf32>
            }
          }
        }
        return
      }
    }
  }
}

//    CHECK-LABEL: func @fused_fill_batch_matmul
//      CHECK-NOT:   vector.transfer
//          CHECK:   %{{.+}}:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write
//          CHECK:    return
