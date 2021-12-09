// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-set-num-workgroups,builtin.module(builtin.func(iree-spirv-tile,iree-spirv-vectorize))))' %s | IreeFileCheck %s

#config = #iree_codegen.lowering.config<tile_sizes = [[8, 64], [8, 4], [0, 0, 4]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [64, 8]>
hal.executable private @matmul_static_shape_f16 {
  hal.interface public @io {
    hal.interface.binding public @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @matmul_static_shape_f16 attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [16: index, 1: index, 1: index],
      translation.info = #translation
    }
    builtin.module  {
      func @matmul_static_shape_f16() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %c4096 = arith.constant 4096 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:4096x4096xf16>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:4096x4096xf16>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:4096x4096xf16>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c4096 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c4096 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4096)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:4096x4096xf16> -> tensor<?x4096xf16>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4096)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [4096, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:4096x4096xf16> -> tensor<4096x?xf16>
            %11 = affine.min affine_map<(d0)[s0] -> (-d0 + 4096, s0)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (-d0 + 4096, s0)>(%arg1)[%workgroup_size_x]
            %13 = linalg.init_tensor [%11, %12] : tensor<?x?xf16>
            %14 = linalg.fill(%cst, %13) : f16, tensor<?x?xf16> -> tensor<?x?xf16>
            %15 = linalg.matmul {lowering.config = #config} ins(%8, %10 : tensor<?x4096xf16>, tensor<4096x?xf16>) outs(%14 : tensor<?x?xf16>) -> tensor<?x?xf16>
            flow.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf16> -> !flow.dispatch.tensor<writeonly:4096x4096xf16>
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

//    CHECK-LABEL: func @matmul_static_shape_f16
//      CHECK-NOT:   vector.transfer
//          CHECK:   %{{.+}}:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write
//          CHECK:    return

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[8, 64], [8, 4], [0, 0, 4]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [64, 8]>

hal.executable private @matmul_static_shape_f32 {
  hal.interface public @io {
    hal.interface.binding public @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @matmul_static_shape_f32 attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [16: index, 1: index, 1: index],
      translation.info = #translation
    }
    builtin.module  {
      func @matmul_static_shape_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c4096 = arith.constant 4096 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:4096x4096xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:4096x4096xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:4096x4096xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c4096 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c4096 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4096)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:4096x4096xf32> -> tensor<?x4096xf32>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4096)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [4096, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:4096x4096xf32> -> tensor<4096x?xf32>
            %11 = affine.min affine_map<(d0)[s0] -> (-d0 + 4096, s0)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (-d0 + 4096, s0)>(%arg1)[%workgroup_size_x]
            %13 = linalg.init_tensor [%11, %12] : tensor<?x?xf32>
            %14 = linalg.fill(%cst, %13) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %15 = linalg.matmul {lowering.config = #config} ins(%8, %10 : tensor<?x4096xf32>, tensor<4096x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:4096x4096xf32>
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

//    CHECK-LABEL: func @matmul_static_shape_f32
//      CHECK-NOT:   vector.transfer
//          CHECK:   %{{.+}}:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write
//          CHECK:    return
