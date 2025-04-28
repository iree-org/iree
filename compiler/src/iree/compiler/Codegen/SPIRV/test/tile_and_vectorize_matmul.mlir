// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-gpu-tile,canonicalize,cse,iree-codegen-generic-vectorization,iree-spirv-initial-vector-lowering,iree-codegen-optimize-tensor-insert-extract-slices,iree-spirv-final-vector-lowering,canonicalize,cse)))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[8, 64], [8, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matmul_static_shape_f16 {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @matmul_static_shape_f16 layout(#pipeline_layout) attributes {
      workgroup_size = [16: index, 1: index, 1: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @matmul_static_shape_f16() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %c4096 = arith.constant 4096 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf16>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c4096 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c4096 step %6 {
            %7 = affine.min affine_map<(d0) -> (8, -d0 + 4096)>(%arg0)[]
            %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>> -> tensor<?x4096xf16>
            %9 = affine.min affine_map<(d0) -> (64, -d0 + 4096)>(%arg1)[]
            %10 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [4096, %9], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>> -> tensor<4096x?xf16>
            %11 = affine.min affine_map<(d0) -> (-d0 + 4096, 8)>(%arg0)[]
            %12 = affine.min affine_map<(d0) -> (-d0 + 4096, 64)>(%arg1)[]
            %13 = tensor.empty(%11, %12) : tensor<?x?xf16>
            %14 = linalg.fill ins(%cst : f16) outs(%13 : tensor<?x?xf16>) -> tensor<?x?xf16>
            %15 = linalg.matmul {lowering_config = #config} ins(%8, %10 : tensor<?x4096xf16>, tensor<4096x?xf16>) outs(%14 : tensor<?x?xf16>) -> tensor<?x?xf16>
            iree_tensor_ext.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf16>>
          }
        }
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @matmul_static_shape_f16
//      CHECK-NOT:   vector.transfer
//          CHECK:   %{{.+}}:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write
//          CHECK:    return

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[8, 64], [8, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matmul_static_shape_f32 {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @matmul_static_shape_f32 layout(#pipeline_layout) attributes {
      workgroup_size = [16: index, 1: index, 1: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @matmul_static_shape_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c4096 = arith.constant 4096 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c4096 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c4096 step %6 {
            %7 = affine.min affine_map<(d0) -> (8, -d0 + 4096)>(%arg0)[]
            %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<?x4096xf32>
            %9 = affine.min affine_map<(d0) -> (64, -d0 + 4096)>(%arg1)[]
            %10 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [4096, %9], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<4096x?xf32>
            %11 = affine.min affine_map<(d0) -> (-d0 + 4096, 8)>(%arg0)[]
            %12 = affine.min affine_map<(d0) -> (-d0 + 4096, 64)>(%arg1)[]
            %13 = tensor.empty(%11, %12) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = linalg.matmul {lowering_config = #config} ins(%8, %10 : tensor<?x4096xf32>, tensor<4096x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            iree_tensor_ext.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
          }
        }
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @matmul_static_shape_f32
//      CHECK-NOT:   vector.transfer
//          CHECK:   %{{.+}}:8 = scf.for
// CHECK-COUNT-12:     vector.transfer_read
// CHECK-COUNT-32:     vector.fma
//      CHECK:         scf.yield
//  CHECK-COUNT-8:    vector.transfer_write
//          CHECK:    return
