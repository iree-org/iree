// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-spirv-tile-and-vectorize-to-cooperative-ops))))' %s | FileCheck %s

#config = #iree_codegen.lowering.config<tile_sizes = [[16, 16, 16], [16, 16, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"SPIRVVectorizeToCooperativeOps", workload_per_wg = [16, 16]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x128_div_sub {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
          [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
          {cooperative_matrix_properties_nv = [
            {a_type = i8, b_type = i8, c_type = i32, k_size = 32 : i32,
             m_size = 8 : i32, n_size = 8 : i32, result_type = i32, scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f16, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f16,
             scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f32, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f32,
             scope = 3 : i32}],
           max_compute_shared_memory_size = 49152 : i32,
           max_compute_workgroup_invocations = 1024 : i32,
           max_compute_workgroup_size = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
           subgroup_size = 32 : i32}>}> {
    hal.executable.entry_point public @matmul_256x1024x128_div_sub layout(#executable_layout) attributes {
      translation.info = #translation,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    } {
    ^bb0(%arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg0]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg1]
      hal.return %0, %1, %c1 : index, index, index
    }
    builtin.module  {
      func @matmul_256x1024x128_div_sub() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<256x1024xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<256x1024xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<256x128xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : memref<128x1024xf16>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : memref<256x1024xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_y]
        %6 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_y]
        scf.for %arg0 = %5 to %c256 step %6 {
          %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
          %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
          scf.for %arg1 = %7 to %c1024 step %8 {
            %9 = memref.subview %0[%arg0, %arg1] [16, 16] [1, 1] : memref<256x1024xf16> to memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            %10 = memref.subview %1[%arg0, %arg1] [16, 16] [1, 1] : memref<256x1024xf16> to memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            %11 = memref.subview %2[%arg0, 0] [16, 128] [1, 1] : memref<256x128xf16> to memref<16x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
            %12 = memref.subview %3[0, %arg1] [128, 16] [1, 1] : memref<128x1024xf16> to memref<128x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            %13 = memref.subview %4[%arg0, %arg1] [16, 16] [1, 1] : memref<256x1024xf16> to memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            linalg.fill(%cst, %13) {lowering.config = #config} : f16, memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
            linalg.matmul {lowering.config = #config}
              ins(%11, %12 : memref<16x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>, memref<128x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              outs(%13 : memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
              ins(%13, %9, %10 : memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              outs(%13 : memref<16x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
              attrs =  {lowering.config = #config} {
            ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
              %14 = arith.divf %arg2, %arg3 : f16
              %15 = arith.subf %14, %arg4 : f16
              linalg.yield %15 : f16
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func @matmul_256x1024x128_div_sub

// CHECK-DAG: %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf16>
// CHECK-DAG: %[[PAD:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C48:.+]] = arith.constant 48 : index
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C80:.+]] = arith.constant 80 : index
// CHECK-DAG: %[[C96:.+]] = arith.constant 96 : index
// CHECK-DAG: %[[C112:.+]] = arith.constant 112 : index

// CHECK: %[[DIV_BUFFER:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<256x1024xf16>
// CHECK: %[[SUB_BUFFER:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<256x1024xf16>
// CHECK: %[[LHS_BUFFER:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<256x128xf16>
// CHECK: %[[RHS_BUFFER:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : memref<128x1024xf16>
// CHECK: %[[ACC_BUFFER:.+]] = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : memref<256x1024xf16>

// CHECK: scf.for %[[IV_Y:.+]] =
// CHECK:   %[[LHS_TILE:.+]] = memref.subview %[[LHS_BUFFER]][%[[IV_Y]], 0] [16, 128] [1, 1]
// CHECK:   scf.for %[[IV_X:.+]] =
// CHECK:     %[[DIV_TILE:.+]] = memref.subview %[[DIV_BUFFER]][%[[IV_Y]], %[[IV_X]]] [16, 16] [1, 1]
// CHECK:     %[[SUB_TILE:.+]] = memref.subview %[[SUB_BUFFER]][%[[IV_Y]], %[[IV_X]]] [16, 16] [1, 1]
// CHECK:     %[[RHS_TILE:.+]] = memref.subview %[[RHS_BUFFER]][0, %[[IV_X]]] [128, 16] [1, 1]
// CHECK:     %[[ACC_TILE:.+]] = memref.subview %[[ACC_BUFFER]][%[[IV_Y]], %[[IV_X]]] [16, 16] [1, 1]
// CHECK:     vector.transfer_write %[[INIT]], %[[ACC_TILE]][%[[C0]], %[[C0]]]
// CHECK:     %[[LHS_0:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK:     %[[LHS_1:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C16]]], %[[PAD]]
// CHECK:     %[[LHS_2:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C32]]], %[[PAD]]
// CHECK:     %[[LHS_3:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C48]]], %[[PAD]]
// CHECK:     %[[LHS_4:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C64]]], %[[PAD]]
// CHECK:     %[[LHS_5:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C80]]], %[[PAD]]
// CHECK:     %[[LHS_6:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C96]]], %[[PAD]]
// CHECK:     %[[LHS_7:.+]] = vector.transfer_read %[[LHS_TILE]][%[[C0]], %[[C112]]], %[[PAD]]
// CHECK:     %[[RHS_0:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK:     %[[RHS_1:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C16]], %[[C0]]], %[[PAD]]
// CHECK:     %[[RHS_2:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C32]], %[[C0]]], %[[PAD]]
// CHECK:     %[[RHS_3:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C48]], %[[C0]]], %[[PAD]]
// CHECK:     %[[RHS_4:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C64]], %[[C0]]], %[[PAD]]
// CHECK:     %[[RHS_5:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C80]], %[[C0]]], %[[PAD]]
// CHECK:     %[[RHS_6:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C96]], %[[C0]]], %[[PAD]]
// CHECK:     %[[RHS_7:.+]] = vector.transfer_read %[[RHS_TILE]][%[[C112]], %[[C0]]], %[[PAD]]
// CHECK:     %[[ACC_0:.+]] = vector.transfer_read %[[ACC_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK:     %[[ACC_1:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_0]], %[[RHS_0]], %[[ACC_0]]
// CHECK:     %[[ACC_2:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_1]], %[[RHS_1]], %[[ACC_1]]
// CHECK:     %[[ACC_3:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_2]], %[[RHS_2]], %[[ACC_2]]
// CHECK:     %[[ACC_4:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_3]], %[[RHS_3]], %[[ACC_3]]
// CHECK:     %[[ACC_5:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_4]], %[[RHS_4]], %[[ACC_4]]
// CHECK:     %[[ACC_6:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_5]], %[[RHS_5]], %[[ACC_5]]
// CHECK:     %[[ACC_7:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_6]], %[[RHS_6]], %[[ACC_6]]
// CHECK:     %[[ACC_8:.+]] = vector.contract {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_7]], %[[RHS_7]], %[[ACC_7]]
// CHECK:     vector.transfer_write %[[ACC_8]], %[[ACC_TILE]][%[[C0]], %[[C0]]]
// CHECK:     %[[MAT_VEC:.+]] = vector.transfer_read %[[ACC_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK:     %[[DIV_VEC:.+]] = vector.transfer_read %[[DIV_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK:     %[[SUB_VEC:.+]] = vector.transfer_read %[[SUB_TILE]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK:     %[[DIV:.+]] = arith.divf %[[MAT_VEC]], %[[DIV_VEC]] : vector<16x16xf16>
// CHECK:     %[[SUB:.+]] = arith.subf %[[DIV]], %[[SUB_VEC]] : vector<16x16xf16>
// CHECK:     vector.transfer_write %[[SUB]], %[[ACC_TILE]][%[[C0]], %[[C0]]]
