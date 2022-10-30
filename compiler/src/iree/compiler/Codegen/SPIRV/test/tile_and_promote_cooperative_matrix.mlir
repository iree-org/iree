// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-promote{skip-thread=true}))))' --cse %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 32], [16, 16, 16], [0, 0, 32]]>

hal.executable @matmul_f16_32x32x32 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.5,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
      [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_nv = [
          #spirv.coop_matrix_props<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @matmul_f16_32x32x32 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @matmul_f16_32x32x32() {
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<32x32xf16>
        memref.assume_alignment %0, 64 : memref<32x32xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<32x32xf16>
        memref.assume_alignment %1, 64 : memref<32x32xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : memref<32x32xf16>
        memref.assume_alignment %2, 64 : memref<32x32xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : memref<32x32xf16>
        memref.assume_alignment %3, 64 : memref<32x32xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
        scf.for %arg0 = %4 to %c32 step %5 {
          %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
          %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
          scf.for %arg1 = %6 to %c32 step %7 {
            linalg.fill {lowering_config = #config} ins(%cst : f16) outs(%3 : memref<32x32xf16>)
            linalg.matmul {lowering_config = #config}
              ins(%0, %1 : memref<32x32xf16>, memref<32x32xf16>) outs(%3 : memref<32x32xf16>)
            linalg.generic {
                indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel"]}
            ins(%2 : memref<32x32xf16>) outs(%3 : memref<32x32xf16>) attrs = {lowering_config = #config} {
            ^bb0(%in: f16, %out: f16):
              %8 = arith.divf %out, %in : f16
              linalg.yield %8 : f16
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @matmul_f16_32x32x32()

//       CHECK:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0)
//       CHECK:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1)

//   CHECK-NOT:   memref.alloc()
//   CHECK-NOT:   memref.copy

//       CHECK:   linalg.matmul
//  CHECK-SAME:     __internal_linalg_transform__ = "workgroup_memory"
//  CHECK-SAME:     ins(%[[LHS]], %[[RHS]] : memref<32x32xf16>, memref<32x32xf16>)
