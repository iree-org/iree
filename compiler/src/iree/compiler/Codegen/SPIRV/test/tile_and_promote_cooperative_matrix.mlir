// RUN: iree-opt --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-promote{promote-c=false skip-thread=true}, cse)))))' \
// RUN:   %s | FileCheck %s

// RUN: iree-opt --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-promote{promote-c=true skip-thread=true}, cse)))))' \
// RUN:   %s | FileCheck %s --check-prefix=PROMOTEC

// Single tile per workgroup means no subview ops for promotion.

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
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
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
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
        memref.assume_alignment %0, 64 : memref<32x32xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
        memref.assume_alignment %1, 64 : memref<32x32xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
        memref.assume_alignment %2, 64 : memref<32x32xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
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
            linalg.fill ins(%cst : f16) outs(%3 : memref<32x32xf16>)
            linalg.matmul {lowering_config = #config}
              ins(%0, %1 : memref<32x32xf16>, memref<32x32xf16>) outs(%3 : memref<32x32xf16>)
            linalg.generic {
                indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel"]}
            ins(%2 : memref<32x32xf16>) outs(%3 : memref<32x32xf16>) {
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


// PROMOTEC-LABEL: func.func @matmul_f16_32x32x32()

//       PROMOTEC:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0)
//       PROMOTEC:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1)

//   PROMOTEC-NOT:   memref.alloc()
//   PROMOTEC-NOT:   memref.copy

//       PROMOTEC:   linalg.matmul
//  PROMOTEC-SAME:     __internal_linalg_transform__ = "workgroup_memory"
//  PROMOTEC-SAME:     ins(%[[LHS]], %[[RHS]] : memref<32x32xf16>, memref<32x32xf16>)

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32, 32], [1, 16, 16, 16], [0, 0, 0, 32]]>
hal.executable @generic_batch_matmul_f16_32x128x512x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @generic_batch_matmul_f16_32x128x512x64 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @generic_batch_matmul_f16_32x128x512x64() {
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %span0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<128x32x64xf16>
        %span1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64x512xf16>
        %span2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %span3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c32 step %workgroup_count_z {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
          scf.for %arg1 = %3 to %c128 step %4 {
            %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %5 to %c512 step %6 {
              %subview = memref.subview %span2[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              %subview_0 = memref.subview %span0[%arg1, %arg0, 0] [32, 1, 64] [1, 1, 1] : memref<128x32x64xf16> to memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>
              %subview_1 = memref.subview %span1[%arg0, 0, %arg2] [1, 64, 32] [1, 1, 1] : memref<32x64x512xf16> to memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>
              linalg.fill ins(%cst : f16) outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%subview_0, %subview_1 : memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>, memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              attrs = {lowering_config = #config} {
              ^bb0(%in: f16, %in_2: f16, %out: f16):
                %7 = arith.mulf %in, %in_2 : f16
                %8 = arith.addf %out, %7 : f16
                linalg.yield %8 : f16
              }
              %subview_2 = memref.subview %span3[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              linalg.generic {
                  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                  iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%subview_2 : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>) {
              ^bb0(%in: f16, %out: f16):
                // spirv.GL.Exp is not permitted to use cooperative matrix types per the spec.
                %8 = math.exp %in : f16
                linalg.yield %8 : f16
              }
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @generic_batch_matmul_f16_32x128x512x64()

//      CHECK: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>
//      CHECK: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x1x32xf16, #gpu.address_space<workgroup>>

//      CHECK: linalg.fill
// CHECK-SAME:   __internal_linalg_transform__ = "workgroup_memory"

//      CHECK: scf.for %{{.+}} = %c0 to %c64 step %c32
//      CHECK:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c32, %c1, %c32]
//      CHECK:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c32]
//      CHECK:   gpu.barrier
//      CHECK:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// CHECK-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// CHECK-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:   gpu.barrier
//      CHECK:   linalg.generic
// CHECK-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// CHECK-SAME:    __internal_linalg_transform__ = "workgroup_memory"


// PROMOTEC-LABEL: func.func @generic_batch_matmul_f16_32x128x512x64()

//      PROMOTEC: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>
//      PROMOTEC: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x1x32xf16, #gpu.address_space<workgroup>>
//      PROMOTEC: %[[C_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[C_ALLOC]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c64 step %c32
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c32, %c1, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c32]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.generic
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[C_ALLOC]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
//      PROMOTEC: gpu.barrier
//      PROMOTEC: linalg.generic
//      PROMOTEC:    ins(%{{.+}}, %[[C_ALLOC]]
// PROMOTEC-SAME:   __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC: gpu.barrier

// -----

// Cooperative matrix fusable elementwise ops do not need promote C.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32, 32], [1, 16, 16, 16], [0, 0, 0, 32]]>
hal.executable @generic_batch_matmul_f16_32x128x512x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @generic_batch_matmul_f16_32x128x512x64 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @generic_batch_matmul_f16_32x128x512x64() {
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %span0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<128x32x64xf16>
        %span1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64x512xf16>
        %span2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %span3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c32 step %workgroup_count_z {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
          scf.for %arg1 = %3 to %c128 step %4 {
            %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %5 to %c512 step %6 {
              %subview = memref.subview %span2[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              %subview_0 = memref.subview %span0[%arg1, %arg0, 0] [32, 1, 64] [1, 1, 1] : memref<128x32x64xf16> to memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>
              %subview_1 = memref.subview %span1[%arg0, 0, %arg2] [1, 64, 32] [1, 1, 1] : memref<32x64x512xf16> to memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>
              linalg.fill ins(%cst : f16) outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%subview_0, %subview_1 : memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>, memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              attrs = {lowering_config = #config} {
              ^bb0(%in: f16, %in_2: f16, %out: f16):
                %7 = arith.mulf %in, %in_2 : f16
                %8 = arith.addf %out, %7 : f16
                linalg.yield %8 : f16
              }
              %subview_2 = memref.subview %span3[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              linalg.generic {
                  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                  iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%subview_2 : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>) {
              ^bb0(%in: f16, %out: f16):
                %8 = arith.divf %out, %in : f16
                linalg.yield %8 : f16
              }
            }
          }
        }
        return
      }
    }
  }
}

// PROMOTEC-LABEL: func.func @generic_batch_matmul_f16_32x128x512x64()

//      PROMOTEC: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>
//      PROMOTEC: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x1x32xf16, #gpu.address_space<workgroup>>

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c64 step %c32
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c32, %c1, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c32]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.generic
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"


// -----

// No need to promote C if there is no fused element wise ops.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32, 32], [1, 16, 16, 16], [0, 0, 0, 32]]>
hal.executable @generic_batch_matmul_f16_32x128x512x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @generic_batch_matmul_f16_32x128x512x64 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @generic_batch_matmul_f16_32x128x512x64() {
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<128x32x64xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64x512xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c32 step %workgroup_count_z {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
          scf.for %arg1 = %3 to %c128 step %4 {
            %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %5 to %c512 step %6 {
              %subview = memref.subview %2[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              %subview_0 = memref.subview %0[%arg1, %arg0, 0] [32, 1, 64] [1, 1, 1] : memref<128x32x64xf16> to memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>
              %subview_1 = memref.subview %1[%arg0, 0, %arg2] [1, 64, 32] [1, 1, 1] : memref<32x64x512xf16> to memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>
              linalg.fill ins(%cst : f16) outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%subview_0, %subview_1 : memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>, memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              attrs = {lowering_config = #config} {
              ^bb0(%in: f16, %in_2: f16, %out: f16):
                %7 = arith.mulf %in, %in_2 : f16
                %8 = arith.addf %out, %7 : f16
                linalg.yield %8 : f16
              }
            }
          }
        }
        return
      }
    }
  }
}


// PROMOTEC-LABEL: func.func @generic_batch_matmul_f16_32x128x512x64()

//  PROMOTEC-NOT: memref.alloc()
//      PROMOTEC: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>
//      PROMOTEC: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x1x32xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-NOT: memref.alloc()

//      PROMOTEC: %[[SPAN2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//      PROMOTEC: %[[OUT_VIEW:.+]] = memref.subview %[[SPAN2]]

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[OUT_VIEW]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c64 step %c32
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c32, %c1, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c32]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.generic
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[OUT_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
//  PROMOTEC-NOT: gpu.barrier
//  PROMOTEC-NOT: memref.copy

// -----

// No need to promote again with allocations from bufferization.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128], [1, 32, 64], [0, 0, 0, 32], [1, 16, 16, 16]]>

hal.executable @batch_matmul_f16_1x64x128x512 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_NV_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}> {
    hal.executable.export public @batch_matmul_f16_1x64x128x512 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>,
      workgroup_size = [128 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @batch_matmul_f16_1x64x128x512() {
        %c4096 = arith.constant 4096 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<1x4096x512xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<1x512x4096xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<1x4096x4096xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c4096 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c4096 step %6 {
            %subview = memref.subview %2[0, %arg0, %arg1] [1, 64, 128] [1, 1, 1] : memref<1x4096x4096xf32> to memref<1x64x128xf32, strided<[16777216, 4096, 1], offset: ?>>
            %subview_0 = memref.subview %0[0, %arg0, 0] [1, 64, 512] [1, 1, 1] : memref<1x4096x512xf16> to memref<1x64x512xf16, strided<[2097152, 512, 1], offset: ?>>
            %subview_1 = memref.subview %1[0, 0, %arg1] [1, 512, 128] [1, 1, 1] : memref<1x512x4096xf16> to memref<1x512x128xf16, strided<[2097152, 4096, 1], offset: ?>>
            %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x64x128xf16, #gpu.address_space<workgroup>>
            linalg.fill ins(%cst : f16) outs(%alloc : memref<1x64x128xf16, #gpu.address_space<workgroup>>)
            linalg.batch_matmul {lowering_config = #config}
              ins(%subview_0, %subview_1 : memref<1x64x512xf16, strided<[2097152, 512, 1], offset: ?>>, memref<1x512x128xf16, strided<[2097152, 4096, 1], offset: ?>>)
              outs(%alloc : memref<1x64x128xf16, #gpu.address_space<workgroup>>)
            linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%alloc : memref<1x64x128xf16, #gpu.address_space<workgroup>>)
              outs(%subview : memref<1x64x128xf32, strided<[16777216, 4096, 1], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            }
          }
        }
        return
      }
    }
  }
}

// PROMOTEC-LABEL: func.func @batch_matmul_f16_1x64x128x512()

//  PROMOTEC-DAG: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<1x64x32xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x128xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[C_ALLOC:.+]] = memref.alloc() {alignment = 128 : i64} : memref<1x64x128xf16, #gpu.address_space<workgroup>>

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[C_ALLOC]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c512 step %c32 {
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c1, %c64, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c128]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.batch_matmul
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[C_ALLOC]]
//      PROMOTEC: }
//      PROMOTEC: gpu.barrier
//      PROMOTEC: linalg.generic
//      PROMOTEC:    ins(%[[C_ALLOC]]
// PROMOTEC-SAME:   __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC: gpu.barrier

// -----

// Broadcasted elementwise ops does not need promoting C matrix.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 128], [32, 64], [0, 0, 32], [16, 16, 16]]>

hal.executable @matmul_f16_f512x4096x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_NV_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}> {
    hal.executable.export public @matmul_f16_f512x4096x64 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>,
      workgroup_size = [128 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @matmul_f16_f512x4096x64() {
        %c512 = arith.constant 512 : index
        %c4096 = arith.constant 4096 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<512x64xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<64x4096xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<4096xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<512x4096xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
        scf.for %arg0 = %4 to %c512 step %5 {
          %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
          %7 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
          scf.for %arg1 = %6 to %c4096 step %7 {
            %subview = memref.subview %3[%arg0, %arg1] [64, 128] [1, 1] : memref<512x4096xf16> to memref<64x128xf16, strided<[4096, 1], offset: ?>>
            %subview_0 = memref.subview %0[%arg0, 0] [64, 64] [1, 1] : memref<512x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            %subview_1 = memref.subview %1[0, %arg1] [64, 128] [1, 1] : memref<64x4096xf16> to memref<64x128xf16, strided<[4096, 1], offset: ?>>
            linalg.fill ins(%cst : f16) outs(%subview : memref<64x128xf16, strided<[4096, 1], offset: ?>>)
            linalg.matmul {lowering_config = #config}
              ins(%subview_0, %subview_1 : memref<64x64xf16, strided<[64, 1], offset: ?>>, memref<64x128xf16, strided<[4096, 1], offset: ?>>)
              outs(%subview : memref<64x128xf16, strided<[4096, 1], offset: ?>>)
            %subview_2 = memref.subview %2[%arg1] [128] [1] : memref<4096xf16> to memref<128xf16, strided<[1], offset: ?>>
            linalg.generic {
                indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel"]}
              ins(%subview_2 : memref<128xf16, strided<[1], offset: ?>>) outs(%subview : memref<64x128xf16, strided<[4096, 1], offset: ?>>) {
            ^bb0(%in: f16, %out: f16):
              %8 = arith.addf %out, %in : f16
              linalg.yield %8 : f16
            }
          }
        }
        return
      }
    }
  }
}

// PROMOTEC-LABEL: func.func @matmul_f16_f512x4096x64()

//  PROMOTEC-NOT: memref.alloc()
//  PROMOTEC-DAG: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<64x32xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x128xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-NOT: memref.alloc()

//      PROMOTEC: %[[SPAN2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//      PROMOTEC: %[[SPAN3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//      PROMOTEC: %[[OUT_VIEW:.+]] = memref.subview %[[SPAN3]]

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[OUT_VIEW]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c64 step %c32 {
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0] [%c64, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0] [%c32, %c128]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.matmul
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[OUT_VIEW]]
//      PROMOTEC: }

//  PROMOTEC-NOT: gpu.barrier
//  PROMOTEC-NOT: memref.copy
//      PROMOTEC: %[[BCAST_VIEW:.+]] = memref.subview %[[SPAN2]][%{{.+}}] [128] [1]
//      PROMOTEC: linalg.generic
// PROMOTEC-SAME:    ins(%[[BCAST_VIEW]]
// PROMOTEC-SAME:    outs(%[[OUT_VIEW]]
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"

// -----

// Transposed+broadcasted elementwise ops does not need promoting C matrix.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 128], [32, 64], [0, 0, 32], [16, 16, 16]]>

hal.executable @matmul_f16_f512x4096x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_NV_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}> {
    hal.executable.export public @matmul_f16_f512x4096x64 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>,
      workgroup_size = [128 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @matmul_f16_f512x4096x64() {
        %c512 = arith.constant 512 : index
        %c4096 = arith.constant 4096 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<512x64xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<64x4096xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<512xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<512x4096xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
        scf.for %arg0 = %4 to %c512 step %5 {
          %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
          %7 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
          scf.for %arg1 = %6 to %c4096 step %7 {
            %subview = memref.subview %3[%arg0, %arg1] [64, 128] [1, 1] : memref<512x4096xf16> to memref<64x128xf16, strided<[4096, 1], offset: ?>>
            %subview_0 = memref.subview %0[%arg0, 0] [64, 64] [1, 1] : memref<512x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            %subview_1 = memref.subview %1[0, %arg1] [64, 128] [1, 1] : memref<64x4096xf16> to memref<64x128xf16, strided<[4096, 1], offset: ?>>
            linalg.fill ins(%cst : f16) outs(%subview : memref<64x128xf16, strided<[4096, 1], offset: ?>>)
            linalg.matmul {lowering_config = #config}
              ins(%subview_0, %subview_1 : memref<64x64xf16, strided<[64, 1], offset: ?>>, memref<64x128xf16, strided<[4096, 1], offset: ?>>)
              outs(%subview : memref<64x128xf16, strided<[4096, 1], offset: ?>>)
            %subview_2 = memref.subview %2[%arg0] [64] [1] : memref<512xf16> to memref<64xf16, strided<[1], offset: ?>>
            linalg.generic {
                  indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
              ins(%subview_2 : memref<64xf16, strided<[1], offset: ?>>)
              outs(%subview : memref<64x128xf16, strided<[4096, 1], offset: ?>>) {
            ^bb0(%in: f16, %out: f16):
              %8 = arith.addf %out, %in : f16
              linalg.yield %8 : f16
            }
          }
        }
        return
      }
    }
  }
}

// PROMOTEC-LABEL: func.func @matmul_f16_f512x4096x64()

//  PROMOTEC-NOT: memref.alloc()
//  PROMOTEC-DAG: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<64x32xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x128xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-NOT: memref.alloc()

//      PROMOTEC: %[[SPAN2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//      PROMOTEC: %[[SPAN3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//      PROMOTEC: %[[OUT_VIEW:.+]] = memref.subview %[[SPAN3]]

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[OUT_VIEW]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c64 step %c32 {
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0] [%c64, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0] [%c32, %c128]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.matmul
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[OUT_VIEW]]
//      PROMOTEC: }

//  PROMOTEC-NOT: gpu.barrier
//  PROMOTEC-NOT: memref.copy
//      PROMOTEC: %[[BCAST_VIEW:.+]] = memref.subview %[[SPAN2]][%{{.+}}] [64] [1]
//      PROMOTEC: linalg.generic
// PROMOTEC-SAME:    ins(%[[BCAST_VIEW]]
// PROMOTEC-SAME:    outs(%[[OUT_VIEW]]
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"

// -----

// Inlined large constant array needs promoting C matrix.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 128], [32, 64], [0, 0, 32], [16, 16, 16]]>

hal.executable @matmul_f16_128x262144x2304 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_NV_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}> {
    hal.executable.export public @matmul_f16_128x262144x2304 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>,
      workgroup_size = [128 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @matmul_f16_128x262144x2304() {
        %c128 = arith.constant 128 : index
        %c262144 = arith.constant 262144 : index
        %c96565312 = arith.constant 96565312 : index
        %c806357120 = arith.constant 806357120 : index
        %c134217728 = arith.constant 134217728 : index
        %cst = arith.constant 0.000000e+00 : f16
        %cst_0 = arith.constant dense<"0x69222B2E40A3002A45AC1AAB2E2E202DA21C212680264C2A102314A041A7D029CB28352E5BAAD3B02F299D9A142B8AA1D1285C28412B25AF9A24EE2BA22C242D53AD9E2948A9289FCF301D28012F08AD68A6DD20ECAC912465290B2E9420C5AA50A222A912AB9526B62ADA2039AD4D912C9FDD287B20B224D329BA2A4D2C41A76DAB7E30B027F62ED1A0F1273A2BAE9D0FA48029812992A65AA92A2C9C2EE9A744A4632C5FA8A9A4CF2D70A482A0F5A2DBA7B6304B9D22A52B1B9DA8E424722AB5ACD0248A2B8B29C82D782E402D1A99F0A60CA4DE2DD32815266F2A6B247FA6FE214E2853AA402390AB6925F1A339307F2664A23CACBE28BA2B3D286DB0BA2E"> : tensor<128xf16>
        %0 = bufferization.to_memref %cst_0 : memref<128xf16>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c96565312)  : memref<128x2304xf16>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c806357120) : memref<2304x262144xf16>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c134217728) : memref<128x262144xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
        scf.for %arg0 = %4 to %c128 step %5 {
          %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
          %7 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
          scf.for %arg1 = %6 to %c262144 step %7 {
            %subview = memref.subview %3[%arg0, %arg1] [64, 128] [1, 1] : memref<128x262144xf16> to memref<64x128xf16, strided<[262144, 1], offset: ?>>
            %subview_1 = memref.subview %1[%arg0, 0] [64, 2304] [1, 1] : memref<128x2304xf16> to memref<64x2304xf16, strided<[2304, 1], offset: ?>>
            %subview_2 = memref.subview %2[0, %arg1] [2304, 128] [1, 1] : memref<2304x262144xf16> to memref<2304x128xf16, strided<[262144, 1], offset: ?>>
            linalg.fill ins(%cst : f16) outs(%subview : memref<64x128xf16, strided<[262144, 1], offset: ?>>)
            linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128], [32, 64], [0, 0, 32], [16, 16, 16]]>}
              ins(%subview_1, %subview_2 : memref<64x2304xf16, strided<[2304, 1], offset: ?>>, memref<2304x128xf16, strided<[262144, 1], offset: ?>>)
              outs(%subview : memref<64x128xf16, strided<[262144, 1], offset: ?>>)
            %subview_3 = memref.subview %0[%arg0] [64] [1] : memref<128xf16> to memref<64xf16, strided<[1], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
              ins(%subview_3 : memref<64xf16, strided<[1], offset: ?>>) outs(%subview : memref<64x128xf16, strided<[262144, 1], offset: ?>>) {
            ^bb0(%in: f16, %out: f16):
              %8 = arith.addf %out, %in : f16
              linalg.yield %8 : f16
            }
          }
        }
        return
      }
    }
  }
}

// PROMOTEC-LABEL: func.func @matmul_f16_128x262144x2304()

//  PROMOTEC-DAG: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<64x32xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x128xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[C_ALLOC:.+]] = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[C_ALLOC]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c2304 step %c32 {
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0] [%c64, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0] [%c32, %c128]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.matmul
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[C_ALLOC]]
//      PROMOTEC: }
//      PROMOTEC: gpu.barrier
//      PROMOTEC: linalg.generic
//      PROMOTEC:    ins(%{{.+}}, %[[C_ALLOC]]
// PROMOTEC-SAME:   __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC: gpu.barrier
