// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-to-cooperative-ops, iree-spirv-vectorize-to-cooperative-ops, canonicalize, cse)))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 32], [16, 16], [0, 0, 32], [16, 16, 16]]>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x128_div_add {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
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
    hal.executable.export public @matmul_256x1024x128_div_add layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg0]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg1]
      hal.return %0, %1, %c1 : index, index, index
    }
    builtin.module  {
      func.func @matmul_256x1024x128_div_add() {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %0 = gpu.thread_id  x
      %1 = gpu.thread_id  y
      %2 = gpu.thread_id  z
      %alloc = memref.alloc() : memref<32x32xf16, 3>
      %alloc_0 = memref.alloc() : memref<32x32xf16, 3>
      %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<256x1024xf16>
      %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<1024x128xf16>
      %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<256x128xf16>
      %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<256x128xf16>
      %7 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : memref<256x128xf16>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %8 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
      %9 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
      %subview = memref.subview %7[%8, %9] [32, 32] [1, 1] : memref<256x128xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
      %subview_1 = memref.subview %3[%8, 0] [32, 1024] [1, 1] : memref<256x1024xf16> to memref<32x1024xf16, strided<[1024, 1], offset: ?>>
      %subview_2 = memref.subview %4[0, %9] [1024, 32] [1, 1] : memref<1024x128xf16> to memref<1024x32xf16, strided<[128, 1], offset: ?>>
      linalg.fill {__internal_linalg_transform__ = "workgroup_memory"} ins(%cst : f16) outs(%subview : memref<32x32xf16, strided<[128, 1], offset: ?>>)
      scf.for %arg0 = %c0 to %c1024 step %c32 {
        %subview_5 = memref.subview %subview_1[0, %arg0] [32, 32] [1, 1] : memref<32x1024xf16, strided<[1024, 1], offset: ?>> to memref<32x32xf16, strided<[1024, 1], offset: ?>>
        %subview_6 = memref.subview %subview_2[%arg0, 0] [32, 32] [1, 1] : memref<1024x32xf16, strided<[128, 1], offset: ?>> to memref<32x32xf16, strided<[128, 1], offset: ?>>
        gpu.barrier
        %subview_7 = memref.subview %alloc[%c0, %c0] [32, 32] [1, 1] : memref<32x32xf16, 3> to memref<32x32xf16, strided<[32, 1], offset: ?>, 3>
        %10 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %11 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %12 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %13 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %subview_8 = memref.subview %subview_5[%10, %11] [1, 8] [1, 1] : memref<32x32xf16, strided<[1024, 1], offset: ?>> to memref<1x8xf16, strided<[1024, 1], offset: ?>>
        %subview_9 = memref.subview %subview_7[%12, %13] [1, 8] [1, 1] : memref<32x32xf16, strided<[32, 1], offset: ?>, 3> to memref<1x8xf16, strided<[32, 1], offset: ?>, 3>
        %14 = vector.transfer_read %subview_8[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x8xf16, strided<[1024, 1], offset: ?>>, vector<1x8xf16>
        vector.transfer_write %14, %subview_9[%c0, %c0] {in_bounds = [true, true]} : vector<1x8xf16>, memref<1x8xf16, strided<[32, 1], offset: ?>, 3>
        %subview_10 = memref.subview %alloc_0[%c0, %c0] [32, 32] [1, 1] : memref<32x32xf16, 3> to memref<32x32xf16, strided<[32, 1], offset: ?>, 3>
        %15 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %16 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %17 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %18 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %subview_11 = memref.subview %subview_6[%15, %16] [1, 8] [1, 1] : memref<32x32xf16, strided<[128, 1], offset: ?>> to memref<1x8xf16, strided<[128, 1], offset: ?>>
        %subview_12 = memref.subview %subview_10[%17, %18] [1, 8] [1, 1] : memref<32x32xf16, strided<[32, 1], offset: ?>, 3> to memref<1x8xf16, strided<[32, 1], offset: ?>, 3>
        %19 = vector.transfer_read %subview_11[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x8xf16, strided<[128, 1], offset: ?>>, vector<1x8xf16>
        vector.transfer_write %19, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x8xf16>, memref<1x8xf16, strided<[32, 1], offset: ?>, 3>
        gpu.barrier
        linalg.matmul {__internal_linalg_transform__ = "workgroup_memory", lowering_config = #config}
          ins(%alloc, %alloc_0 : memref<32x32xf16, 3>, memref<32x32xf16, 3>) outs(%subview : memref<32x32xf16, strided<[128, 1], offset: ?>>)
      }
      %subview_3 = memref.subview %5[%8, %9] [32, 32] [1, 1] : memref<256x128xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
      %subview_4 = memref.subview %6[%8, %9] [32, 32] [1, 1] : memref<256x128xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
      linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%subview_3, %subview_4 : memref<32x32xf16, strided<[128, 1], offset: ?>>, memref<32x32xf16, strided<[128, 1], offset: ?>>)
      outs(%subview : memref<32x32xf16, strided<[128, 1], offset: ?>>)
      attrs =  {__internal_linalg_transform__ = "workgroup_memory"} {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %10 = arith.divf %out, %in : f16
        %11 = arith.addf %10, %in_5 : f16
        linalg.yield %11 : f16
      }
      return
      }
    }
  }
}

//       CHECK: #[[$MAP_Y:.+]] = affine_map<()[s0] -> (s0 * 16)>
//       CHECK: #[[$MAP_X:.+]] = affine_map<()[s0] -> ((s0 floordiv 32) * 16)>

// CHECK-LABEL: func.func @matmul_256x1024x128_div_add()

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[C1024:.+]] = arith.constant 1024 : index
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf16>

//   CHECK-DAG:   %[[ID_X:.+]] = gpu.thread_id  x
//   CHECK-DAG:   %[[ID_Y:.+]] = gpu.thread_id  y

//   CHECK-DAG:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x32xf16, 3>
//   CHECK-DAG:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x32xf16, 3>

//       CHECK:   %[[OFFSET_Y:.+]] = affine.apply #[[$MAP_Y]]()[%[[ID_Y]]]
//       CHECK:   %[[OFFSET_X:.+]] = affine.apply #[[$MAP_X]]()[%[[ID_X]]]

//       CHECK:   scf.for %{{.+}} = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_X]] to %[[C32]] step %[[C32]]
//       CHECK:       vector.transfer_write %[[ZERO]], {{.+}} : vector<16x16xf16>, memref<16x16xf16, strided<[128, 1], offset: ?>>
//       CHECK:   scf.for %{{.+}} = %[[C0]] to %[[C1024]] step %[[C32]]
//       CHECK:     gpu.barrier
//       CHECK:     vector.transfer_read {{.+}} vector<1x8xf16>
//       CHECK:     vector.transfer_write
//       CHECK:     vector.transfer_read {{.+}} vector<1x8xf16>
//       CHECK:     vector.transfer_write
//       CHECK:     gpu.barrier
//       CHECK:     scf.for %[[IV_Y:.+]] = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:       %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][%[[IV_Y]], 0]
//       CHECK:       %[[READ0:.+]] = vector.transfer_read %[[LHS_VIEW]][%[[C0]], %[[C0]]]
//       CHECK:       %[[READ1:.+]] = vector.transfer_read %[[LHS_VIEW]][%[[C0]], %[[C16]]]
//       CHECK:       scf.for %[[IV_X:.+]] = %[[OFFSET_X]] to %[[C32]] step %[[C32]]
//       CHECK:         %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, %[[IV_X]]]
//       CHECK:         %[[READ2:.+]] = vector.transfer_read %[[RHS_VIEW]][%[[C0]], %[[C0]]]
//       CHECK:         %[[READ3:.+]] = vector.transfer_read %[[RHS_VIEW]][%[[C16]], %[[C0]]]
//       CHECK:         %[[READ4:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]]]
//       CHECK:         %[[CT0:.+]] = vector.contract
//  CHECK-SAME:           %[[READ0]], %[[READ2]], %[[READ4]] : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
//       CHECK:         %[[CT1:.+]] = vector.contract
//  CHECK-SAME:           %[[READ1]], %[[READ3]], %[[CT0]] : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
//       CHECK:         vector.transfer_write %[[CT1]], %{{.+}}[%[[C0]], %[[C0]]]
//       CHECK:   scf.for %{{.+}} = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_X]] to %[[C32]] step %[[C32]]
//       CHECK:       %[[READ5:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]]]
//       CHECK:       %[[READ6:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]]]
//       CHECK:       %[[READ7:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]]]
//       CHECK:       %[[DIV:.+]] = arith.divf %[[READ7]], %[[READ5]] : vector<16x16xf16>
//       CHECK:       %[[ADD:.+]] = arith.addf %[[DIV]], %[[READ6]] : vector<16x16xf16>
//       CHECK:       vector.transfer_write %[[ADD]], %{{.+}}[%[[C0]], %[[C0]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32], [1, 16, 16], [0, 0, 0, 32], [1, 16, 16, 16]]>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x128_div_add {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
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
    hal.executable.export public @matmul_256x1024x128_div_add layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg0]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg1]
      hal.return %0, %1, %c1 : index, index, index
    }
    builtin.module  {
      func.func @matmul_256x1024x128_div_add() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c512 = arith.constant 512 : index
        %c1 = arith.constant 1 : index
        %0 = gpu.thread_id  x
        %1 = gpu.thread_id  y
        %2 = gpu.thread_id  z
        %alloc = memref.alloc() : memref<1x32x32xf16, 3>
        %alloc_0 = memref.alloc() : memref<1x32x32xf16, 3>
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<16x128x512xf16>
        memref.assume_alignment %3, 64 : memref<16x128x512xf16>
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<16x512x256xf16>
        memref.assume_alignment %4, 64 : memref<16x512x256xf16>
        %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x128x256xf16>
        memref.assume_alignment %5, 64 : memref<16x128x256xf16>
        %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<16x128x256xf16>
        memref.assume_alignment %6, 64 : memref<16x128x256xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
        %8 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        %subview = memref.subview %6[%workgroup_id_z, %7, %8] [1, 32, 32] [1, 1, 1] : memref<16x128x256xf16> to memref<1x32x32xf16, strided<[32768, 256, 1], offset: ?>>
        %subview_1 = memref.subview %3[%workgroup_id_z, %7, 0] [1, 32, 512] [1, 1, 1] : memref<16x128x512xf16> to memref<1x32x512xf16, strided<[65536, 512, 1], offset: ?>>
        %subview_2 = memref.subview %4[%workgroup_id_z, 0, %8] [1, 512, 32] [1, 1, 1] : memref<16x512x256xf16> to memref<1x512x32xf16, strided<[131072, 256, 1], offset: ?>>
        linalg.fill {__internal_linalg_transform__ = "workgroup_memory"}
          ins(%cst : f16) outs(%subview : memref<1x32x32xf16, strided<[32768, 256, 1], offset: ?>>)
        scf.for %arg0 = %c0 to %c512 step %c32 {
          %subview_4 = memref.subview %subview_1[0, 0, %arg0] [1, 32, 32] [1, 1, 1] : memref<1x32x512xf16, strided<[65536, 512, 1], offset: ?>> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
          %subview_5 = memref.subview %subview_2[0, %arg0, 0] [1, 32, 32] [1, 1, 1] : memref<1x512x32xf16, strided<[131072, 256, 1], offset: ?>> to memref<1x32x32xf16, strided<[131072, 256, 1], offset: ?>>
          gpu.barrier
          %subview_6 = memref.subview %alloc[%c0, %c0, %c0] [1, 32, 32] [1, 1, 1] : memref<1x32x32xf16, 3> to memref<1x32x32xf16, strided<[1024, 32, 1], offset: ?>, 3>
          %9 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
          %10 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
          %11 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
          %12 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
          %subview_7 = memref.subview %subview_4[0, %9, %10] [1, 1, 8] [1, 1, 1] : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>> to memref<1x1x8xf16, strided<[65536, 512, 1], offset: ?>>
          %subview_8 = memref.subview %subview_6[0, %11, %12] [1, 1, 8] [1, 1, 1] : memref<1x32x32xf16, strided<[1024, 32, 1], offset: ?>, 3> to memref<1x1x8xf16, strided<[1024, 32, 1], offset: ?>, 3>
          %13 = vector.transfer_read %subview_7[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8xf16, strided<[65536, 512, 1], offset: ?>>, vector<1x1x8xf16>
          vector.transfer_write %13, %subview_8[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x8xf16>, memref<1x1x8xf16, strided<[1024, 32, 1], offset: ?>, 3>
          %subview_9 = memref.subview %alloc_0[%c0, %c0, %c0] [1, 32, 32] [1, 1, 1] : memref<1x32x32xf16, 3> to memref<1x32x32xf16, strided<[1024, 32, 1], offset: ?>, 3>
          %14 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
          %15 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
          %16 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
          %17 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
          %subview_10 = memref.subview %subview_5[0, %14, %15] [1, 1, 8] [1, 1, 1] : memref<1x32x32xf16, strided<[131072, 256, 1], offset: ?>> to memref<1x1x8xf16, strided<[131072, 256, 1], offset: ?>>
          %subview_11 = memref.subview %subview_9[0, %16, %17] [1, 1, 8] [1, 1, 1] : memref<1x32x32xf16, strided<[1024, 32, 1], offset: ?>, 3> to memref<1x1x8xf16, strided<[1024, 32, 1], offset: ?>, 3>
          %18 = vector.transfer_read %subview_10[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8xf16, strided<[131072, 256, 1], offset: ?>>, vector<1x1x8xf16>
          vector.transfer_write %18, %subview_11[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x8xf16>, memref<1x1x8xf16, strided<[1024, 32, 1], offset: ?>, 3>
          gpu.barrier
          linalg.batch_matmul {__internal_linalg_transform__ = "workgroup_memory", lowering_config = #config}
            ins(%alloc, %alloc_0 : memref<1x32x32xf16, 3>, memref<1x32x32xf16, 3>) outs(%subview : memref<1x32x32xf16, strided<[32768, 256, 1], offset: ?>>)
        }
        %subview_3 = memref.subview %5[%workgroup_id_z, %7, %8] [1, 32, 32] [1, 1, 1] : memref<16x128x256xf16> to memref<1x32x32xf16, strided<[32768, 256, 1], offset: ?>>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%subview_3 : memref<1x32x32xf16, strided<[32768, 256, 1], offset: ?>>)
        outs(%subview : memref<1x32x32xf16, strided<[32768, 256, 1], offset: ?>>)
        attrs = {__internal_linalg_transform__ = "workgroup_memory"} {
        ^bb0(%in: f16, %out: f16):
          %9 = arith.divf %out, %in : f16
          linalg.yield %9 : f16
        }
        return
      }
    }
  }
}
//       CHECK: #[[$MAP_Y:.+]] = affine_map<()[s0] -> (s0 * 16)>
//       CHECK: #[[$MAP_X:.+]] = affine_map<()[s0] -> ((s0 floordiv 32) * 16)>

// CHECK-LABEL: func.func @matmul_256x1024x128_div_add()

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<1x16x16xf16>

//   CHECK-DAG:   %[[ID_X:.+]] = gpu.thread_id  x
//   CHECK-DAG:   %[[ID_Y:.+]] = gpu.thread_id  y
//   CHECK-DAG:   %[[ID_Z:.+]] = gpu.thread_id  z

//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, 3>
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, 3>

//       CHECK:   %[[OFFSET_Y:.+]] = affine.apply #[[$MAP_Y]]()[%[[ID_Y]]]
//       CHECK:   %[[OFFSET_X:.+]] = affine.apply #[[$MAP_X]]()[%[[ID_X]]]

//       CHECK:   scf.for %{{.+}} = %[[ID_Z]] to %[[C1]] step %[[C1]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:       scf.for %{{.+}} = %[[OFFSET_X]] to %[[C32]] step %[[C32]]
//       CHECK:         vector.transfer_write %[[ZERO]], {{.+}} : vector<1x16x16xf16>, memref<1x16x16xf16, strided<[32768, 256, 1], offset: ?>>

//       CHECK:   scf.for %{{.+}} = %[[C0]] to %[[C512]] step %[[C32]]
//       CHECK:     gpu.barrier
//       CHECK:     vector.transfer_read {{.+}} vector<1x1x8xf16>
//       CHECK:     vector.transfer_write
//       CHECK:     vector.transfer_read {{.+}} vector<1x1x8xf16>
//       CHECK:     vector.transfer_write
//       CHECK:     gpu.barrier
//       CHECK:     scf.for %[[IV_Z:.+]] = %[[ID_Z]] to %[[C1]] step %[[C1]]
//       CHECK:       scf.for %[[IV_Y:.+]] = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:         %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][%[[IV_Z]], %[[IV_Y]], 0] [1, 16, 32]
//       CHECK:         %[[READ0:.+]] = vector.transfer_read %[[LHS_VIEW]][%[[C0]], %[[C0]], %[[C0]]]
//       CHECK:         %[[READ1:.+]] = vector.transfer_read %[[LHS_VIEW]][%[[C0]], %[[C0]], %[[C16]]]
//       CHECK:         scf.for %[[IV_X:.+]] = %[[OFFSET_X]] to %[[C32]] step %[[C32]] {
//       CHECK:           %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][%[[IV_Z]], 0, %[[IV_X]]] [1, 32, 16]
//       CHECK:           %[[READ2:.+]] = vector.transfer_read %[[RHS_VIEW]][%[[C0]], %[[C0]], %[[C0]]]
//       CHECK:           %[[READ3:.+]] = vector.transfer_read %[[RHS_VIEW]][%[[C0]], %[[C16]], %[[C0]]]
//       CHECK:           %[[READ4:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]], %[[C0]]]
//       CHECK:           %[[CT0:.+]] = vector.contract
//  CHECK-SAME:             %[[READ0]], %[[READ2]], %[[READ4]] : vector<1x16x16xf16>, vector<1x16x16xf16> into vector<1x16x16xf16>
//       CHECK:           %[[CT1:.+]] = vector.contract
//  CHECK-SAME:             %[[READ1]], %[[READ3]], %[[CT0]] : vector<1x16x16xf16>, vector<1x16x16xf16> into vector<1x16x16xf16>
//       CHECK:           vector.transfer_write %[[CT1]], %{{.+}}[%[[C0]], %[[C0]], %[[C0]]]
//       CHECK:   scf.for %{{.+}} = %[[ID_Z]] to %[[C1]] step %[[C1]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:       scf.for %{{.+}} = %[[OFFSET_X]] to %[[C32]] step %[[C32]]
//       CHECK:         %[[READ5:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]], %[[C0]]]
//       CHECK:         %[[READ6:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]], %[[C0]]]
//       CHECK:         %[[DIV:.+]] = arith.divf %[[READ6]], %[[READ5]] : vector<1x16x16xf16>
//       CHECK:         vector.transfer_write %[[DIV]], %{{.+}}[%[[C0]], %[[C0]], %[[C0]]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 32], [16, 16], [0, 0, 32], [16, 16, 16]]>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x128_mixed_signedness_int8 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
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
    hal.executable.export public @matmul_256x1024x128_mixed_signedness_int8 layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg0]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%arg1]
      hal.return %0, %1, %c1 : index, index, index
    }
    builtin.module  {
      func.func @matmul_256x1024x128_mixed_signedness_int8() {
      %cst = arith.constant 0 : i32
      %cst_i8 = arith.constant 0 : i8
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %0 = gpu.thread_id  x
      %1 = gpu.thread_id  y
      %2 = gpu.thread_id  z
      %alloc = memref.alloc() : memref<32x32xi8, 3>
      %alloc_0 = memref.alloc() : memref<32x32xi8, 3>
      %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<256x1024xi8>
      %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<1024x128xi8>
      %7 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : memref<256x128xi32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %8 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
      %9 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
      %subview = memref.subview %7[%8, %9] [32, 32] [1, 1] : memref<256x128xi32> to memref<32x32xi32, strided<[128, 1], offset: ?>>
      %subview_1 = memref.subview %3[%8, 0] [32, 1024] [1, 1] : memref<256x1024xi8> to memref<32x1024xi8, strided<[1024, 1], offset: ?>>
      %subview_2 = memref.subview %4[0, %9] [1024, 32] [1, 1] : memref<1024x128xi8> to memref<1024x32xi8, strided<[128, 1], offset: ?>>
      linalg.fill {__internal_linalg_transform__ = "workgroup_memory"} ins(%cst : i32) outs(%subview : memref<32x32xi32, strided<[128, 1], offset: ?>>)
      scf.for %arg0 = %c0 to %c1024 step %c32 {
        %subview_5 = memref.subview %subview_1[0, %arg0] [32, 32] [1, 1] : memref<32x1024xi8, strided<[1024, 1], offset: ?>> to memref<32x32xi8, strided<[1024, 1], offset: ?>>
        %subview_6 = memref.subview %subview_2[%arg0, 0] [32, 32] [1, 1] : memref<1024x32xi8, strided<[128, 1], offset: ?>> to memref<32x32xi8, strided<[128, 1], offset: ?>>
        gpu.barrier
        %subview_7 = memref.subview %alloc[%c0, %c0] [32, 32] [1, 1] : memref<32x32xi8, 3> to memref<32x32xi8, strided<[32, 1], offset: ?>, 3>
        %10 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %11 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %12 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %13 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %subview_8 = memref.subview %subview_5[%10, %11] [1, 8] [1, 1] : memref<32x32xi8, strided<[1024, 1], offset: ?>> to memref<1x8xi8, strided<[1024, 1], offset: ?>>
        %subview_9 = memref.subview %subview_7[%12, %13] [1, 8] [1, 1] : memref<32x32xi8, strided<[32, 1], offset: ?>, 3> to memref<1x8xi8, strided<[32, 1], offset: ?>, 3>
        %14 = vector.transfer_read %subview_8[%c0, %c0], %cst_i8 {in_bounds = [true, true]} : memref<1x8xi8, strided<[1024, 1], offset: ?>>, vector<1x8xi8>
        vector.transfer_write %14, %subview_9[%c0, %c0] {in_bounds = [true, true]} : vector<1x8xi8>, memref<1x8xi8, strided<[32, 1], offset: ?>, 3>
        %subview_10 = memref.subview %alloc_0[%c0, %c0] [32, 32] [1, 1] : memref<32x32xi8, 3> to memref<32x32xi8, strided<[32, 1], offset: ?>, 3>
        %15 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %16 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %17 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]
        %18 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
        %subview_11 = memref.subview %subview_6[%15, %16] [1, 8] [1, 1] : memref<32x32xi8, strided<[128, 1], offset: ?>> to memref<1x8xi8, strided<[128, 1], offset: ?>>
        %subview_12 = memref.subview %subview_10[%17, %18] [1, 8] [1, 1] : memref<32x32xi8, strided<[32, 1], offset: ?>, 3> to memref<1x8xi8, strided<[32, 1], offset: ?>, 3>
        %19 = vector.transfer_read %subview_11[%c0, %c0], %cst_i8 {in_bounds = [true, true]} : memref<1x8xi8, strided<[128, 1], offset: ?>>, vector<1x8xi8>
        vector.transfer_write %19, %subview_12[%c0, %c0] {in_bounds = [true, true]} : vector<1x8xi8>, memref<1x8xi8, strided<[32, 1], offset: ?>, 3>
        gpu.barrier
        linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
        }
        ins(%alloc, %alloc_0 : memref<32x32xi8, 3>, memref<32x32xi8, 3>) outs(%subview : memref<32x32xi32, strided<[128, 1], offset: ?>>)
        attrs =  {__internal_linalg_transform__ = "workgroup_memory", lowering_config = #config} {
        ^bb0(%in: i8, %in_5: i8, %out: i32):
          %20 = arith.extui %in : i8 to i32
          %21 = arith.extsi %in_5 : i8 to i32
          %22 = arith.muli %20, %21 : i32
          %23 = arith.addi %22, %out : i32
          linalg.yield %23 : i32
        }
      }
      return
      }
    }
  }
}

//       CHECK: #[[$MAP_Y:.+]] = affine_map<()[s0] -> (s0 * 16)>
//       CHECK: #[[$MAP_X:.+]] = affine_map<()[s0] -> ((s0 floordiv 32) * 16)>

// CHECK-LABEL: func.func @matmul_256x1024x128_mixed_signedness_int8()

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[C1024:.+]] = arith.constant 1024 : index
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0> : vector<16x16xi32>

//   CHECK-DAG:   %[[ID_X:.+]] = gpu.thread_id  x
//   CHECK-DAG:   %[[ID_Y:.+]] = gpu.thread_id  y

//   CHECK-DAG:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x32xi8, 3>
//   CHECK-DAG:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x32xi8, 3>

//       CHECK:   %[[OFFSET_Y:.+]] = affine.apply #[[$MAP_Y]]()[%[[ID_Y]]]
//       CHECK:   %[[OFFSET_X:.+]] = affine.apply #[[$MAP_X]]()[%[[ID_X]]]

//       CHECK:   scf.for %{{.+}} = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_X]] to %[[C32]] step %[[C32]]
//       CHECK:       vector.transfer_write %[[ZERO]], {{.+}} : vector<16x16xi32>, memref<16x16xi32, strided<[128, 1], offset: ?>>
//       CHECK:   scf.for %{{.+}} = %[[C0]] to %[[C1024]] step %[[C32]]
//       CHECK:     gpu.barrier
//       CHECK:     vector.transfer_read {{.+}} vector<1x8xi8>
//       CHECK:     vector.transfer_write
//       CHECK:     vector.transfer_read {{.+}} vector<1x8xi8>
//       CHECK:     vector.transfer_write
//       CHECK:     gpu.barrier
//       CHECK:     scf.for %[[IV_Y:.+]] = %[[OFFSET_Y]] to %[[C32]] step %[[C32]]
//       CHECK:       %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][%[[IV_Y]], 0]
//       CHECK:       %[[READ0:.+]] = vector.transfer_read %[[LHS_VIEW]][%[[C0]], %[[C0]]]
//       CHECK:       %[[READ1:.+]] = vector.transfer_read %[[LHS_VIEW]][%[[C0]], %[[C16]]]
//       CHECK:       scf.for %[[IV_X:.+]] = %[[OFFSET_X]] to %[[C32]] step %[[C32]]
//       CHECK:         %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, %[[IV_X]]]
//       CHECK:         %[[READ2:.+]] = vector.transfer_read %[[RHS_VIEW]][%[[C0]], %[[C0]]]
//       CHECK:         %[[READ3:.+]] = vector.transfer_read %[[RHS_VIEW]][%[[C16]], %[[C0]]]
//       CHECK:         %[[READ4:.+]] = vector.transfer_read %{{.+}}[%[[C0]], %[[C0]]]
//       CHECK:         %[[EXTUI0:.+]] = arith.extui %[[READ0]] : vector<16x16xi8> to vector<16x16xi32>
//       CHECK:         %[[EXTUI1:.+]] = arith.extui %[[READ1]] : vector<16x16xi8> to vector<16x16xi32>
//       CHECK:         %[[EXTSI0:.+]] = arith.extsi %[[READ2]] : vector<16x16xi8> to vector<16x16xi32>
//       CHECK:         %[[EXTSI1:.+]] = arith.extsi %[[READ3]] : vector<16x16xi8> to vector<16x16xi32>
//       CHECK:         %[[CT0:.+]] = vector.contract
//  CHECK-SAME:           %[[EXTUI0]], %[[EXTSI0]], %[[READ4]] : vector<16x16xi32>, vector<16x16xi32> into vector<16x16xi32>
//       CHECK:         %[[CT1:.+]] = vector.contract
//  CHECK-SAME:           %[[EXTUI1]], %[[EXTSI1]], %[[CT0]] : vector<16x16xi32>, vector<16x16xi32> into vector<16x16xi32>
//       CHECK:         vector.transfer_write %[[CT1]], %{{.+}}[%[[C0]], %[[C0]]]
