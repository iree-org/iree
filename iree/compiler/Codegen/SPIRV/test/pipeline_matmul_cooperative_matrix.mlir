// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline))" %s | IreeFileCheck %s
// TODO(#5608): iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline))" -iree-spirv-use-workgroup-memory %s | IreeFileCheck %s

hal.executable @matmul_cooperative_matrix attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @matmul_cooperative_matrix attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {
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
           subgroup_size = 32 : i32}>} {
      func @matmul_cooperative_matrix() {
        %c32 = constant 32 : index
        %c4096 = constant 4096 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<4096x4096xf16>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<4096x4096xf16>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<4096x4096xf16>
        %3 = hal.interface.workgroup.id[0] : index
        %4 = hal.interface.workgroup.id[1] : index
        scf.for %arg0 = %c0 to %c4096 step %c32 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%4]
          %6 = memref.subview %0[%5, %arg0] [64, 32] [1, 1] : memref<4096x4096xf16> to memref<64x32xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          %7 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%3]
          %8 = memref.subview %1[%arg0, %7] [32, 64] [1, 1] : memref<4096x4096xf16> to memref<32x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          %9 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%4]
          %10 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%3]
          %11 = memref.subview %2[%9, %10] [64, 64] [1, 1] : memref<4096x4096xf16> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
          linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%6, %8 : memref<64x32xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>, memref<32x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>) outs(%11 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>)
        }
        return

      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//    CHECK-LABEL: spv.func @matmul_cooperative_matrix
// CHECK-COUNT-16:   spv.CooperativeMatrixLoadNV
//          CHECK:   spv.mlir.loop
// CHECK-COUNT-16:     spv.CooperativeMatrixLoadNV
// CHECK-COUNT-32:     spv.CooperativeMatrixMulAddNV
// CHECK-COUNT-16:     spv.CooperativeMatrixStoreNV
